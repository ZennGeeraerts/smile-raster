#include "Rasterizer.cuh"

#include "Utils.cuh"

// Pipeline
#include "Pipeline/InputAssembler.cu"
#include "Pipeline/VertexShader.cu"
#include "Pipeline/PrimitiveAssembler.cu"
#include "Pipeline/RasterizerStage.cu"
#include "Pipeline/PixelShader.cu"

#include <iostream>

namespace Smile
{
	namespace Raster
	{
		Rasterizer::Rasterizer(const RenderConfig& renderCfg)
			: m_RenderConfig{ renderCfg }
		{
			size_t size{ sizeof(Bin) * renderCfg.BinSizeX * renderCfg.BinSizeY };
			GPU_ERROR_CHECK(cudaMalloc(&m_DevBins, size));
		}

		Rasterizer::~Rasterizer()
		{
			GPU_ERROR_CHECK(cudaFree(m_DevBins));
		}

		void Rasterizer::SetFramebuffer(Framebuffer* pFramebuffer)
		{
			if (!m_pFramebuffer || ((m_pFramebuffer->Width != pFramebuffer->Width) || (m_pFramebuffer->Height != pFramebuffer->Height)))
			{
				m_BinWidth = { static_cast<uint32_t>(ceil(static_cast<float>(pFramebuffer->Width) / m_RenderConfig.BinSizeX)) };
				m_BinHeight = { static_cast<uint32_t>(ceil(static_cast<float>(pFramebuffer->Height) / m_RenderConfig.BinSizeY)) };
			}

			m_pFramebuffer = pFramebuffer;
		}

		void Rasterizer::Draw(uint32_t primitiveCount)
		{
			assert(m_pFramebuffer != nullptr);
			assert(m_pVertexBuffer != nullptr);
			assert(m_pIndexBuffer != nullptr);

			uint32_t vertexCount = static_cast<uint32_t>(ceil(m_pVertexBuffer->ByteWidth / static_cast<float>(m_VertexStride)));
			dim3 blockSize = { m_RenderConfig.BlockSize, m_RenderConfig.BlockSize };

			// Input assembler
			dim3 gridSize = static_cast<uint32_t>(ceil(vertexCount / static_cast<float>(m_RenderConfig.BlockSize)));
			InputAssemblerKernel << <gridSize, blockSize >> > (m_pVertexBuffer->DevVertices, vertexCount, m_pVertexBuffer->DevVertexShaderInput, m_VertexStride);
			GPU_ERROR_CHECK(cudaDeviceSynchronize());

			// VertexShader
			VertexShaderKernel << <gridSize, blockSize >> > (m_pVertexBuffer->DevVertexShaderInput, m_pVertexBuffer->DevVertexShaderOutput, vertexCount, m_Shader.Mat4Data["ViewProjection"], m_Shader.Mat4Data["World"]);
			GPU_ERROR_CHECK(cudaDeviceSynchronize());

			// Primitive assembler
			gridSize = static_cast<uint32_t>(ceil(primitiveCount / static_cast<float>(m_RenderConfig.BlockSize)));
			PrimitiveAssemblerKernel << <gridSize, blockSize >> > (m_DevPrimitiveBuffer, primitiveCount, m_pVertexBuffer->DevVertexShaderOutput, m_pIndexBuffer->DevIndices);
			GPU_ERROR_CHECK(cudaDeviceSynchronize());

			if (m_RenderConfig.RasterTech == RasterizerTechnique::eObjectSpace)
			{
				// Rasterizer
				RasterizerKernel<<<gridSize, blockSize>>>(m_DevPrimitiveBuffer, primitiveCount, m_pFramebuffer->DevPixelData, m_pFramebuffer->DevDepthBuffer, m_pFramebuffer->Width, m_pFramebuffer->Height);
				GPU_ERROR_CHECK(cudaDeviceSynchronize());
			}
			else if (m_RenderConfig.RasterTech == RasterizerTechnique::eScreenSpace)
			{
				// Bin Rasterizer
				BinRasterizerKernel << <gridSize, blockSize >> > (m_DevPrimitiveBuffer, primitiveCount, m_DevBins, m_RenderConfig.BinSizeX, m_RenderConfig.BinSizeY,
					m_BinWidth, m_BinHeight, m_pFramebuffer->Width, m_pFramebuffer->Height);
				GPU_ERROR_CHECK(cudaDeviceSynchronize());

				// Fine Rasterizer
				gridSize = { static_cast<uint32_t>(ceil(m_RenderConfig.BinSizeX / static_cast<float>(m_RenderConfig.BlockSize))),
					static_cast<uint32_t>(ceil(m_RenderConfig.BinSizeY / static_cast<float>(m_RenderConfig.BlockSize))) };
				FineRasterizerKernel << <gridSize, blockSize >> > (m_DevBins, m_DevPrimitiveBuffer, m_RenderConfig.BinSizeX, m_RenderConfig.BinSizeY, m_BinWidth, m_BinHeight, m_pFramebuffer->DevPixelData, m_pFramebuffer->DevDepthBuffer, m_pFramebuffer->Width);
				GPU_ERROR_CHECK(cudaDeviceSynchronize());
			}

			// Pixel Shader
			gridSize = { static_cast<uint32_t>(ceil(m_pFramebuffer->Width / static_cast<float>(m_RenderConfig.BlockSize))),
								static_cast<uint32_t>(ceil(m_pFramebuffer->Height / static_cast<float>(m_RenderConfig.BlockSize))) };
			PixelShaderKernel << <gridSize, blockSize >> > (*m_pFramebuffer, m_Shader.Texture2DData["AlbedoMap"]);
			GPU_ERROR_CHECK(cudaDeviceSynchronize());

			// Copy data to host buffer
			size_t size = sizeof(uint8_t) * m_pFramebuffer->ColorChannelCount * m_pFramebuffer->Width * m_pFramebuffer->Height;
			GPU_ERROR_CHECK(cudaMemcpy(m_pFramebuffer->pHostOutput, m_pFramebuffer->DevColorBuffer, size, cudaMemcpyDeviceToHost));
		}
	}
}