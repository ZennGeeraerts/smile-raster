#include "DataStructs.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

namespace Smile
{
	namespace Raster
	{
		template <typename AttributeType>
		__device__ __host__ AttributeType InterpolateAttribute(const AttributeType& attribute0, const AttributeType& attribute1, const AttributeType& attribute2, float weight0, float weight1, float weight2,
			float w0, float w1, float w2, float wValue)
		{
			return ((attribute0 / w0 * weight0) + (attribute1 / w1 * weight1) + (attribute2 / w2 * weight2)) * wValue;
		}

		__global__ void FineRasterizerKernel(Bin* pBins, const Triangle* pTriangles, uint32_t binCountX, uint32_t binCountY, uint32_t binWidth, uint32_t binHeight, VertexShaderOutput* pPixelData, float* pDepthBuffer, uint32_t width)
		{
			uint32_t binX = (blockIdx.x * blockDim.x) + threadIdx.x;
			uint32_t binY = (blockIdx.y * blockDim.y) + threadIdx.y;

			if ((binX < binCountX) && (binY < binCountY))
			{
				Bin& bin{ pBins[binY * binCountX + binX] };

				uint32_t minX = binX * binWidth;
				uint32_t minY = binY * binHeight;
				uint32_t maxX = minX + binWidth;
				uint32_t maxY = minY + binHeight;

				bin.QueueSize = min(bin.QueueSize, SMR_BIN_QUEUE_SIZE);

				for (uint32_t t{}; t < bin.QueueSize; ++t)
				{
					const Triangle& triangle{ pTriangles[bin.Queue[t]] };
					
					const glm::vec3 a{ triangle.Vertex1.Position - triangle.Vertex0.Position };
					const glm::vec3 b{ triangle.Vertex2.Position - triangle.Vertex1.Position };
					const glm::vec3 c{ triangle.Vertex0.Position - triangle.Vertex2.Position };

					for (uint32_t y{ minY }; y < maxY; ++y)
					{
						for (uint32_t x{ minX }; x < maxX; ++x)
						{
							uint32_t pixelIndex = y * width + x;
							glm::vec2 pixel{ x, y };

							// Get the vector from each vertex to the pixel
							const glm::vec2 ap{ pixel - glm::vec2{ triangle.Vertex0.Position } };
							const glm::vec2 bp{ pixel - glm::vec2{ triangle.Vertex1.Position } };
							const glm::vec2 cp{ pixel - glm::vec2{ triangle.Vertex2.Position } };

							// Get the cross product between each edge and the previous calculated vector
							const float crossA{ a.x * ap.y - a.y * ap.x };
							const float crossB{ b.x * bp.y - b.y * bp.x };
							const float crossC{ c.x * cp.y - c.y * cp.x };

							// Inside outside check
							if ((crossA >= 0) && (crossB >= 0) && (crossC >= 0))
							{
								const float area2{ a.x * b.y - a.y * b.x };
								const float weight0{ crossB / area2 };
								const float weight1{ crossC / area2 };
								const float weight2{ crossA / area2 };

								const float depthValue{ 1 / ((1 / triangle.Vertex0.Position.z * weight0) + (1 / triangle.Vertex1.Position.z * weight1) + (1 / triangle.Vertex2.Position.z * weight2)) };

								// Depth test
								if (depthValue < pDepthBuffer[pixelIndex])
								{
									pDepthBuffer[pixelIndex] = depthValue;

									const float wValue{ 1 / ((1 / triangle.Vertex0.Position.w * weight0) + (1 / triangle.Vertex1.Position.w * weight1) + (1 / triangle.Vertex2.Position.w * weight2)) };
									pPixelData[pixelIndex].Position = { pixel.x, pixel.y, depthValue, wValue };
									pPixelData[pixelIndex].Normal = InterpolateAttribute(triangle.Vertex0.Normal, triangle.Vertex1.Normal, triangle.Vertex2.Normal, weight0, weight1, weight2,
										triangle.Vertex0.Position.w, triangle.Vertex1.Position.w, triangle.Vertex2.Position.w, wValue);
									pPixelData[pixelIndex].TexCoord = InterpolateAttribute(triangle.Vertex0.TexCoord, triangle.Vertex1.TexCoord, triangle.Vertex2.TexCoord, weight0, weight1, weight2,
										triangle.Vertex0.Position.w, triangle.Vertex1.Position.w, triangle.Vertex2.Position.w, wValue);
									pPixelData[pixelIndex].Tangent = InterpolateAttribute(triangle.Vertex0.Tangent, triangle.Vertex1.Tangent, triangle.Vertex2.Tangent, weight0, weight1, weight2,
										triangle.Vertex0.Position.w, triangle.Vertex1.Position.w, triangle.Vertex2.Position.w, wValue);
								}
							}
						}
					}
				}

				bin.QueueSize = 0;
			}
		}
	}
}