/*=============================================================================*/
// Copyright 2022-2023 Smile Raster
// Authors: Zenn Geeraerts
/*=============================================================================*/
#include "pixel_shader.cuh"

namespace smile
{
	namespace Raster
	{
		__device__ glm::vec3 Texture2DSample(const Texture2D& texture2D, const glm::vec2& texCoord)
		{
			uint32_t x = static_cast<uint32_t>(texCoord.x * texture2D.Width);
			uint32_t y = static_cast<uint32_t>(texCoord.y * texture2D.Height);
			uint32_t pixelIndex{ x + y * texture2D.Width };

			return glm::vec3{ texture2D.DevPixels[pixelIndex * 4], texture2D.DevPixels[pixelIndex * 4 + 1], texture2D.DevPixels[pixelIndex * 4 + 2] } / 255.f;
		}

		__global__ void PixelShaderKernel(Framebuffer framebuffer, Texture2D albedoMap)
		{
			uint32_t pixelX = (blockIdx.x * blockDim.x) + threadIdx.x;
			uint32_t pixelY = (blockIdx.y * blockDim.y) + threadIdx.y;
			uint32_t pixelIndex = pixelY * framebuffer.Width + pixelX;

			if (pixelIndex < (framebuffer.Width * framebuffer.Height))
			{
				if (framebuffer.DevDepthBuffer[pixelIndex] < FLT_MAX)
				{
					glm::vec3 lightDirection{ 0.577f, -0.577f, 0.577f };
					glm::vec3 color = glm::vec3{ 1.0f, 0.0f, 0.0f };

					float diffuseStrength = glm::dot(framebuffer.DevPixelData[pixelIndex].Normal, -lightDirection);
					diffuseStrength = diffuseStrength * 0.5f + 0.5f;
					diffuseStrength = glm::clamp(diffuseStrength, 0.f, 1.f);
					color *= diffuseStrength;

					framebuffer.DevColorBuffer[pixelIndex * framebuffer.ColorChannelCount] = color.b * 255.f;
					framebuffer.DevColorBuffer[pixelIndex * framebuffer.ColorChannelCount + 1] = color.g * 255.f;
					framebuffer.DevColorBuffer[pixelIndex * framebuffer.ColorChannelCount + 2] = color.r * 255.f;
				}
			}
		}
	}
}