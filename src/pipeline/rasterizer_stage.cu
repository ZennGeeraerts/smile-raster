/*=============================================================================*/
// Copyright 2022-2023 Smile Raster
// Authors: Zenn Geeraerts
/*=============================================================================*/
#include "rasterizer_stage.cuh"
#include "bin_rasterizer.cuh"
#include "fine_rasterizer.cuh"

namespace smile
{
	namespace Raster
	{
		__global__ void RasterizerKernel(Triangle* devTriangles, uint32_t triangleCount, VertexShaderOutput* devPixelData, float* devDepthBuffer, uint32_t width, uint32_t height)
		{
			uint32_t triangleIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

			if (triangleIndex < triangleCount)
			{
				Triangle& triangle = devTriangles[triangleIndex];

				for (uint32_t i{}; i < 3; ++i)
				{
					// Check if the triangle is outside the frustum
					if ((triangle.Vertices[i].Position.x < -1.f) || (triangle.Vertices[i].Position.x > 1.f)
						|| (triangle.Vertices[i].Position.y < -1.f) || (triangle.Vertices[i].Position.y > 1.f)
						|| (triangle.Vertices[i].Position.z < 0.0f) || (triangle.Vertices[i].Position.z > 1.f))
					{
						return;
					}
					else
					{
						// Transform to raster space
						triangle.Vertices[i].Position.x = (triangle.Vertices[i].Position.x + 1.0f) * 0.5f * width;
						triangle.Vertices[i].Position.y = (1.0f - triangle.Vertices[i].Position.y) * 0.5f * height;
					}
				}

				glm::vec2 boundingBoxMin{};
				glm::vec2 boundingBoxMax{};
				FindAABB(triangle, boundingBoxMin, boundingBoxMax);

				boundingBoxMin.x = max(boundingBoxMin.x, 0.0f);
				boundingBoxMin.y = max(boundingBoxMin.y, 0.0f);
				boundingBoxMax.x = min(boundingBoxMax.x, static_cast<float>(width));
				boundingBoxMax.y = min(boundingBoxMax.y, static_cast<float>(height));

				for (uint32_t y = static_cast<uint32_t>(std::floor(boundingBoxMin.y)); y < static_cast<uint32_t>(std::ceil(boundingBoxMax.y)); ++y)
				{
					for (uint32_t x = static_cast<uint32_t>(std::floor(boundingBoxMin.x)); x < static_cast<uint32_t>(std::ceil(boundingBoxMax.x)); ++x)
					{
						uint32_t pixelIndex = y * width + x;
						glm::vec2 pixel{ x, y };

						// Calculate the edges of the triangle
						const glm::vec3 a{ triangle.Vertex1.Position - triangle.Vertex0.Position };
						const glm::vec3 b{ triangle.Vertex2.Position - triangle.Vertex1.Position };
						const glm::vec3 c{ triangle.Vertex0.Position - triangle.Vertex2.Position };

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
							const float area2{ a.x + b.y - a.y - b.x };
							const float weight0{ crossB / area2 };
							const float weight1{ crossC / area2 };
							const float weight2{ crossA / area2 };

							const float depthValue{ 1 / ((1 / triangle.Vertex0.Position.z * weight0) + (1 / triangle.Vertex1.Position.z * weight1) + (1 / triangle.Vertex2.Position.z * weight2)) };

							// Try to win the depth test
							atomicMin(reinterpret_cast<int*>(&devDepthBuffer[pixelIndex]), *(int*)(&depthValue));

							// Depth test
							if ( (*(int*)(&depthValue)) == (*reinterpret_cast<int*>(&devDepthBuffer[pixelIndex])))
							{
								devDepthBuffer[pixelIndex] = depthValue;

								const float wValue{ 1 / ((1 / triangle.Vertex0.Position.w * weight0) + (1 / triangle.Vertex1.Position.w * weight1) + (1 / triangle.Vertex2.Position.w * weight2)) };
								devPixelData[pixelIndex].Position = { pixel.x, pixel.y, depthValue, wValue };
								devPixelData[pixelIndex].Normal = InterpolateAttribute(triangle.Vertex0.Normal, triangle.Vertex1.Normal, triangle.Vertex2.Normal, weight0, weight1, weight2,
									triangle.Vertex0.Position.w, triangle.Vertex1.Position.w, triangle.Vertex2.Position.w, wValue);
								devPixelData[pixelIndex].TexCoord = InterpolateAttribute(triangle.Vertex0.TexCoord, triangle.Vertex1.TexCoord, triangle.Vertex2.TexCoord, weight0, weight1, weight2,
									triangle.Vertex0.Position.w, triangle.Vertex1.Position.w, triangle.Vertex2.Position.w, wValue);
								devPixelData[pixelIndex].Tangent = InterpolateAttribute(triangle.Vertex0.Tangent, triangle.Vertex1.Tangent, triangle.Vertex2.Tangent, weight0, weight1, weight2,
									triangle.Vertex0.Position.w, triangle.Vertex1.Position.w, triangle.Vertex2.Position.w, wValue);
							}
						}
					}
				}
			}
		}
	}
}