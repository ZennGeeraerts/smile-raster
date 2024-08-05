/*=============================================================================*/
// Copyright 2022-2023 Smile Raster
// Authors: Zenn Geeraerts
/*=============================================================================*/
#include "vertex_shader.cuh"

namespace smile
{
	namespace Raster
	{
		__global__ void VertexShaderKernel(const VertexShaderInput* devInput, VertexShaderOutput* devOutput, uint32_t vertexBufferCount, glm::mat4 viewProjection, glm::mat4 world)
		{
			uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

			if (index < vertexBufferCount)
			{
				const VertexShaderInput& input = devInput[index];
				VertexShaderOutput& output = devOutput[index];

				glm::mat4 worldViewProjectionMatrix = viewProjection * world;

				output.Position = worldViewProjectionMatrix * glm::vec4{ input.Position, 1.f };
				output.Normal = glm::normalize((glm::mat3)world * input.Normal);
				output.Tangent = input.Tangent;
				output.TexCoord = input.TexCoord;

				// Perspective divide
				output.Position.x /= output.Position.w;
				output.Position.y /= output.Position.w;
				output.Position.z /= output.Position.w;
			}
		}
	}
}