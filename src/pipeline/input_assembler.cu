/*=============================================================================*/
// Copyright 2022-2023 Smile Raster
// Authors: Zenn Geeraerts
/*=============================================================================*/
#include "input_assembler.cuh"

namespace smile
{
	namespace Raster
	{
		__global__ void InputAssemblerKernel(const void* devVertices, uint32_t vertexBufferCount, VertexShaderInput* devVertexShaderInput, uint32_t vertexStride)
		{
			uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

			if (index < vertexBufferCount)
			{
				const void* devDataLocation = (uint8_t*)devVertices + vertexStride * index;
				memcpy(&devVertexShaderInput[index], devDataLocation, min(vertexStride, static_cast<uint32_t>(sizeof(VertexShaderInput))));
			}
		}
	}
}