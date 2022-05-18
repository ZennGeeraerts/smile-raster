#include "DataStructs.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace Smile
{
	namespace Raster
	{
		__global__ void PrimitiveAssemblerKernel(Triangle* devPrimitives, uint32_t primitiveCount, VertexShaderOutput* devVertexOutput, uint32_t* devIndices)
		{
			uint32_t triangleIndex = (blockIdx.x * blockDim.x) + threadIdx.x;

			if (triangleIndex < primitiveCount)
			{
                uint32_t vertexIndex0 = devIndices[3 * triangleIndex];
                uint32_t vertexIndex1 = devIndices[3 * triangleIndex + 1];
                uint32_t vertexIndex2 = devIndices[3 * triangleIndex + 2];

				devPrimitives[triangleIndex].Vertices[0] = devVertexOutput[vertexIndex0];
				devPrimitives[triangleIndex].Vertices[1] = devVertexOutput[vertexIndex1];
				devPrimitives[triangleIndex].Vertices[2] = devVertexOutput[vertexIndex2];
			}
		}
	}
}