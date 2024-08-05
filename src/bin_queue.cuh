/*=============================================================================*/
// Copyright 2022-2023 Smile Raster
// Authors: Zenn Geeraerts
/*=============================================================================*/
#include "data_structs.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace smile
{
	namespace Raster
	{
		namespace BinQueue
		{
			__device__ inline void Push(Bin& bin, uint32_t triangleIndex)
			{
				int writeID = atomicAdd(&bin.QueueSize, 1);
				if (writeID < SMR_BIN_QUEUE_SIZE)
				{
					bin.Queue[writeID] = triangleIndex;
				}
			}

			__device__ inline void Clear(Bin& bin)
			{
				atomicExch(&bin.QueueSize, 0);
			}
		}
	}
}