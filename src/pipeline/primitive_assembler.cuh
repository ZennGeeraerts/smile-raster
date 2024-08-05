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
        __global__ void PrimitiveAssemblerKernel( Triangle *devPrimitives,
            uint32_t primitiveCount,
            VertexShaderOutput *devVertexOutput,
            uint32_t *devIndices );
    }
}