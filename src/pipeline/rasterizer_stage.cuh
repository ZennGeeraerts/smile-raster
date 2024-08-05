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
        __global__ void RasterizerKernel( Triangle *devTriangles,
            uint32_t triangleCount,
            VertexShaderOutput *devPixelData,
            float *devDepthBuffer,
            uint32_t width,
            uint32_t height );
    }
}