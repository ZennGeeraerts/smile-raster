/*=============================================================================*/
// Copyright 2022-2024 Smile Raster
// Authors: Zenn Geeraerts
/*=============================================================================*/
#include "data_structs.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace smile
{
    namespace Raster
    {
        __global__ void BinRasterizerKernel( Triangle *devTriangles,
            uint32_t triangleCount,
            Bin *devBins,
            uint32_t binCountX,
            uint32_t binCountY,
            uint32_t binWidth,
            uint32_t binHeight,
            uint32_t width,
            uint32_t height );
        __device__ void FindAABB( const Triangle &triangle, glm::vec2 &minPoint, glm::vec2 &maxPoint );
    }
}