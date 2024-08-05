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
        __device__ glm::vec3 Texture2DSample( const Texture2D &texture2D, const glm::vec2 &texCoord );
        __global__ void PixelShaderKernel( Framebuffer framebuffer, Texture2D albedoMap );
    }
}