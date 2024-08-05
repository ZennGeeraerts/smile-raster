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
        __global__ void VertexShaderKernel( const VertexShaderInput *devInput,
            VertexShaderOutput *devOutput,
            uint32_t vertexBufferCount,
            glm::mat4 viewProjection,
            glm::mat4 world );
    }
}