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
        __global__ void InputAssemblerKernel( const void *devVertices,
            uint32_t vertexBufferCount,
            VertexShaderInput *devVertexShaderInput,
            uint32_t vertexStride );
    }
}