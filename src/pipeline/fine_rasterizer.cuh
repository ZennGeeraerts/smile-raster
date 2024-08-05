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
        __global__ void FineRasterizerKernel( Bin *pBins,
            const Triangle *pTriangles,
            uint32_t binCountX,
            uint32_t binCountY,
            uint32_t binWidth,
            uint32_t binHeight,
            VertexShaderOutput *pPixelData,
            float *pDepthBuffer,
            uint32_t width );

        template < typename AttributeType >
        __device__ __host__ AttributeType InterpolateAttribute( const AttributeType &attribute0,
            const AttributeType &attribute1,
            const AttributeType &attribute2,
            float weight0,
            float weight1,
            float weight2,
            float w0,
            float w1,
            float w2,
            float wValue )
        {
            return ( ( attribute0 / w0 * weight0 ) + ( attribute1 / w1 * weight1 ) + ( attribute2 / w2 * weight2 ) ) *
                   wValue;
        }
    }
}