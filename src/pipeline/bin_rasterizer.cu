/*=============================================================================*/
// Copyright 2022-2023 Smile Raster
// Authors: Zenn Geeraerts
/*=============================================================================*/
#include "bin_rasterizer.cuh"
#include "bin_queue.cuh"

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
            uint32_t height )
        {
            uint32_t triangleIndex = blockIdx.x * blockDim.x + threadIdx.x;

            if ( triangleIndex < triangleCount )
            {
                Triangle &triangle = devTriangles[triangleIndex];

                for ( uint32_t i{}; i < 3; ++i )
                {
                    // Check if the triangle is outside the frustum
                    if ( ( triangle.Vertices[i].Position.x < -1.f ) || ( triangle.Vertices[i].Position.x > 1.f ) ||
                         ( triangle.Vertices[i].Position.y < -1.f ) || ( triangle.Vertices[i].Position.y > 1.f ) ||
                         ( triangle.Vertices[i].Position.z < 0.0f ) || ( triangle.Vertices[i].Position.z > 1.f ) )
                    {
                        return;
                    }
                    else
                    {
                        // Transform to raster space
                        triangle.Vertices[i].Position.x = ( triangle.Vertices[i].Position.x + 1.0f ) * 0.5f * width;
                        triangle.Vertices[i].Position.y = ( 1.0f - triangle.Vertices[i].Position.y ) * 0.5f * height;
                    }
                }

                FindAABB( triangle, triangle.AABBMinPoint, triangle.AABBMaxPoint );

                triangle.AABBMinPoint.x = max( triangle.AABBMinPoint.x, 0.0f );
                triangle.AABBMinPoint.y = max( triangle.AABBMinPoint.y, 0.0f );
                triangle.AABBMaxPoint.x = min( triangle.AABBMaxPoint.x, static_cast< float >( width ) );
                triangle.AABBMaxPoint.y = min( triangle.AABBMaxPoint.y, static_cast< float >( height ) );

                uint32_t minBinX = static_cast< uint32_t >( floor( triangle.AABBMinPoint.x / binWidth ) );
                uint32_t minBinY = static_cast< uint32_t >( floor( triangle.AABBMinPoint.y / binHeight ) );
                uint32_t maxBinX = static_cast< uint32_t >( ceil( triangle.AABBMaxPoint.x / binWidth ) );
                uint32_t maxBinY = static_cast< uint32_t >( ceil( triangle.AABBMaxPoint.y / binHeight ) );

                for ( uint32_t binY{ minBinY }; binY < maxBinY; ++binY )
                {
                    for ( uint32_t binX{ minBinX }; binX < maxBinX; ++binX )
                    {
                        Bin &bin{ devBins[binY * binCountX + binX] };
                        BinQueue::Push( bin, triangleIndex );
                    }
                }
            }
        }

        __device__ void FindAABB( const Triangle &triangle, glm::vec2 &minPoint, glm::vec2 &maxPoint )
        {
            minPoint.x =
                min( min( triangle.Vertex0.Position.x, triangle.Vertex1.Position.x ), triangle.Vertex2.Position.x );
            minPoint.y =
                min( min( triangle.Vertex0.Position.y, triangle.Vertex1.Position.y ), triangle.Vertex2.Position.y );

            maxPoint.x =
                max( max( triangle.Vertex0.Position.x, triangle.Vertex1.Position.x ), triangle.Vertex2.Position.x );
            maxPoint.y =
                max( max( triangle.Vertex0.Position.y, triangle.Vertex1.Position.y ), triangle.Vertex2.Position.y );
        }
    }
}