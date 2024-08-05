/*=============================================================================*/
// Copyright 2022-2023 Smile Raster
// Authors: Zenn Geeraerts
/*=============================================================================*/
#include "device_context.cuh"

#include "utils.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace smile
{
    namespace Raster
    {
        DeviceContext::DeviceContext( const RenderConfig &renderCfg ) : m_pRasterizer{ new Rasterizer{ renderCfg } }
        {
        }

        DeviceContext::~DeviceContext()
        {
            delete m_pRasterizer;

            for ( BufferID i{}; i < m_FramebufferCount; ++i )
            {
                GPU_ERROR_CHECK( cudaFree( m_Framebuffers[i].DevColorBuffer ) );
                GPU_ERROR_CHECK( cudaFree( m_Framebuffers[i].DevDepthBuffer ) );
                GPU_ERROR_CHECK( cudaFree( m_Framebuffers[i].DevPixelData ) );
            }

            for ( BufferID i{}; i < m_VertexBufferCount; ++i )
            {
                GPU_ERROR_CHECK( cudaFree( m_VertexBuffers[i].DevVertices ) );
                GPU_ERROR_CHECK( cudaFree( m_VertexBuffers[i].DevVertexShaderInput ) );
                GPU_ERROR_CHECK( cudaFree( m_VertexBuffers[i].DevVertexShaderOutput ) );
            }

            for ( BufferID i{}; i < m_IndexBufferCount; ++i )
            {
                GPU_ERROR_CHECK( cudaFree( m_IndexBuffers[i].DevIndices ) );
                GPU_ERROR_CHECK( cudaFree( m_DevPrimitiveBuffers[i] ) );
            }
        }

        // Buffer creation
        BufferID DeviceContext::CreateFramebuffer( uint8_t *pBuffer,
            uint32_t width,
            uint32_t height,
            ColorbufferFormat colorFormat )
        {
            if ( m_FramebufferCount < SMR_MAX_BUFFER_COUNT )
            {
                Framebuffer &framebuffer = m_Framebuffers[m_FramebufferCount];
                switch ( colorFormat )
                {
                    case ColorbufferFormat::eRGB:
                        framebuffer.ColorChannelCount = 3;
                        break;

                    case ColorbufferFormat::eRGBA:
                        framebuffer.ColorChannelCount = 4;
                        break;
                }

                framebuffer.Width = width;
                framebuffer.Height = height;
                framebuffer.pHostOutput = pBuffer;

                size_t size = sizeof( uint8_t ) * framebuffer.ColorChannelCount * width * height;
                GPU_ERROR_CHECK( cudaMalloc( &framebuffer.DevColorBuffer, size ) );

                size = sizeof( float ) * width * height;
                GPU_ERROR_CHECK( cudaMalloc( &framebuffer.DevDepthBuffer, size ) );

                size = sizeof( InterpolatedAttributes ) * width * height;
                GPU_ERROR_CHECK( cudaMalloc( &framebuffer.DevPixelData, size ) );

                ++m_FramebufferCount;
                return m_FramebufferCount - 1;
            }

            return SMR_INVALID_BUFFER_ID;
        }

        BufferID DeviceContext::CreateVertexBuffer( void *pVertices, uint32_t count, uint32_t byteWidth )
        {
            if ( m_VertexBufferCount < SMR_MAX_BUFFER_COUNT )
            {
                VertexBuffer &vertexBuffer = m_VertexBuffers[m_VertexBufferCount];

                GPU_ERROR_CHECK( cudaMalloc( &vertexBuffer.DevVertices, byteWidth ) );
                GPU_ERROR_CHECK( cudaMemcpy( vertexBuffer.DevVertices, pVertices, byteWidth, cudaMemcpyHostToDevice ) );
                vertexBuffer.ByteWidth = byteWidth;

                size_t size = sizeof( VertexShaderInput ) * count;
                GPU_ERROR_CHECK( cudaMalloc( &vertexBuffer.DevVertexShaderInput, size ) );
                size = sizeof( VertexShaderOutput ) * count;
                GPU_ERROR_CHECK( cudaMalloc( &vertexBuffer.DevVertexShaderOutput, size ) );

                ++m_VertexBufferCount;

                return m_VertexBufferCount - 1;
            }

            return SMR_INVALID_BUFFER_ID;
        }

        BufferID DeviceContext::CreateIndexBuffer( uint32_t *pIndices, uint32_t count )
        {
            if ( m_IndexBufferCount < SMR_MAX_BUFFER_COUNT )
            {
                size_t size = sizeof( uint32_t ) * count;
                GPU_ERROR_CHECK( cudaMalloc( &m_IndexBuffers[m_IndexBufferCount].DevIndices, size ) );
                GPU_ERROR_CHECK( cudaMemcpy(
                    m_IndexBuffers[m_IndexBufferCount].DevIndices, pIndices, size, cudaMemcpyHostToDevice ) );

                size = sizeof( Triangle ) * count / 3;
                GPU_ERROR_CHECK( cudaMalloc( &m_DevPrimitiveBuffers[m_IndexBufferCount], size ) );

                ++m_IndexBufferCount;

                return m_IndexBufferCount - 1;
            }

            return SMR_INVALID_BUFFER_ID;
        }

        TextureID DeviceContext::CreateTexture2D( uint8_t *pPixels, uint32_t width, uint32_t height )
        {
            if ( m_TextureCount < SMR_MAX_TEXTURE_COUNT )
            {
                Texture2D &texture = m_Textures[m_TextureCount];
                texture.Width = width;
                texture.Height = height;

                size_t size = sizeof( uint8_t ) * 4 * width * height;
                GPU_ERROR_CHECK( cudaMalloc( &texture.DevPixels, size ) );
                GPU_ERROR_CHECK( cudaMemcpy( texture.DevPixels, pPixels, size, cudaMemcpyHostToDevice ) );

                ++m_TextureCount;
                return m_TextureCount - 1;
            }

            return SMR_INVALID_TEXTURE_ID;
        }

        bool DeviceContext::BindFramebuffer( BufferID id )
        {
            if ( ( id < static_cast< int >( m_FramebufferCount ) ) && ( id > SMR_INVALID_BUFFER_ID ) )
            {
                m_pRasterizer->SetFramebuffer( &m_Framebuffers[id] );
                return true;
            }

            m_pRasterizer->m_pFramebuffer = nullptr;
            return false;
        }

        bool DeviceContext::BindVertexBuffer( BufferID id, uint32_t stride )
        {
            if ( ( id < static_cast< int >( m_VertexBufferCount ) ) && ( id > SMR_INVALID_BUFFER_ID ) )
            {
                m_pRasterizer->m_pVertexBuffer = &m_VertexBuffers[id];
                m_pRasterizer->m_VertexStride = stride;
                return true;
            }

            m_pRasterizer->m_pVertexBuffer = nullptr;
            return false;
        }

        bool DeviceContext::BindIndexBuffer( BufferID id )
        {
            if ( ( id < static_cast< int >( m_IndexBufferCount ) ) && ( id > SMR_INVALID_BUFFER_ID ) )
            {
                m_pRasterizer->m_pIndexBuffer = &m_IndexBuffers[id];
                m_pRasterizer->m_DevPrimitiveBuffer = m_DevPrimitiveBuffers[id];
                return true;
            }

            m_pRasterizer->m_pIndexBuffer = nullptr;
            return false;
        }

        // Clearing
        __global__ void
        ClearFramebufferKernel( Framebuffer framebuffer, DirectX::XMFLOAT4 clearColor, bool bClearDepth )
        {
            uint32_t pixelX = ( blockIdx.x * blockDim.x ) + threadIdx.x;
            uint32_t pixelY = ( blockIdx.y * blockDim.y ) + threadIdx.y;
            uint32_t bufferIndex = pixelY * framebuffer.Width + pixelX;

            if ( bufferIndex < ( framebuffer.Width * framebuffer.Height ) )
            {
                framebuffer.DevColorBuffer[bufferIndex * framebuffer.ColorChannelCount] =
                    static_cast< uint8_t >( clearColor.z * 255.f );
                framebuffer.DevColorBuffer[bufferIndex * framebuffer.ColorChannelCount + 1] =
                    static_cast< uint8_t >( clearColor.y * 255.f );
                framebuffer.DevColorBuffer[bufferIndex * framebuffer.ColorChannelCount + 2] =
                    static_cast< uint8_t >( clearColor.x * 255.f );

                if ( framebuffer.ColorChannelCount > 3 )
                    framebuffer.DevColorBuffer[bufferIndex * framebuffer.ColorChannelCount + 3] =
                        static_cast< uint8_t >( clearColor.w * 255.f );

                if ( bClearDepth )
                    framebuffer.DevDepthBuffer[bufferIndex] = FLT_MAX;
            }
        }

        void DeviceContext::Clear( BufferID framebufferID, const DirectX::XMFLOAT4 &clearColor, bool bClearDepth )
        {
            Framebuffer &framebuffer = m_Framebuffers[framebufferID];

            dim3 blockSize = { m_pRasterizer->m_RenderConfig.BlockSize, m_pRasterizer->m_RenderConfig.BlockSize };
            dim3 gridSize = {
                static_cast< uint32_t >(
                    ceil( framebuffer.Width / static_cast< float >( m_pRasterizer->m_RenderConfig.BlockSize ) ) ),
                static_cast< uint32_t >(
                    ceil( framebuffer.Height / static_cast< float >( m_pRasterizer->m_RenderConfig.BlockSize ) ) ) };

            ClearFramebufferKernel<<<gridSize, blockSize>>>( framebuffer, clearColor, bClearDepth );
        }

        // Rendering
        void DeviceContext::DrawIndexed( uint32_t indexCount )
        {
            assert( indexCount % 3 == 0 );

            m_pRasterizer->Draw( indexCount / 3 );
        }

        void DeviceContext::Resize( BufferID framebufferID, uint32_t width, uint32_t height, uint8_t *pScreenBuffer )
        {
            if ( ( framebufferID >= static_cast< int >( m_FramebufferCount ) ) ||
                 ( framebufferID <= SMR_INVALID_BUFFER_ID ) )
                return;

            Framebuffer &framebuffer = m_Framebuffers[framebufferID];

            GPU_ERROR_CHECK( cudaFree( framebuffer.DevColorBuffer ) );
            GPU_ERROR_CHECK( cudaFree( framebuffer.DevDepthBuffer ) );
            GPU_ERROR_CHECK( cudaFree( framebuffer.DevPixelData ) );

            framebuffer.Width = width;
            framebuffer.Height = height;
            framebuffer.pHostOutput = pScreenBuffer;

            size_t size = sizeof( uint8_t ) * framebuffer.ColorChannelCount * width * height;
            GPU_ERROR_CHECK( cudaMalloc( &framebuffer.DevColorBuffer, size ) );

            size = sizeof( float ) * width * height;
            GPU_ERROR_CHECK( cudaMalloc( &framebuffer.DevDepthBuffer, size ) );

            size = sizeof( InterpolatedAttributes ) * width * height;
            GPU_ERROR_CHECK( cudaMalloc( &framebuffer.DevPixelData, size ) );

            m_pRasterizer->m_BinWidth = { static_cast< uint32_t >(
                ceil( static_cast< float >( width ) / m_pRasterizer->m_RenderConfig.BinSizeX ) ) };
            m_pRasterizer->m_BinHeight = { static_cast< uint32_t >(
                ceil( static_cast< float >( height ) / m_pRasterizer->m_RenderConfig.BinSizeY ) ) };
        }

        void DeviceContext::UploadMat4( const std::string &sementicName, const DirectX::XMFLOAT4X4 &mat )
        {
            auto it = m_pRasterizer->m_Shader.Mat4Data.find( sementicName );
            if ( it != m_pRasterizer->m_Shader.Mat4Data.end() )
                ( *it ).second = ConvertToGLMMat( mat );
        }

        void DeviceContext::UploadTexture2D( const std::string &sementicName, TextureID texture )
        {
            auto it = m_pRasterizer->m_Shader.Texture2DData.find( sementicName );
            if ( ( it != m_pRasterizer->m_Shader.Texture2DData.end() ) && ( texture > SMR_INVALID_TEXTURE_ID ) &&
                 ( texture < static_cast< int >( m_TextureCount ) ) )
            {
                ( *it ).second = m_Textures[texture];
            }
        }
    }
}