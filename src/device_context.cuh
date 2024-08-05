/*=============================================================================*/
// Copyright 2022-2023 Smile Raster
// Authors: Zenn Geeraerts
/*=============================================================================*/
#include "rasterizer.cuh"

namespace smile
{
    namespace Raster
    {
#define SMR_MAX_BUFFER_COUNT 10
#define SMR_INVALID_BUFFER_ID -1

#define SMR_MAX_TEXTURE_COUNT 7
#define SMR_INVALID_TEXTURE_ID -1

        typedef int BufferID;
        typedef int TextureID;

        class DeviceContext final
        {
          public:
            DeviceContext( const RenderConfig &renderCfg );
            ~DeviceContext();

            DeviceContext( const DeviceContext & ) = delete;
            DeviceContext( DeviceContext && ) = delete;
            DeviceContext &operator=( const DeviceContext & ) = delete;
            DeviceContext &operator=( const DeviceContext && ) = delete;

            BufferID
            CreateFramebuffer( uint8_t *pBuffer, uint32_t width, uint32_t height, ColorbufferFormat colorFormat );
            BufferID CreateVertexBuffer( void *pVertices, uint32_t count, uint32_t byteWidth );
            BufferID CreateIndexBuffer( uint32_t *pIndices, uint32_t count );
            TextureID CreateTexture2D( uint8_t *pPixels, uint32_t width, uint32_t height );

            bool BindFramebuffer( BufferID id );
            bool BindVertexBuffer( BufferID id, uint32_t stride );
            bool BindIndexBuffer( BufferID id );

            void Clear( BufferID framebufferID, const DirectX::XMFLOAT4 &clearColor, bool bClearDepth );
            void DrawIndexed( uint32_t indexCount );

            void Resize( BufferID framebufferID, uint32_t width, uint32_t height, uint8_t *pScreenBuffer );

            void UploadMat4( const std::string &sementicName, const DirectX::XMFLOAT4X4 &mat );
            void UploadTexture2D( const std::string &sementicName, TextureID texture );

          private:
            Rasterizer *m_pRasterizer = nullptr;

            Framebuffer m_Framebuffers[SMR_MAX_BUFFER_COUNT];
            VertexBuffer m_VertexBuffers[SMR_MAX_BUFFER_COUNT];
            IndexBuffer m_IndexBuffers[SMR_MAX_BUFFER_COUNT];
            Triangle *m_DevPrimitiveBuffers[SMR_MAX_BUFFER_COUNT];
            Texture2D m_Textures[SMR_MAX_TEXTURE_COUNT];

            uint32_t m_FramebufferCount = 0;
            uint32_t m_VertexBufferCount = 0;
            uint32_t m_IndexBufferCount = 0;
            uint32_t m_TextureCount = 0;
        };
    }
}