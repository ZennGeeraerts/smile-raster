#include "DataStructs.cuh"

namespace Smile
{
	namespace Raster
	{
		enum class RasterizerTechnique
		{
			eObjectSpace,
			eScreenSpace
		};

		struct RenderConfig final
		{
			uint32_t BlockSize = 2;
			uint32_t BinSizeX = 512;
			uint32_t BinSizeY = 256;
			RasterizerTechnique RasterTech = RasterizerTechnique::eScreenSpace;
		};

		class Rasterizer final
		{
		public:
			Rasterizer(const RenderConfig& renderCfg);
			~Rasterizer();

			void SetFramebuffer(Framebuffer* pFramebuffer);
			void Draw(uint32_t primitiveCount);

		private:
			RenderConfig m_RenderConfig;

			Framebuffer* m_pFramebuffer = nullptr;
			VertexBuffer* m_pVertexBuffer = nullptr;
			uint32_t m_VertexStride = 0;
			IndexBuffer* m_pIndexBuffer = nullptr;
			Triangle* m_DevPrimitiveBuffer = nullptr;
			Shader m_Shader{};

			Bin* m_DevBins = nullptr;
			uint32_t m_BinWidth{};
			uint32_t m_BinHeight{};

			friend class DeviceContext;
		};
	}
}