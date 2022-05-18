#include <curand_kernel.h>
#include <stdio.h>
#include <stdint.h>
#include <DirectXMath.h>

namespace Smile
{
    namespace Raster
    {
		#define GPU_ERROR_CHECK(ans) { GPUAssert((ans), __FILE__, __LINE__); }
		inline void GPUAssert(cudaError_t error, const char* pFile, int line, bool bAbort = true)
		{
			if (error != cudaSuccess)
			{
				fprintf(stderr, "GPUAssert: %s %s %d\n", cudaGetErrorString(error), pFile, line);
				if (bAbort)
					exit(error);
			}
		}

		inline glm::mat4 ConvertToGLMMat(const DirectX::XMFLOAT4X4& mat)
		{
			glm::mat4 converted{};

			converted[0][0] = mat._11;
			converted[1][0] = mat._21;
			converted[2][0] = mat._31;
			converted[3][0] = mat._41;
			converted[0][1] = mat._12;
			converted[1][1] = mat._22;
			converted[2][1] = mat._32;
			converted[3][1] = mat._42;
			converted[0][2] = mat._13;
			converted[1][2] = mat._23;
			converted[2][2] = mat._33;
			converted[3][2] = mat._43;
			converted[0][3] = mat._14;
			converted[1][3] = mat._24;
			converted[2][3] = mat._34;
			converted[3][3] = mat._44;

			return converted;
		}
    }
}