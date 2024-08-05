/*=============================================================================*/
// Copyright 2022-2023 Smile Raster
// Authors: Zenn Geeraerts
/*=============================================================================*/
#pragma once
#include <stdint.h>
#include <DirectXMath.h>
#include <glm.hpp>
#include <unordered_map>
#include <string>

namespace smile
{
	namespace Raster
	{
		struct VertexShaderInput final
		{
			glm::vec3 Position{};
			glm::vec3 Normal{};
			glm::vec2 TexCoord{};
			glm::vec3 Tangent{};
			glm::vec4 BlendIndices{};
			glm::vec4 BlendWeights{};
		};

		struct VertexShaderOutput final
		{
			glm::vec4 Position{};
			glm::vec3 Normal{};
			glm::vec2 TexCoord{};
			glm::vec3 Tangent{};
			glm::vec4 BlendIndices{};
			glm::vec4 BlendWeights{};
		};

		using InterpolatedAttributes = VertexShaderOutput;

		struct Texture2D final
		{
			uint8_t* DevPixels = nullptr;
			uint32_t Width{};
			uint32_t Height{};
		};

		struct Shader final
		{
			std::unordered_map<std::string, glm::mat4> Mat4Data{
				{ "ViewProjection", glm::mat4{ 1.0f } },
				{ "World", glm::mat4{ 1.0f } },
				{ "ViewInverse", glm::mat4{ 1.0f } } 
			};

			std::unordered_map<std::string, Texture2D> Texture2DData{ 
				{ "AlbedoMap", {} } 
			};
		};

		enum class ColorbufferFormat
		{
			eRGB,
			eRGBA
		};

		struct Framebuffer final
		{
			uint8_t* DevColorBuffer{ nullptr };
			float* DevDepthBuffer{ nullptr };
			InterpolatedAttributes* DevPixelData{ nullptr };

			uint32_t Width = 0;
			uint32_t Height = 0;
			uint8_t ColorChannelCount = 3;
			uint8_t* pHostOutput{ nullptr };
		};

		struct VertexBuffer final
		{
			void* DevVertices = nullptr;
			VertexShaderInput* DevVertexShaderInput = nullptr;
			VertexShaderOutput* DevVertexShaderOutput = nullptr;
			uint32_t ByteWidth = 0;
		};

		struct IndexBuffer final
		{
			uint32_t* DevIndices = nullptr;
		};

		struct Triangle final
		{
			union
			{
				VertexShaderOutput Vertices[3]{};
				struct
				{
					VertexShaderOutput Vertex0;
					VertexShaderOutput Vertex1;
					VertexShaderOutput Vertex2;
				};
			};

			glm::vec2 AABBMinPoint{};
			glm::vec2 AABBMaxPoint{};
		};

		struct Bin final
		{
			#define SMR_BIN_QUEUE_SIZE 512
			uint32_t Queue[SMR_BIN_QUEUE_SIZE]{};
			uint32_t QueueSize = 0;
		};
	}
}