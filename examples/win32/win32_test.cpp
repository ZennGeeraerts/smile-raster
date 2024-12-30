#include <device_context.cuh>
#include <Windows.h>
#include <DirectXColors.h>
#include <iostream>

#define SDL_MAIN_HANDLED
#include <SDL_image.h>

bool g_ShouldClose = false;

static LRESULT CALLBACK WindowsProcedureStatic(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	if (msg == WM_NCCREATE)
	{
		const CREATESTRUCTW* const pCreate = reinterpret_cast<CREATESTRUCTW*>(lParam);
		SetWindowLongPtr(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pCreate->lpCreateParams));
	}
	else
	{
		switch (msg)
		{
		case WM_ACTIVATE:
		{
			return 0;
		}

		case WM_CLOSE:
		{
			g_ShouldClose = true;
			DestroyWindow(hWnd);
			break;
		}

		case WM_DESTROY:
		{
			PostQuitMessage(0);
			break;
		}

		case WM_SIZE:
		{
			uint32_t width = LOWORD(lParam);
			uint32_t height = HIWORD(lParam);
			break;
		}

		default:
			return DefWindowProc(hWnd, msg, wParam, lParam);
		}

		return 0;
	}
	return DefWindowProc(hWnd, msg, wParam, lParam);
}

int main(int argc, char** argv)
{
	return WinMain(GetModuleHandle(NULL), NULL, NULL, SW_SHOW);
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR lpCmdLine, int nCmdShow)
{
	const char* className = "ExampleWindowClass";
	WNDCLASSEX windowClass = {};
	windowClass.cbSize = sizeof(WNDCLASSEX);
	windowClass.style = CS_HREDRAW | CS_VREDRAW;
	windowClass.cbClsExtra = 0;
	windowClass.cbWndExtra = 0;

	windowClass.hCursor = LoadCursor(nullptr, IDC_ARROW);
	windowClass.hbrBackground = (HBRUSH)GetStockObject(NULL_BRUSH);

	windowClass.lpszClassName = className;
	windowClass.lpszMenuName = nullptr;

	windowClass.hInstance = HINSTANCE();
	windowClass.lpfnWndProc = WindowsProcedureStatic;

	int success = RegisterClassEx(&windowClass);
	if (!success)
	{
		fprintf(stderr, "Could not register window class.");
		return 1;
	}

	const uint32_t width = 640;
	const uint32_t height = 480;

	RECT windowRect{};
	windowRect.left = 0;
	windowRect.right = width + windowRect.left;
	windowRect.top = 0;
	windowRect.bottom = height + windowRect.top;
	AdjustWindowRect(
		&windowRect, WS_OVERLAPPEDWINDOW | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX | WS_MAXIMIZEBOX, FALSE);

	const std::string windowTitle = "smile-raster win32 example";

	HWND windowHandle = CreateWindow(windowClass.lpszClassName,
		windowTitle.c_str(),
		WS_OVERLAPPEDWINDOW | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX | WS_MAXIMIZEBOX,
		CW_USEDEFAULT,
		CW_USEDEFAULT,
		windowRect.right - windowRect.left,
		windowRect.bottom - windowRect.top,
		nullptr,
		nullptr,
		HINSTANCE(),
		nullptr);

	if (!windowHandle)
	{
		fprintf(stderr, "Could not create window.");
		return 1;
	}

	ShowWindow(windowHandle, SW_SHOW);
	UpdateWindow(windowHandle);

	smile::Raster::RenderConfig config{};
	auto pContext = new smile::Raster::DeviceContext{ config };

	BITMAPINFO bitmapInfo{};
	uint8_t* pColorBuffer{};

	bitmapInfo.bmiHeader.biBitCount = sizeof(uint8_t) * 8 * 3;
	bitmapInfo.bmiHeader.biClrImportant = 0;
	bitmapInfo.bmiHeader.biClrUsed = 0;
	bitmapInfo.bmiHeader.biCompression = BI_RGB;
	bitmapInfo.bmiHeader.biWidth = width;
	bitmapInfo.bmiHeader.biHeight = -static_cast<int>(height);
	bitmapInfo.bmiHeader.biPlanes = 1;
	bitmapInfo.bmiHeader.biSize = sizeof(BITMAPINFO);
	bitmapInfo.bmiHeader.biSizeImage = width * height * 3;
	bitmapInfo.bmiHeader.biXPelsPerMeter = 0;
	bitmapInfo.bmiHeader.biYPelsPerMeter = 0;

	HDC hDC = GetDC(windowHandle);
	HDC compHDC = CreateCompatibleDC(hDC);
	ReleaseDC(windowHandle, hDC);

	HBITMAP bitmap = CreateDIBSection(
		compHDC, &bitmapInfo, DIB_RGB_COLORS, reinterpret_cast<void**>(&pColorBuffer), NULL, 0);

	if (!bitmap)
	{
		fprintf(stderr, "Failed to create BitmapDIB");
		return 1;
	}

	HBITMAP bitmapOld = static_cast<HBITMAP>(SelectObject(compHDC, bitmap));

	memset(pColorBuffer, 0, sizeof(uint8_t) * width * height * 3);

	smile::Raster::BufferID framebuffer = pContext->CreateFramebuffer(
		pColorBuffer, width, height, smile::Raster::ColorbufferFormat::eRGB);
	pContext->BindFramebuffer(framebuffer);

	const uint32_t vertexCount = 3;
	float vertices[]{
		-0.5f, 0.0f, 0.5f,
		0.0f, 0.5f, 0.5f,
		0.5f, 0.0f, 0.5f
	};
	smile::Raster::BufferID vertexBuffer = pContext->CreateVertexBuffer(
		vertices, vertexCount, sizeof(float) * 3 * vertexCount
	);

	uint32_t indices[]{
		0, 1, 2
	};
	smile::Raster::BufferID indexBuffer = pContext->CreateIndexBuffer(
		indices, 3
	);

	SDL_Surface* pSurface = IMG_Load("data/uv_grid.png");
	if (!pSurface)
	{
		std::cout << SDL_GetError() << '\n';
		return 1;
	}

	smile::Raster::TextureID textureID = pContext->CreateTexture2D(static_cast<uint8_t*>(pSurface->pixels), pSurface->w, pSurface->h);

	while (!g_ShouldClose)
	{
		MSG message{};
		if (PeekMessage(&message, 0, 0, 0, PM_REMOVE))
		{
			TranslateMessage(&message);
			DispatchMessage(&message);
		}

		pContext->BindFramebuffer(framebuffer);
		pContext->Clear(framebuffer, { DirectX::Colors::DodgerBlue.f[0],
		DirectX::Colors::DodgerBlue.f[1],
		DirectX::Colors::DodgerBlue.f[2],
		DirectX::Colors::DodgerBlue.f[3] }, true);

		pContext->UploadTexture2D("AlbedoMap", textureID);
		pContext->BindVertexBuffer(vertexBuffer, 12);
		pContext->BindIndexBuffer(indexBuffer);
		pContext->DrawIndexed(3);

		// Present
		HDC hDC = GetDC(windowHandle);
		BitBlt(hDC, 0, 0, width, height, compHDC, 0, 0, SRCCOPY);
		ReleaseDC(windowHandle, hDC);
	}

	SDL_FreeSurface(pSurface);

	SelectObject(compHDC, bitmapOld);
	DeleteObject(bitmapOld);

	DeleteObject(bitmap);
	DeleteDC(compHDC);

	DestroyWindow(windowHandle);
	UnregisterClass(windowClass.lpszClassName, windowClass.hInstance);

	return 0;
}