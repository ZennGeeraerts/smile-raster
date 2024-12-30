cmake_minimum_required(VERSION "3.25.0")

project(smile-raster-win32-example LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 17)

if(WIN32)
    set(WIN32_LIBS Ws2_32 Winmm Version Bcrypt)
else()
    set(WIN32_LIBS)
endif()

add_executable(smile-raster-win32-example win32_test.cpp)
target_include_directories(smile-raster-win32-example PRIVATE
	${CMAKE_SOURCE_DIR}/smile-raster/src
	${CMAKE_SOURCE_DIR}/thirdparty/sdl2/include
	${CMAKE_SOURCE_DIR}/thirdparty/sdl2_image/include
)
target_link_libraries(smile-raster-win32-example PRIVATE 
	smile-raster
	${WIN32_LIBS}
	${CMAKE_SOURCE_DIR}/thirdparty/sdl2/x64/SDL2.lib
	${CMAKE_SOURCE_DIR}/thirdparty/sdl2_image/x64/SDL2_image.lib
)

add_custom_command(
    TARGET smile-raster-win32-example
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/data
        ${CMAKE_CURRENT_BINARY_DIR}/data
    ) 

add_custom_command(
    TARGET smile-raster-win32-example
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/thirdparty/sdl2/x64/SDL2.dll
        ${CMAKE_CURRENT_BINARY_DIR}
    )

add_custom_command(
    TARGET smile-raster-win32-example
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/thirdparty/sdl2_image/x64/SDL2_image.dll
        ${CMAKE_CURRENT_BINARY_DIR}
    )

add_custom_command(
    TARGET smile-raster-win32-example
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/thirdparty/sdl2_image/x64/libpng16-16.dll
        ${CMAKE_CURRENT_BINARY_DIR}
    )

add_custom_command(
    TARGET smile-raster-win32-example
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_SOURCE_DIR}/thirdparty/sdl2_image/x64/zlib1.dll
        ${CMAKE_CURRENT_BINARY_DIR}
    )

set_property(TARGET smile-raster-win32-example PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})