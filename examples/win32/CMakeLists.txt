cmake_minimum_required(VERSION "3.25.0")

project(smile-raster-win32-test LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 17)

if(WIN32)
    set(WIN32_LIBS Ws2_32 Winmm Version Bcrypt)
else()
    set(WIN32_LIBS)
endif()

add_executable(smile-raster-win32-test win32_test.cpp)
target_include_directories(smile-raster-win32-test PRIVATE
	${CMAKE_SOURCE_DIR}/smile-raster/src
)
target_link_libraries(smile-raster-win32-test PRIVATE 
	smile-raster
	${WIN32_LIBS}
)

set_property(TARGET smile-raster-win32-test PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")