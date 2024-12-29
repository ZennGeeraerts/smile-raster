cmake_minimum_required(VERSION "3.25.0")
project(smile-raster LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

find_package(CUDAToolkit 12.3 REQUIRED)

file(GLOB_RECURSE SOURCE_FILES 
	src/*.cu
	src/*.cuh)

set(INCLUDE_DIRS
    src
	thirdparty/glm
)

add_library(${PROJECT_NAME} STATIC ${SOURCE_FILES})
set_target_properties(${PROJECT_NAME}
   PROPERTIES
       CUDA_SEPARABLE_COMPILATION ON
       CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDE_DIRS})

set_target_properties(${PROJECT_NAME}
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "50")
set_target_properties (${PROJECT_NAME} PROPERTIES FOLDER Dependencies)

target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)

add_subdirectory(examples/win32)