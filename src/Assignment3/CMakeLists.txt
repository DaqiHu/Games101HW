cmake_minimum_required(VERSION 3.10)
project(Rasterizer3)

set(OpenCV_DIR "${VCPKG_INSTALLED_DIR}/x64-windows/share/opencv2")
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)

include_directories(/usr/local/include ./include)

add_executable(Rasterizer3 main.cpp rasterizer.hpp rasterizer.cpp global.hpp Triangle.hpp Triangle.cpp Texture.hpp Texture.cpp Shader.hpp OBJ_Loader.h)
#target_compile_options(Rasterizer PUBLIC -Wall -Wextra -pedantic)

find_package(fmt CONFIG REQUIRED)
target_link_libraries(Rasterizer3 PRIVATE fmt::fmt-header-only)

set(OpenCV_DIR "${VCPKG_INSTALLED_DIR}/x64-windows/share/opencv2")
find_package(OpenCV REQUIRED)
target_link_libraries(Rasterizer3 PRIVATE ${OpenCV_LIBRARIES})

set(Eigen3_DIR "${VCPKG_INSTALLED_DIR}/x64-windows/share/eigen3")
find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(Rasterizer3 PRIVATE Eigen3::Eigen)
