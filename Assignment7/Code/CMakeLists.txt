cmake_minimum_required(VERSION 3.10)
project(RayTracing3)

set(CMAKE_CXX_STANDARD 17)

add_executable(RayTracing3 main.cpp Object.hpp Vector.cpp Vector.hpp Sphere.hpp global.hpp Triangle.hpp Scene.cpp
        Scene.hpp Light.hpp AreaLight.hpp BVH.cpp BVH.hpp Bounds3.hpp Ray.hpp Material.hpp Intersection.hpp
        Renderer.cpp Renderer.hpp)

find_package(fmt CONFIG REQUIRED)
target_link_libraries(RayTracing3 PRIVATE fmt::fmt-header-only)

set(OpenCV_DIR "${VCPKG_INSTALLED_DIR}/x64-windows/share/opencv2")
find_package(OpenCV REQUIRED)
target_link_libraries(RayTracing3 PRIVATE ${OpenCV_LIBRARIES})

set(Eigen3_DIR "${VCPKG_INSTALLED_DIR}/x64-windows/share/eigen3")
find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(RayTracing3 PRIVATE Eigen3::Eigen)
