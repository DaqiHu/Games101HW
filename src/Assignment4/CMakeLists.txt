cmake_minimum_required(VERSION 3.10)
project(BezierCurve)

set(OpenCV_DIR "${VCPKG_INSTALLED_DIR}/x64-windows/share/opencv2")
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 14)

add_executable(BezierCurve main.cpp)

find_package(fmt CONFIG REQUIRED)
target_link_libraries(BezierCurve PRIVATE fmt::fmt-header-only)

set(OpenCV_DIR "${VCPKG_INSTALLED_DIR}/x64-windows/share/opencv2")
find_package(OpenCV REQUIRED)
target_link_libraries(BezierCurve PRIVATE ${OpenCV_LIBRARIES})

set(Eigen3_DIR "${VCPKG_INSTALLED_DIR}/x64-windows/share/eigen3")
find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(BezierCurve PRIVATE Eigen3::Eigen)
