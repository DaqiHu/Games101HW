cmake_minimum_required(VERSION 3.10)
project(Rasterizer)

set(CMAKE_CXX_STANDARD 17)

add_executable(Rasterizer main.cpp rasterizer.hpp rasterizer.cpp Triangle.hpp Triangle.cpp)

install(TARGETS Rasterizer DESTINATION bin)
install(DIRECTORY ${OpenCV_Dll_Debug_DIR} DESTINATION bin)

find_package(fmt CONFIG REQUIRED)
target_link_libraries(Rasterizer PRIVATE fmt::fmt-header-only)

# set(zlib_DIR "${VCPKG_INSTALLED_DIR}/x64-windows/share/zlib")
# find_package(zlib CONFIG REQUIRED)
# target_link_libraries(Rasterizer PRIVATE zlib::zlib)
set(OpenCV_DIR "${VCPKG_INSTALLED_DIR}/x64-windows/share/opencv2")

# set(quirc_DIR "${VCPKG_INSTALLED_DIR}/x64-windows/share/quirc")
find_package(OpenCV CONFIG REQUIRED)
target_link_libraries(Rasterizer PRIVATE ${OpenCV_LIBRARIES})

set(Eigen3_DIR "${VCPKG_INSTALLED_DIR}/x64-windows/share/eigen3")
find_package(Eigen3 CONFIG REQUIRED)
target_link_libraries(Rasterizer PRIVATE Eigen3::Eigen)

# Assuming VCPKG_ROOT is set as an environment variable
if(DEFINED ENV{VCPKG_ROOT})
    set(VCPKG_ROOT $ENV{VCPKG_ROOT})
else()
    set(VCPKG_ROOT "E:/GitHub/vcpkg")
endif()

set(DLL_PATH "${VCPKG_ROOT}/packages/opencv2_x64-windows/debug/bin")
set(VCPKG_INSTALLED_DLL_DIR "${VCPKG_INSTALLED_DIR}/x64-windows/bin")
# Copy DLLs to the output directory
add_custom_command(TARGET Rasterizer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${DLL_PATH}
    $<TARGET_FILE_DIR:Rasterizer>)

# Copy DLLs to the output directory
add_custom_command(TARGET Rasterizer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${VCPKG_INSTALLED_DLL_DIR}
    $<TARGET_FILE_DIR:Rasterizer>)
