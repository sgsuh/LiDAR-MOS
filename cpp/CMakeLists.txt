cmake_minimum_required(VERSION 3.10.0)
project(lmnet VERSION 0.1.0)

include(CTest)
enable_testing()

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")

# Only the common + io components are needed (pcl::PointCloud + pcd_io).
find_package(PCL REQUIRED COMPONENTS common io)
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenMP)

file(GLOB CPP_SOURCES "src/*.cpp")

include_directories(include 
                    ${PCL_INCLUDE_DIRS} 
                    ${TensorRT_INCLUDE_DIRS} 
                    ${CUDA_INCLUDE_DIRS} 
                    ${OpenCV_INCLUDE_DIRS} 
                    ${YAML_CPP_INCLUDE_DIRS})

add_executable(lmnet main.cpp ${CPP_SOURCES})

target_link_libraries(lmnet 
                      ${PCL_LIBRARIES} 
                      Eigen3::Eigen 
                      ${TensorRT_LIBRARIES} 
                      ${CUDA_LIBRARIES} 
                      ${OpenCV_LIBS}
                      ${YAML_CPP_LIBRARIES}
                      OpenMP::OpenMP_CXX)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
