cmake_minimum_required(VERSION 3.5)
SET(PROJECT_NAME TensorRT_SSD)

SET( CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/)

PROJECT(${PROJECT_NAME})
SET(MODULENAME ${PROJECT_NAME}_core)

SET(CMAKE_CXX_STANDARD 14)

ADD_DEFINITIONS(-DOPENCV)
ADD_DEFINITIONS(-DGPU)

FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)
INCLUDE_DIRECTORIES("${CUDA_INCLUDE_DIRS}")

FILE(GLOB CUDA_SOURCE_FILES
        pluginImplement.h
        pluginImplement.cpp
        tensorNet.cpp
        tensorNet.h
        util/cuda/mathFunctions.h
        util/cuda/mathFunctions.cu
        util/cuda/mathFunctions.cpp
        util/cuda/cudaUtility.h
        util/cuda/cudaMappedMemory.h
        util/cuda/kernel.cu)

SET(TENSORRT_CUDA_LIBNAME ${MODULENAME}_cuda CACHE INTERNAL "${MODULENAME}: cuda library" FORCE)

CUDA_ADD_LIBRARY(
        ${TENSORRT_CUDA_LIBNAME} SHARED
        ${CUDA_SOURCE_FILES})

add_executable(tensorrt_ssd recognition/recognition.cpp)

TARGET_LINK_LIBRARIES(
        tensorrt_ssd
        ${OpenCV_LIBS}
        ${TENSORRT_CUDA_LIBNAME}
        ${CUDA_LIBRARIES}
        nvcaffe_parser
        nvinfer
        nvinfer_plugin
        glog)






