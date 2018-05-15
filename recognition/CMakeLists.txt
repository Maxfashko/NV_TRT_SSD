
file(GLOB recognitionSources *.cpp)
file(GLOB recognitionIncludes *.h )

cuda_add_executable(recognition ${recognitionSources})
target_link_libraries(recognition nvcaffe_parser nvinfer)
