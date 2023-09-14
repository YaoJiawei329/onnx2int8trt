
SET(INC_DIR 
	${CMAKE_SOURCE_DIR}/3rdparty/cuda-11.7/include
	${CMAKE_SOURCE_DIR}/3rdparty/cudnn-windows-x86_64-8.4.1.50_cuda11.6/include
	
	${CMAKE_SOURCE_DIR}/3rdparty/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8/include
	${CMAKE_SOURCE_DIR}/3rdparty/opencv-460/include
	${CMAKE_SOURCE_DIR}/3rdparty/opencv-460/include/opencv2
)
SET(LIB_DIR
	${CMAKE_SOURCE_DIR}/3rdparty/cuda-11.7/libs
	${CMAKE_SOURCE_DIR}/3rdparty/cudnn-windows-x86_64-8.4.1.50_cuda11.6/lib
	
	${CMAKE_SOURCE_DIR}/3rdparty/TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8/lib
	${CMAKE_SOURCE_DIR}/3rdparty/opencv-460/libs
)

SET(CUDA_LIBS cuda cublas cudart curand cufft)
SET(CUDNN_LIB cudnn64_8)
SET(OPENCV_LIBS opencv_world460)
SET(TRT_LIBS nvinfer nvinfer_plugin nvonnxparser nvparsers)

include_directories(${INC_DIR})
link_directories(${LIB_DIR})

ADD_EXECUTABLE(int8demo ${CMAKE_SOURCE_DIR}/Longi_popian_quant.cpp)
TARGET_LINK_LIBRARIES(int8demo ${CUDA_LIBS} ${TRT_LIBS} ${OPENCV_LIBS})