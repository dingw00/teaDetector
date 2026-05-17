cmake_minimum_required(VERSION 3.16)
project(tea_detector_deploy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 按需修改：ONNX Runtime / OpenCV 安装路径
# set(ONNXRUNTIME_ROOT "C:/libs/onnxruntime")
# set(OpenCV_DIR "C:/libs/opencv/build")

find_package(OpenCV REQUIRED)
find_package(onnxruntime CONFIG QUIET)

if(NOT onnxruntime_FOUND)
  find_path(ONNXRUNTIME_INCLUDE onnxruntime_cxx_api.h
    HINTS "${ONNXRUNTIME_ROOT}/include")
  find_library(ONNXRUNTIME_LIB onnxruntime
    HINTS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib/x64")
  if(NOT ONNXRUNTIME_INCLUDE OR NOT ONNXRUNTIME_LIB)
    message(FATAL_ERROR "Set ONNXRUNTIME_ROOT or install onnxruntime CMake package")
  endif()
endif()

add_library(tea_deploy STATIC
  src/config.cpp
  src/preprocess.cpp
  src/postprocess.cpp
  src/detector.cpp
)
target_include_directories(tea_deploy PUBLIC include)

target_link_libraries(tea_deploy PUBLIC ${OpenCV_LIBS})

if(onnxruntime_FOUND)
  target_link_libraries(tea_deploy PUBLIC onnxruntime::onnxruntime)
else()
  target_include_directories(tea_deploy PUBLIC ${ONNXRUNTIME_INCLUDE})
  target_link_libraries(tea_deploy PUBLIC ${ONNXRUNTIME_LIB})
endif()

add_executable(tea_detector_demo example/main.cpp)
target_link_libraries(tea_detector_demo PRIVATE tea_deploy)
