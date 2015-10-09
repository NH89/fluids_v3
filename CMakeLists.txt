
cmake_minimum_required(VERSION 2.8)

find_package(CUDA)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_20,code=sm_20)
set(CUDA_SEPARABLE_COMPILATION ON)
set(BUILD_SHARED_LIBS OFF)

#add_definitions(-g)
#add_definitions(-w)

include_directories(.)
include_directories(./common)
include_directories(./fluids)

#list all source files here
cuda_add_executable(fluids_v3
		            main.cpp
		            common/camera3d.cpp
		            common/geomx.cpp
		            common/gl_helper.cpp
		            common/image.cpp
		            common/matrix.cpp
		            common/mdebug.cpp
		            common/mesh.cpp
		            common/mtime.cpp
		            common/particle.cpp
		            common/pivotx.cpp
		            common/point_set.cpp
		            common/tinystr.cpp
		            common/tinyxml.cpp
		            common/tinyxmlerror.cpp
		            common/tinyxmlparser.cpp
		            common/utilities.cpp
		            common/vector.cpp
		            common/xml_settings.cpp
		            fluids/bank_checker.cpp
		            fluids/cmd_arg_reader.cpp
		            fluids/cutil.cpp
		            fluids/fluid.cpp
		            fluids/fluid_system.cpp
)

cuda_add_library(fluid_system_host fluids/fluid_system_host.cu fluids/fluid_system_kern.cu)

target_link_libraries(fluids_v3 GL)
target_link_libraries(fluids_v3 GLU)
target_link_libraries(fluids_v3 glut)
target_link_libraries(fluids_v3 Cg)
target_link_libraries(fluids_v3 CgGL)
target_link_libraries(fluids_v3 GLee)
target_link_libraries(fluids_v3 OpenCL)
target_link_libraries(fluids_v3 fluid_system_host)
