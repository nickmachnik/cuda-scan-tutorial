file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${MPS_SOURCE_DIR}/include/*.h")

add_library(cu_sums STATIC sum_host.cu sum_kernels.cu ${HEADER_LIST})
# set_source_files_properties(prep.cpp PROPERTIES LANGUAGE CUDA)
target_include_directories(cu_sums PUBLIC ../include)
target_compile_features(cu_sums PUBLIC cxx_std_17)
