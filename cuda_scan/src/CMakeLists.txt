file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${MPS_SOURCE_DIR}/include/*.h")

add_library(cu_sums STATIC sum_hosts.cu sum_kernels.cu ${HEADER_LIST})
# set_source_files_properties(prep.cpp PROPERTIES LANGUAGE CUDA)
target_include_directories(corr PUBLIC ../include)
target_compile_features(corr PUBLIC cxx_std_17)
