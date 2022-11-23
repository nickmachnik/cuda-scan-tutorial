add_executable(parsum parsum.cpp)
set_source_files_properties(parsum.cpp PROPERTIES LANGUAGE CUDA)
target_compile_features(parsum PRIVATE cxx_std_17)
target_link_libraries(parsum PRIVATE cu_sums)