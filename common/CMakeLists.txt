add_library(run_common
    args.cpp args.hpp
    dump.cpp dump.hpp
)

target_include_directories(run_common
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(run_common
    PRIVATE
        madrona_libcxx
        madrona_cuda
        madrona_mw_core
        stb
)
