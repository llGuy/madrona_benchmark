set(HIDESEEK_SIMULATOR_SRCS
    sim.hpp sim.cpp
    sim_flags.hpp sim_flags.inl
    geo_gen.hpp geo_gen.inl geo_gen.cpp
    level_gen.hpp level_gen.cpp
)

add_library(gpu_hideseek_cpu_impl STATIC
    ${HIDESEEK_SIMULATOR_SRCS}
)

target_link_libraries(gpu_hideseek_cpu_impl
    PUBLIC
        madrona_mw_core
    PRIVATE
        madrona_common
        madrona_mw_physics
        madrona_rendering_system
)

add_library(gpu_hideseek_mgr STATIC
    mgr.hpp mgr.cpp
)

target_link_libraries(gpu_hideseek_mgr
    PUBLIC
        madrona_python_utils
    PRIVATE
        gpu_hideseek_cpu_impl
        madrona_mw_cpu
        madrona_common
        madrona_importer
        madrona_physics_loader
        madrona_render
)

if (TARGET madrona_mw_gpu)
    madrona_build_compile_defns(
        OUT_TARGET
            gpu_hideseek_gpu_srcs
        SOURCES_DEFN
            GPU_HIDESEEK_SRC_LIST
        FLAGS_DEFN
            GPU_HIDESEEK_COMPILE_FLAGS 
        SRCS
            ${HIDESEEK_SIMULATOR_SRCS}
    )

    target_link_libraries(gpu_hideseek_mgr PRIVATE
        gpu_hideseek_gpu_srcs
        madrona_mw_gpu
    )
endif ()

target_compile_definitions(gpu_hideseek_mgr PRIVATE
    -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
)

if (TARGET madrona_viz)
    add_executable(hideseek_viewer viewer.cpp)
    target_link_libraries(hideseek_viewer PRIVATE
        madrona_mw_core gpu_hideseek_mgr madrona_viz run_common)

    target_compile_definitions(hideseek_viewer PRIVATE
        -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
    )

    target_compile_definitions(hideseek_viewer PRIVATE
        -DMADRONA_VIEWER
    )

    if (TARGET madrona_cuda)
        target_link_libraries(hideseek_viewer PRIVATE madrona_cuda stb)
    endif()
endif()

add_executable(hideseek_headless headless.cpp)
target_link_libraries(hideseek_headless 
    madrona_mw_core gpu_hideseek_mgr stb madrona_cuda run_common)
