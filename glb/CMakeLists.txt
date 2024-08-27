set(GLB_SIMULATOR_SRCS
    types.hpp
    sim.hpp sim.inl sim.cpp
)
SET(GLB_COMPILE_FLAGS
    -v)

add_library(glb_cpu_impl STATIC
    ${GLB_SIMULATOR_SRCS}
)

target_link_libraries(glb_cpu_impl
    PUBLIC
        madrona_mw_core
    PRIVATE
        madrona_common
        madrona_mw_physics
        madrona_rendering_system
)

add_library(glb_mgr STATIC
    mgr.hpp mgr.cpp
)

target_link_libraries(glb_mgr 
    PUBLIC
        madrona_python_utils
    PRIVATE
        glb_cpu_impl
        madrona_mw_cpu
        madrona_common
        madrona_importer
        madrona_physics_loader
        madrona_render
        madrona_ktx
        madrona_render_asset_processor
)

if (TARGET madrona_mw_gpu)
    madrona_build_compile_defns(
        OUT_TARGET
            glb_gpu_srcs
        SOURCES_DEFN
            GLB_SRC_LIST
        FLAGS_DEFN
            GLB_COMPILE_FLAGS 
        SRCS
            ${GLB_SIMULATOR_SRCS}
    )

    target_link_libraries(glb_mgr PRIVATE
        glb_gpu_srcs
        madrona_mw_gpu
    )
endif ()

target_compile_definitions(glb_mgr PRIVATE
    -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
)

if (TARGET madrona_viz)
    add_executable(glb_viewer viewer.cpp)
    target_link_libraries(glb_viewer 
        PUBLIC
        glb_mgr
        PRIVATE
        madrona_mw_core madrona_viz madrona_cuda stb run_common)

    target_compile_definitions(glb_viewer PUBLIC
        -DMADRONA_VIEWER
    )

    if (TARGET madrona_cuda)
        target_link_libraries(glb_viewer PRIVATE madrona_cuda)
    endif()
    target_compile_definitions(glb_viewer PRIVATE
        -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
    )
endif()

add_executable(glb_headless headless.cpp)
target_link_libraries(glb_headless 
    PUBLIC 
        madrona_mw_core glb_mgr madrona_viz madrona_cuda stb run_common
)
