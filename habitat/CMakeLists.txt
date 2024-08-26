set(HABITAT_SIMULATOR_SRCS
    types.hpp
    sim.hpp sim.inl sim.cpp
)
SET(HABITAT_COMPILE_FLAGS
    -v)

add_library(habitat_cpu_impl STATIC
    ${HABITAT_SIMULATOR_SRCS}
)

target_link_libraries(habitat_cpu_impl
    PUBLIC
        madrona_mw_core
    PRIVATE
        madrona_common
        madrona_mw_physics
        madrona_rendering_system
)

add_library(habitat_importer STATIC
    import.cpp
)

target_link_libraries(habitat_importer
    PRIVATE
        madrona_libcxx
    PUBLIC
        simdjson::simdjson
)

add_library(habitat_mgr STATIC
    mgr.hpp mgr.cpp
)

target_link_libraries(habitat_mgr 
    PUBLIC
        madrona_python_utils
        habitat_importer
    PRIVATE
        habitat_cpu_impl
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
            habitat_gpu_srcs
        SOURCES_DEFN
            HABITAT_SRC_LIST
        FLAGS_DEFN
            HABITAT_COMPILE_FLAGS 
        SRCS
            ${HABITAT_SIMULATOR_SRCS}
    )

    target_link_libraries(habitat_mgr PRIVATE
        habitat_gpu_srcs
        madrona_mw_gpu
    )
endif ()

target_compile_definitions(habitat_mgr PRIVATE
    -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
)

if (TARGET madrona_viz)
    add_executable(habitat_viewer viewer.cpp)
    target_link_libraries(habitat_viewer 
        PUBLIC
        habitat_importer habitat_mgr
        PRIVATE
        madrona_mw_core madrona_viz madrona_cuda stb)

    target_compile_definitions(habitat_viewer PUBLIC
        -DMADRONA_VIEWER
    )

    if (TARGET madrona_cuda)
        target_link_libraries(habitat_viewer PRIVATE madrona_cuda)
    endif()
    target_compile_definitions(habitat_viewer PRIVATE
        -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
    )
endif()

add_executable(headless headless.cpp)
target_link_libraries(headless PUBLIC madrona_mw_core habitat_mgr madrona_viz madrona_cuda stb)
