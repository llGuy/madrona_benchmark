#include <madrona/viz/viewer.hpp>
#include <madrona/render/render_mgr.hpp>
#include <madrona/window.hpp>
#include <madrona/py/bindings.hpp>

#include "sim.hpp"
#include "mgr.hpp"
#include "types.hpp"

#include <filesystem>
#include <fstream>

#include <imgui.h>

using namespace madrona;
using namespace madrona::viz;

namespace nb = nanobind;

namespace madMJX {

struct VisualizerGPUState {
    WindowManager wm;
    WindowHandle window;
    render::GPUHandle renderGPU;

    inline VisualizerGPUState(int64_t window_width,
                              int64_t window_height,
                              int gpu_id)
        : wm(),
          window(wm.makeWindow("MadMJX", window_width, window_height)),
          renderGPU(wm.initGPU(gpu_id, { window.get() }))
    {}

    inline VisualizerGPUHandles getGPUHandles()
    {
        return VisualizerGPUHandles {
            .renderAPI = wm.gpuAPIManager().backend(),
            .renderDev = renderGPU.device(),
        };
    }
};

struct Visualizer {
    Viewer viewer;

    inline Visualizer(VisualizerGPUState &gpu_state, Manager &mgr)
        : viewer(mgr.getRenderManager(), gpu_state.window.get(), {
            .numWorlds = mgr.numWorlds(),
            .simTickRate = 30,
            .cameraMoveSpeed = 5.f,
            .cameraPosition = { 0, -3, 0 },
            .cameraRotation = { 1, 0, 0, 0 },
        })
    {}

    template <typename Fn>
    inline void loop(Manager &mgr, Fn &&sim_cb)
    {
        // Main loop for the viewer
        viewer.loop(
        [&mgr](CountT world_idx, const Viewer::UserInput &input)
        {
            (void)mgr;
            (void)world_idx;
            (void)input;
        },
        [&mgr](CountT world_idx, CountT,
               const Viewer::UserInput &input)
        {
            (void)mgr;
            (void)world_idx;
            (void)input;
        }, [&]() {
            sim_cb();
        }, [&]() {
            uint32_t raycast_output_resolution = mgr.raycastOutputResolution();

            unsigned char* print_ptr;
#ifdef MADRONA_CUDA_SUPPORT
                int64_t num_bytes = 3 * raycast_output_resolution * raycast_output_resolution;
                print_ptr = (unsigned char*)cu::allocReadback(num_bytes);
#else
                print_ptr = nullptr;
#endif

            char *raycast_tensor = (char *)(mgr.raycastRGBTensor().devicePtr());

            uint32_t bytes_per_image = 3 * raycast_output_resolution * raycast_output_resolution;
            uint32_t image_idx = viewer.getCurrentWorldID() * mgr.numCams() + 
                std::max(viewer.getCurrentViewID(), (CountT)0);
            raycast_tensor += image_idx * bytes_per_image;

            auto exec_mode = mgr.execMode();

            if(exec_mode == ExecMode::CUDA){
#ifdef MADRONA_CUDA_SUPPORT
                cudaMemcpy(print_ptr, raycast_tensor,
                        bytes_per_image,
                        cudaMemcpyDeviceToHost);
                raycast_tensor = (char *)print_ptr;
#endif
            }

            ImGui::Begin("Raycast");

            auto draw2 = ImGui::GetWindowDrawList();
            ImVec2 windowPos = ImGui::GetWindowPos();
            char *raycasters = raycast_tensor;

            int vertOff = 70;

            float pixScale = 3;
            int extentsX = (int)(pixScale * raycast_output_resolution);
            int extentsY = (int)(pixScale * raycast_output_resolution);

            for (int i = 0; i < raycast_output_resolution; i++) {
                for (int j = 0; j < raycast_output_resolution; j++) {
                    uint32_t linear_idx = 3 * (j + i * raycast_output_resolution);

                    auto realColor = IM_COL32(
                            raycasters[linear_idx + 0],
                            raycasters[linear_idx + 1],
                            raycasters[linear_idx + 2], 
                            255);

                    draw2->AddRectFilled(
                        { (i * pixScale) + windowPos.x, 
                          (j * pixScale) + windowPos.y +vertOff }, 
                        { ((i + 1) * pixScale) + windowPos.x,   
                          ((j + 1) * pixScale)+ +windowPos.y+vertOff },
                        realColor, 0, 0);
                }
            }
            ImGui::End();
        });
    }
};

NB_MODULE(_madrona_mjx_visualizer, m) {
    nb::class_<VisualizerGPUState>(m, "VisualizerGPUState")
        .def("__init__", [](VisualizerGPUState *self,
                            int64_t window_width,
                            int64_t window_height,
                            int gpu_id) {
            new (self) VisualizerGPUState(window_width, window_height, gpu_id);
        }, nb::arg("window_width"),
           nb::arg("window_height"),
           nb::arg("gpu_id") = 0)
        .def("get_gpu_handles", &VisualizerGPUState::getGPUHandles, 
             nb::keep_alive<0, 1>())
    ;

    nb::class_<Visualizer>(m, "Visualizer")
        .def("__init__", [](Visualizer *self,
                            VisualizerGPUState *viz_gpu_state,
                            Manager *mgr) {
            new (self) Visualizer(*viz_gpu_state, *mgr);
        }, nb::arg("visualizer_gpu_state"),
           nb::arg("manager"),
           nb::keep_alive<1, 2>(),
           nb::keep_alive<1, 3>())
        .def("loop",
            [](Visualizer *self, Manager *mgr, nb::callable cb,
               nb::object carry)
        {
            self->loop(*mgr, [&]() {
                carry = cb(carry);
            });

            return carry;
        })
    ;
}

}
