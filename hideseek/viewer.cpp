#include <madrona/viz/viewer.hpp>
#include <madrona/render/render_mgr.hpp>

#include "mgr.hpp"
#include "sim.hpp"

#include <filesystem>
#include <fstream>
#include <imgui.h>

#include <stb_image_write.h>

#include "args.hpp"

using namespace madrona;
using namespace madrona::viz;

void transposeImage(char *output, 
                    const char *input,
                    uint32_t res,
                    uint32_t comp)
{
    for (uint32_t y = 0; y < res; ++y) {
        for (uint32_t x = 0; x < res; ++x) {
            output[3*(y + x * res) + 0] = input[3*(x + y * res) + 0];
            output[3*(y + x * res) + 1] = input[3*(x + y * res) + 1];
            output[3*(y + x * res) + 2] = input[3*(x + y * res) + 2];
        }
    }
}

static HeapArray<int32_t> readReplayLog(const char *path)
{
    std::ifstream replay_log(path, std::ios::binary);
    replay_log.seekg(0, std::ios::end);
    int64_t size = replay_log.tellg();
    replay_log.seekg(0, std::ios::beg);

    HeapArray<int32_t> log(size / sizeof(int32_t));

    replay_log.read((char *)log.data(), (size / sizeof(int32_t)) * sizeof(int32_t));

    return log;
}

int main(int argc, char *argv[])
{
    using namespace GPUHideSeek;

    run::ViewerRunArgs args = run::parseViewerArgs(argc, argv);

    uint32_t num_worlds = args.numWorlds;
    ExecMode exec_mode = ExecMode::CUDA;

    uint32_t num_hiders = 3;
    uint32_t num_seekers = 3;
    uint32_t num_views = num_hiders + num_seekers;

    SimFlags sim_flags = SimFlags::Default;
    sim_flags |= SimFlags::IgnoreEpisodeLength;

    bool enable_batch_renderer = (args.renderMode == run::RenderMode::Rasterizer);

    uint32_t output_resolution = args.batchRenderWidth;

    WindowManager wm {};
    WindowHandle window = wm.makeWindow("Hide & Seek", 2730/2, 1536/2);
    render::GPUHandle render_gpu = wm.initGPU(0, { window.get() });

    Manager mgr({
        .execMode = exec_mode,
        .gpuID = 0,
        .numWorlds = num_worlds,
        .simFlags = sim_flags,
        .randSeed = 10,
        .minHiders = num_hiders,
        .maxHiders = num_hiders,
        .minSeekers = num_seekers,
        .maxSeekers = num_seekers,
        .enableBatchRenderer = enable_batch_renderer,
        .batchRenderViewWidth = output_resolution,
        .batchRenderViewHeight = output_resolution,
        .extRenderAPI = wm.gpuAPIManager().backend(),
        .extRenderDev = render_gpu.device(),
        .raycastOutputResolution = output_resolution
    });
    mgr.init();

    math::Quat initial_camera_rotation =
        (math::Quat::angleAxis(-math::pi / 2.f, math::up) *
        math::Quat::angleAxis(-math::pi / 2.f, math::right)).normalize();

    viz::Viewer viewer(mgr.getRenderManager(), window.get(), {
        .numWorlds = num_worlds,
        .simTickRate = 25_u32,
        .cameraMoveSpeed = 10.f,
        .cameraPosition = { 0.f, 0.f, 40 },
        .cameraRotation = initial_camera_rotation,
    });

    viewer.loop(
    [&](CountT world_idx,
        const Viewer::UserInput &input)
    {
        using Key = Viewer::KeyboardKey;

        if (input.keyHit(Key::R)) {
            mgr.triggerReset(world_idx, 1);
        }

        if (input.keyHit(Key::K1)) {
            mgr.triggerReset(world_idx, 1);
        }

        if (input.keyHit(Key::K2)) {
            mgr.triggerReset(world_idx, 2);
        }

        if (input.keyHit(Key::K3)) {
            mgr.triggerReset(world_idx, 3);
        }

        if (input.keyHit(Key::K4)) {
            mgr.triggerReset(world_idx, 4);
        }

        if (input.keyHit(Key::K5)) {
            mgr.triggerReset(world_idx, 5);
        }

        if (input.keyHit(Key::K6)) {
            mgr.triggerReset(world_idx, 6);
        }

        if (input.keyHit(Key::K7)) {
            mgr.triggerReset(world_idx, 7);
        }

        if (input.keyHit(Key::K8)) {
            mgr.triggerReset(world_idx, 8);
        }

        if (input.keyHit(Key::K9)) {
            mgr.triggerReset(world_idx, 9);
        }

    },
    [&](CountT world_idx, CountT agent_idx,
        const Viewer::UserInput &input)
    {
        using Key = Viewer::KeyboardKey;

        int32_t x = 5;
        int32_t y = 5;
        int32_t r = 5;
        bool g = false;
        bool l = false;

        if (input.keyPressed(Key::W)) {
            y += 5;
        }
        if (input.keyPressed(Key::S)) {
            y -= 5;
        }

        if (input.keyPressed(Key::D)) {
            x += 5;
        }
        if (input.keyPressed(Key::A)) {
            x -= 5;
        }

        if (input.keyPressed(Key::Q)) {
            r += 5;
        }
        if (input.keyPressed(Key::E)) {
            r -= 5;
        }

        if (input.keyHit(Key::G)) {
            g = true;
        }
        if (input.keyHit(Key::L)) {
            l = true;
        }

        mgr.setAction(world_idx * num_views + agent_idx, x, y, r, g, l);
    }, [&]() {
        mgr.step();
    }, [&]() {
        unsigned char* print_ptr;
        #ifdef MADRONA_CUDA_SUPPORT
            int64_t num_bytes = 4 * output_resolution * output_resolution;
            print_ptr = (unsigned char*)cu::allocReadback(num_bytes);
        #else
            print_ptr = nullptr;
        #endif

        char *raycast_tensor = (char *)(mgr.raycastTensor().devicePtr());

        uint32_t bytes_per_image = 4 * output_resolution * output_resolution;
        uint32_t image_idx = viewer.getCurrentWorldID() * GPUHideSeek::consts::maxAgents + 
            std::max(viewer.getCurrentViewID(), (CountT)0);
        raycast_tensor += image_idx * bytes_per_image;

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
        int extentsX = (int)(pixScale * output_resolution);
        int extentsY = (int)(pixScale * output_resolution);

        for (int i = 0; i < output_resolution; i++) {
            for (int j = 0; j < output_resolution; j++) {
                uint32_t linear_idx = 4 * (j + i * output_resolution);

                auto realColor = IM_COL32(
                        (uint8_t)raycasters[linear_idx + 0],
                        (uint8_t)raycasters[linear_idx + 1],
                        (uint8_t)raycasters[linear_idx + 2], 
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
