#include "mgr.hpp"

#include <cstdio>
#include <chrono>
#include <string>
#include <filesystem>
#include <fstream>
#include <random>

#include "args.hpp"
#include "dump.hpp"

#include <stb_image_write.h>

using namespace madrona;

int main(int argc, char *argv[])
{
    using namespace GPUHideSeek;

    run::HeadlessRunArgs args = run::parseHeadlessArgs(argc, argv);

    ExecMode exec_mode = ExecMode::CUDA;

    bool enable_batch_renderer =
        (args.renderMode == run::RenderMode::Rasterizer);

    uint64_t num_worlds = args.numWorlds;
    uint64_t num_steps = args.numSteps;
    uint32_t output_resolution = args.batchRenderWidth;

    uint32_t min_hiders = 3;
    uint32_t max_hiders = 3;
    uint32_t min_seekers = 3;
    uint32_t max_seekers = 3;

    Manager mgr({
        .execMode = exec_mode,
        .gpuID = 0,
        .numWorlds = (uint32_t)num_worlds,
        .simFlags = SimFlags::Default,
        .randSeed = 5,
        .minHiders = min_hiders,
        .maxHiders = max_hiders,
        .minSeekers = min_seekers,
        .maxSeekers = max_seekers,
        .enableBatchRenderer = enable_batch_renderer,
        .batchRenderViewWidth = output_resolution,
        .batchRenderViewHeight = output_resolution,
        .raycastOutputResolution = output_resolution,
        .headlessMode = true
    });

    mgr.init();

    auto start = std::chrono::system_clock::now();

    for (CountT i = 0; i < (CountT)num_steps; i++) {
        mgr.step();
    }

    if (args.dumpOutputFile) {
        run::dumpTiledImage({
            .outputPath = args.outputFileName,
            .gpuTensor = (void *)mgr.raycastTensor().devicePtr(),
            .numImages = (uint32_t)((min_hiders + min_seekers) * num_worlds),
            .imageResolution = output_resolution,
            .colorType = run::ColorType::RGB
        });
    }

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    float fps = (double)num_steps * (double)num_worlds / elapsed.count();
    printf("FPS %f\n", fps);
    printf("Average step time: %f ms\n", 1000.0f * elapsed.count() / (double)num_steps);
}
