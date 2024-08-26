#pragma once

#include <string>
#include <stdint.h>

namespace run {

enum class RenderMode {
    Raycaster,
    Rasterizer
};

struct ViewerRunArgs {
    uint32_t numWorlds;

    RenderMode renderMode;

    uint32_t windowWidth;
    uint32_t windowHeight;

    uint32_t batchRenderWidth;
    uint32_t batchRenderHeight;
};

struct HeadlessRunArgs {
    uint32_t numWorlds;
    uint32_t numSteps;

    RenderMode renderMode;

    uint32_t batchRenderWidth;
    uint32_t batchRenderHeight;

    // Dumps the output for the final frame.
    bool dumpOutputFile;
    std::string outputFileName;
};

ViewerRunArgs parseViewerArgs(int argc, char **argv);
HeadlessRunArgs parseHeadlessArgs(int argc, char **argv);

}
