#pragma once

#include <string>

namespace run {

enum class ColorType {
    RGB, Depth
};
    
struct DumpInfo {
    std::string outputPath;

    // Pointer to CUDA memory containing the images.
    void *gpuTensor;

    // We will calculate what the best resolution is for this output.
    uint32_t numImages;

    // Resolution of each individual imagea
    uint32_t imageResolution;

    ColorType colorType;
};

void dumpTiledImage(const DumpInfo &info);
void dumpTiledImage(const DumpInfo &info,
                    uint32_t teaser_width,
                    uint32_t teaser_height);

// `dir_out` is where all the images will get dumped into as .png
void dumpImages(const DumpInfo &info,
                const char *dir_out,
                const char *name_base);

}
