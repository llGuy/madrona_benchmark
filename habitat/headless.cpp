#include "mgr.hpp"

#include <cstdio>
#include <chrono>
#include <string>
#include <filesystem>
#include <fstream>
#include <random>

#include <stb_image_write.h>
#include <madrona/window.hpp>
#include <madrona/heap_array.hpp>
#include <madrona/render/render_mgr.hpp>

using namespace madrona;

[[maybe_unused]] static void saveWorldActions(
    const HeapArray<int32_t> &action_store,
    int32_t total_num_steps,
    int32_t world_idx)
{
    const int32_t *world_base = action_store.data() + world_idx * total_num_steps * 2 * 3;

    std::ofstream f("/tmp/actions", std::ios::binary);
    f.write((char *)world_base,
            sizeof(uint32_t) * total_num_steps * 2 * 3);
}

void transposeImage(char *output, 
                    const char *input,
                    uint32_t res)
{
    for (uint32_t y = 0; y < res; ++y) {
        for (uint32_t x = 0; x < res; ++x) {
            output[4*(y + x * res) + 0] = input[4*(x + y * res) + 0];
            output[4*(y + x * res) + 1] = input[4*(x + y * res) + 1];
            output[4*(y + x * res) + 2] = input[4*(x + y * res) + 2];
            output[4*(y + x * res) + 3] = input[4*(x + y * res) + 3];
        }
    }
}

int main(int argc, char *argv[])
{
    using namespace madEscape;

    if (argc < 4) {
        fprintf(stderr, "%s TYPE NUM_WORLDS NUM_STEPS [--rand-actions]\n", argv[0]);
        return -1;
    }
    std::string type(argv[1]);

    ExecMode exec_mode;
    if (type == "CPU") {
        exec_mode = ExecMode::CPU;
    } else if (type == "CUDA") {
        exec_mode = ExecMode::CUDA;
    } else {
        fprintf(stderr, "Invalid ExecMode\n");
        return -1;
    }

    uint64_t num_worlds = std::stoul(argv[2]);
    uint64_t num_steps = std::stoul(argv[3]);

    HeapArray<int32_t> action_store(
        num_worlds * 2 * num_steps * 3);

    bool rand_actions = false;
    if (argc >= 5) {
        if (std::string(argv[4]) == "--rand-actions") {
            rand_actions = true;
        }
    }

    auto *render_mode = getenv("MADRONA_RENDER_MODE");

    bool enable_batch_renderer =
#ifdef MADRONA_MACOS
        false;
#else
        render_mode[0] == '1';
#endif

    auto *resolution_str = getenv("MADRONA_RENDER_RESOLUTION");

    uint32_t raycast_output_resolution = std::stoi(resolution_str);
    printf("raycast_output_resolution=%d\n", raycast_output_resolution);

    Manager mgr({
        .execMode = exec_mode,
        .gpuID = 0,
        .numWorlds = (uint32_t)num_worlds,
        .autoReset = false,
        .enableBatchRenderer = enable_batch_renderer,
        .batchRenderViewWidth = raycast_output_resolution,
        .batchRenderViewHeight = raycast_output_resolution,
        .raycastOutputResolution = raycast_output_resolution,
        .headlessMode = true
    });

    std::random_device rd;
    std::mt19937 rand_gen(rd());
    std::uniform_int_distribution<int32_t> act_rand(0, 2);

    auto start = std::chrono::system_clock::now();

    for (CountT i = 0; i < (CountT)num_steps; i++) {
        if (rand_actions) {
            for (CountT j = 0; j < (CountT)num_worlds; j++) {
                for (CountT k = 0; k < 2; k++) {
                    int32_t x = act_rand(rand_gen);
                    int32_t y = act_rand(rand_gen);
                    int32_t r = act_rand(rand_gen);

                    mgr.setAction(j, k, x, y, r, 0,act_rand(rand_gen),1,1,2,1);
                    
                    int64_t base_idx = j * num_steps * 2 * 3 + i * 2 * 3 + k * 3;
                    action_store[base_idx] = x;
                    action_store[base_idx + 1] = y;
                    action_store[base_idx + 2] = r;
                }
            }
        }
        mgr.step();

        uint32_t num_images_total = num_worlds;

        unsigned char* print_ptr;
            int64_t num_bytes = 4 * raycast_output_resolution * raycast_output_resolution * num_images_total;
            print_ptr = (unsigned char*)cu::allocReadback(num_bytes);

        char *raycast_tensor = (char *)(mgr.raycastTensor().devicePtr());

        uint32_t bytes_per_image = 4 * raycast_output_resolution * raycast_output_resolution;
        uint32_t row_stride_bytes = 4 * raycast_output_resolution;

        uint32_t image_idx = 0;

        uint32_t base_image_idx = num_images_total * (image_idx / num_images_total);

        raycast_tensor += image_idx * bytes_per_image;

        if(exec_mode == ExecMode::CUDA){
            cudaMemcpy(print_ptr, raycast_tensor,
                    num_bytes,
                    cudaMemcpyDeviceToHost);
            raycast_tensor = (char *)print_ptr;
        }

        char *tmp_image_memory = (char *)malloc(bytes_per_image);

        char *image_memory = (char *)malloc(bytes_per_image * num_images_total);

        uint32_t num_images_y = 10;
        uint32_t num_images_x = num_images_total / num_images_y;

        uint32_t output_num_pixels_x = num_images_x * raycast_output_resolution;

        for (uint32_t image_y = 0; image_y < num_images_y; ++image_y) {
            for (uint32_t image_x = 0; image_x < num_images_x; ++image_x) {
                uint32_t image_idx = image_x + image_y * num_images_x;

                const char *input_image = raycast_tensor + image_idx * bytes_per_image;

                transposeImage(tmp_image_memory, input_image, raycast_output_resolution);

                for (uint32_t row_idx = 0; row_idx < raycast_output_resolution; ++row_idx) {
                    const char *input_row = tmp_image_memory + row_idx * row_stride_bytes;

                    uint32_t output_pixel_x = image_x * raycast_output_resolution;
                    uint32_t output_pixel_y = image_y * raycast_output_resolution + row_idx;
                    char *output_row = image_memory + 4 * (output_pixel_x + output_pixel_y * output_num_pixels_x);

                    memcpy(output_row, input_row, 4 * raycast_output_resolution);
                }
            }
        }

        std::string file_name = std::string("out") + std::to_string(i) + ".png";
        stbi_write_png(file_name.c_str(), raycast_output_resolution * num_images_x, num_images_y * raycast_output_resolution,
                      4, image_memory, 4 * num_images_x * raycast_output_resolution);

        free(image_memory);
    }

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    float fps = (double)num_steps * (double)num_worlds / elapsed.count();
    printf("FPS %f\n", fps);
    printf("Average total step time: %f ms\n",
           1000.0f * elapsed.count() / (double)num_steps);
}
