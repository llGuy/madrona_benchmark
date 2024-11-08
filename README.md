# Madrona Renderer Benchmark

This repository contains the code for the renderer benchmarks from our [SIGGRAPH 
Asia 2024 paper](https://madrona-engine.github.io/renderer.html). This repository
contains code for the Hide & Seek, Habitat, MJX and ProcTHOR rendering-related
environments.

## Setup

```
git submodule update --init --recursive
cd data/hssd-hab
git lfs pull
cd -
cmake -S . -B build
cd build
make -j
cd -
python -m venv mjxenv
source mjxenv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Running with the viewer

From build/ directory to run (Hideseek and Habitat):
```
# Viewer
./habitat_viewer [NUM_WORLDS] [rt|rast] [WINDOW_WIDTH] [WINDOW_HEIGHT] [BATCH_WIDTH] [BATCH_HEIGHT]
./hideseek_viewer [NUM_WORLDS] [rt|rast] [WINDOW_WIDTH] [WINDOW_HEIGHT] [BATCH_WIDTH] [BATCH_HEIGHT]
```

From scripts/ directory to run MJX:
```
python viewer.py [-h] [--gpu-id GPU_ID] --num-worlds NUM_WORLDS --window-width WINDOW_WIDTH --window-height WINDOW_HEIGHT --batch-render-view-width BATCH_RENDER_VIEW_WIDTH --batch-render-view-height
```

## Running headless with benchmarked numbers printed after running:

From build/ directory to run (Hideseek and Habitat):
```
# Headless
./habitat_headless [NUM_WORLDS] [NUM_STEPS] [rt|rast] [BATCH_WIDTH] [BATCH_HEIGHT] [--dump-last-frame file_name_without_extension]
./hideseek_headless [NUM_WORLDS] [NUM_STEPS] [rt|rast] [BATCH_WIDTH] [BATCH_HEIGHT] [--dump-last-frame file_name_without_extension]
```

From scripts/ directory to run MJX:
```
python headless.py [-h] [--gpu-id GPU_ID] --num-worlds NUM_WORLDS --num-steps NUM_STEPS --batch-render-view-width BATCH_RENDER_VIEW_WIDTH --batch-render-view-height BATCH_RENDER_VIEW_HEIGHT
```

## Results

Feel free to look at the paper for a more detailed analysis of the renderer's performance.
Here we list out raw performance numbers on the GPUs we have tested for the Habitat environment
(the most geometrically complex environment in the repository ~7M tri's per scene).

### RTX 4090

| *       | Ray Tracer | Rasterizer |
|---------|------------|------------|
| 64x64   | 60K FPS    | 15K FPS    |
| 128x128 | 30K FPS    | 15K FPS    |
| 256x256 | 13K FPS    | 15K FPS    |

### H100

| *       | Ray Tracer | Rasterizer |
|---------|------------|------------|
| 64x64   | 60K FPS    | 400 FPS    |
| 128x128 | 30K FPS    | 400 FPS    |
| 256x256 | 13K FPS    | 400 FPS    |
