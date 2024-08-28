Setup:
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



Problems (order of priority):
- Fix MJX

From build/ directory to run (Hideseek and Habitat):
```
# Viewer
./habitat_viewer [NUM_WORLDS] [rt|rast] [WINDOW_WIDTH] [WINDOW_HEIGHT] [BATCH_WIDTH] [BATCH_HEIGHT]
./hideseek_viewer [NUM_WORLDS] [rt|rast] [WINDOW_WIDTH] [WINDOW_HEIGHT] [BATCH_WIDTH] [BATCH_HEIGHT]

# Headless
./habitat_headless [NUM_WORLDS] [NUM_STEPS] [rt|rast] [BATCH_WIDTH] [BATCH_HEIGHT] [--dump-last-frame file_name_without_extension]
./hideseek_headless [NUM_WORLDS] [NUM_STEPS] [rt|rast] [BATCH_WIDTH] [BATCH_HEIGHT] [--dump-last-frame file_name_without_extension]
```

From scripts/ directory to run (MJX):
```
python viewer.py [-h] [--gpu-id GPU_ID] --num-worlds NUM_WORLDS --window-width WINDOW_WIDTH --window-height WINDOW_HEIGHT --batch-render-view-width BATCH_RENDER_VIEW_WIDTH --batch-render-view-height
python headless.py [-h] [--gpu-id GPU_ID] --num-worlds NUM_WORLDS --num-steps NUM_STEPS --batch-render-view-width BATCH_RENDER_VIEW_WIDTH --batch-render-view-height BATCH_RENDER_VIEW_HEIGHT
```
