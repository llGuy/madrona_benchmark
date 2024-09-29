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



For NCU
```
sudo MADRONA_MWGPU_KERNEL_CACHE=hssd_cache ncu --import-source on --replay-mode kernel --mode launch-and-attach -k bvhRaycastEntry -c 1 -s 20 --set full --section WarpStateStats --section SourceCounters --section InstructionStats --metrics smsp__pcsamp_warps_issue_stalled_barrier_not_issued,smsp__pcsamp_warps_issue_stalled_branch_resolving_not_issued,smsp__pcsamp_warps_issue_stalled_dispatch_stall_not_issued,smsp__pcsamp_warps_issue_stalled_drain_not_issued,smsp__pcsamp_warps_issue_stalled_imc_miss_not_issued,smsp__pcsamp_warps_issue_stalled_lg_throttle_not_issued,smsp__pcsamp_warps_issue_stalled_long_scoreboard_not_issued,smsp__pcsamp_warps_issue_stalled_math_pipe_throttle_not_issued,smsp__pcsamp_warps_issue_stalled_membar_not_issued,smsp__pcsamp_warps_issue_stalled_mio_throttle_not_issued,smsp__pcsamp_warps_issue_stalled_misc_not_issued,smsp__pcsamp_warps_issue_stalled_no_instructions_not_issued,smsp__pcsamp_warps_issue_stalled_not_selected_not_issued,smsp__pcsamp_warps_issue_stalled_selected_not_issued,smsp__pcsamp_warps_issue_stalled_short_scoreboard_not_issued,smsp__pcsamp_warps_issue_stalled_sleeping_not_issued,smsp__pcsamp_warps_issue_stalled_tex_throttle_not_issued,smsp__pcsamp_warps_issue_stalled_wait_not_issued,branch_inst_executed,group:smsp__pcsamp_warp_stall_reasons,smsp__pcsamp_sample_count,smsp__pcsamp_warps_issue_stalled_long_scoreboard,smsp__branch_targets_threads_divergent,group:smsp__pcsamp_warp_stall_reasons_not_issued,smsp__pcsamp_warps_issue_stalled_dispatch_stall_not_issued,memory_access_type,smsp__cycles_elapsed.avg.per_second,sm__cycles_elapsed.avg.per_second -o new_trace ./habitat_headless 128 64 rt 64 64
```
