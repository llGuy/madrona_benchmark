Problems (order of priority):
- Raytrace kernel illegal memory access when compiled with optimizations
- KTX texture loading in habitat causes memory corruption (look into asan
  because of course asan isn't working now)
- Hideseek is missing some walls

TODO:
- Add MJX

Run commands: (x is 1 for rasterizer, x is 2 for raytracer)
- Hideseek:  `MADRONA_RENDER_MODE=x gdb hideseek_viewer`
- Habitat:   `MADRONA_RENDER_MODE=x gdb habitat_viewer`
