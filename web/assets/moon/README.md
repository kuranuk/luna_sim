# Whole-Moon maps for the globe view

| File | Content | Source |
|---|---|---|
| `color.jpg` | LROC WAC colour map, equirectangular 4096 x 2048 (longitude -180..180 left to right, north up) | NASA SVS "CGI Moon Kit", `lroc_color_poles_16k.tif` (downscaled) |
| `elevation.jpg` | LOLA elevation, equirectangular 1024 x 512, 8-bit (stretched; higher = brighter), used as relief shading | NASA SVS "CGI Moon Kit" elevation map (8-bit) |

Credit: NASA's Scientific Visualization Studio, CGI Moon Kit, https://svs.gsfc.nasa.gov/4720
(LRO LROC and LOLA team data). NASA imagery is not copyrighted (public domain) unless noted.

The copies here were obtained from the example assets of vasturiano/globe.gl
(`example/moon-landing-sites/lunar_surface.jpg`, whose embedded metadata names
`/svs/data/lunar/lroc/albedo/hapke/lroc_color_poles_16k.tif`, and `lunar_bumpmap.jpg`) because the
SVS server was not reachable from the development environment, and converted with:

    python scripts/prepare_moon_texture.py lunar_surface.jpg --width 4096
    python scripts/prepare_moon_texture.py lunar_bumpmap.jpg --kind elevation --width 1024

To replace them with higher-resolution kit files (e.g. `lroc_color_poles_8k.tif`, `ldem_16.tif`), run
the same script on those files.
