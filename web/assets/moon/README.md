# Whole-Moon texture for the globe view

Put an equirectangular lunar colour map here as `color.jpg` (longitude -180..180 left to right,
north up). Without it the globe uses a procedural placeholder (not real lunar features).

Recommended source: NASA SVS "CGI Moon Kit" (https://svs.gsfc.nasa.gov/4720, public domain,
credit: NASA's Scientific Visualization Studio), e.g. `lroc_color_poles_2k.tif` / `_4k.tif`:

    python scripts/prepare_moon_texture.py lroc_color_poles_4k.tif --width 4096
