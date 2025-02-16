# Sembra

Seam carving: content-aware image resizing.

## About

Translated from the Python code here:

https://github.com/li-plus/seam-carving

## CLI Usage

```sh
$ sembra --help
CLI for our seam carving demo

Usage: sembra [OPTIONS] --input <INPUT> --output <OUTPUT>

Options:
      --input <INPUT>              Input image path
      --output <OUTPUT>            Output image path
      --width <WIDTH>              Target width
      --height <HEIGHT>            Target height
      --energy-mode <ENERGY_MODE>  Energy mode: "backward" or "forward" [default: backward]
      --order <ORDER>              Order mode: "width-first" or "height-first" [default: width-first]
      --keep-mask <KEEP_MASK>      Keep mask image path (optional)
      --drop-mask <DROP_MASK>      Drop mask image path (optional)
      --step-ratio <STEP_RATIO>    Step ratio for expansions [default: 0.5]
  -h, --help                       Print help
  -V, --version                    Print version
```

## Examples

Generally speaking, you get the best results if you go smaller. It is quite impressive
how well this works. Here is an example of resizing a rectangular image down into a
square:

<div style="display: flex; gap: 20px; justify-content: center;">
  <div>
    <img src="nes.jpg" alt="Original">
    <p align="center">Original 350x500</p>
  </div>
  <div>
    <img src="nes_small_square.jpg" alt="Resized smaller">
    <p align="center">Reduced 350x350</p>
  </div>
</div>

You can also go bigger, but the results are not as good. Here is an example of the same
image, but this time we enlarge the width to make it square:

<div style="display: flex; gap: 20px; justify-content: center;">
  <div>
    <img src="nes.jpg" alt="Original" style="width: 300px;">
    <p align="center">Original 350x500</p>
  </div>
  <div>
    <img src="nes_big_square.jpg" alt="Resized larger">
    <p align="center">Enlarged 500x500</p>
  </div>
</div>

For comparison, here you can compare the seam-carving enlargement versus what
you get from a typical image resize:

<div style="display: flex; gap: 20px; justify-content: center;">
  <div>
    <img src="nes_big_square.jpg" alt="Original">
    <p align="center">Seam-carved up 500x500</p>
  </div>
  <div>
    <img src="nes-gimp-500.jpg" alt="Resized larger">
    <p align="center">Resampling 500x500</p>
  </div>
</div>

