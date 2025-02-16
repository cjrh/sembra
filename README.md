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
