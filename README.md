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

```bash
$ sembra --input nes.jpg --output nes_big_square.jpg \
    --width 350 --height 350 --energy-mode forward
```

<table>
  <tr>
    <td>
      <img src="nes.jpg" alt="Original">
      <p align="center">Original 350x500, <em>Nymphs and Satyr</em>, 1873, William-Adolphe Bouguereau</p>
    </td>
    <td>
      <img src="nes_small_square.jpg" alt="Resized smaller">
      <p align="center">Reduced 350x350</p>
    </td>
  </tr>
</table>

We turned a rectangular image into a square with very little observable distortion!
Quite remarkable.

You can also go bigger, but the results are not as good. Here is an example of the same
image, but this time we enlarge the width to make it square:

```bash
$ sembra --input nes.jpg --output nes_big_square.jpg \
    --width 500 --height 500 --energy-mode forward
```

<table>
  <tr>
    <td>
      <img src="nes.jpg" alt="Original">
      <p align="center">Original 350x500</p>
    </td>
    <td>
      <img src="nes_big_square.jpg" alt="Resized larger">
      <p align="center">Enlarged 500x500</p>
    </td>
  </tr>
</table>

While there is clearly some distortion, there is also excellent preservation
of some of the more detailed parts of the image. This is what seam carving
gives you. You can see this by comparing the seam-carving enlargement versus what
you get from a typical image resize (resampling) in an image editor:

<table>
  <tr>
    <td>
      <img src="nes_big_square.jpg" alt="Original">
      <p align="center">Seam-carved up 500x500</p>
    </td>
    <td>
      <img src="nes-gimp-500.jpg" alt="Resized larger">
      <p align="center">Resampling 500x500</p>
    </td>
  </tr>
</table>

Side-by-side, you can clearly see that the seam-carving enlargment has preserved
detail in key areas, like faces, fingers, eyes, and so on. Of course, this comes
at the cost greater distortion in other less detailed areas.
