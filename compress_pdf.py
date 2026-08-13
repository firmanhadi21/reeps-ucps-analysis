#!/usr/bin/env python3
"""
compress_pdf.py
───────────────
Shrink a manuscript PDF by downsampling and JPEG-compressing its embedded
raster images, keeping every figure visible.

The size is dominated by the Sentinel-2 basemap panels, which pdflatex embeds
losslessly (Flate). Satellite imagery compresses poorly that way; re-encoding it
as JPEG at a sensible resolution costs almost nothing visually and saves an
enormous amount of space.

    python compress_pdf.py input.pdf output.pdf [--dpi 200] [--quality 85]

Choosing a resolution: the map panels are printed about 8 cm wide, so 300 dpi
keeps them at journal print quality; 150 dpi is ample for a review copy read on
screen. Reviewer 3 criticised image sharpness in the original submission, so do
not go below 300 dpi for anything that will be typeset as the manuscript itself.
"""

import argparse
import os
import subprocess
import sys

GS = "/usr/local/bin/gs"


def compress(src: str, dst: str, dpi: int, quality: int) -> None:
    if not os.path.exists(GS):
        sys.exit(f"Ghostscript not found at {GS}")

    args = [
        GS, "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER",
        "-sDEVICE=pdfwrite",
        "-dCompatibilityLevel=1.5",
        "-dDetectDuplicateImages=true",
        "-dCompressFonts=true",
        "-dSubsetFonts=true",
        # colour images
        "-dDownsampleColorImages=true",
        "-dColorImageDownsampleType=/Bicubic",
        f"-dColorImageResolution={dpi}",
        "-dAutoFilterColorImages=false",
        "-dColorImageFilter=/DCTEncode",
        # greyscale
        "-dDownsampleGrayImages=true",
        "-dGrayImageDownsampleType=/Bicubic",
        f"-dGrayImageResolution={dpi}",
        "-dAutoFilterGrayImages=false",
        "-dGrayImageFilter=/DCTEncode",
        # line art stays crisp
        "-dDownsampleMonoImages=true",
        "-dMonoImageDownsampleType=/Subsample",
        "-dMonoImageResolution=600",
        f"-dJPEGQ={quality}",
        f"-sOutputFile={dst}", src,
    ]
    r = subprocess.run(args, capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        sys.exit(f"ghostscript failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")

    a, b = os.path.getsize(src), os.path.getsize(dst)
    print(f"  {os.path.basename(src)}  {a/1e6:.1f} MB")
    print(f"  -> {os.path.basename(dst)}  {b/1e6:.1f} MB "
          f"({a/b:.0f}x smaller, {100*(1-b/a):.1f}% saved)  "
          f"at {dpi} dpi, JPEG q{quality}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--quality", type=int, default=85)
    a = p.parse_args()
    compress(a.input, a.output, a.dpi, a.quality)


if __name__ == "__main__":
    main()
