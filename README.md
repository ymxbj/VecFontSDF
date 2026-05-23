<div align="center">

# VecFontSDF: Learning to Reconstruct and Synthesize High-quality Vector Fonts via Signed Distance Functions

**CVPR 2023**

[Zeqing Xia](https://xiazeqing.github.io/)<sup>*</sup> &nbsp;·&nbsp;
[Bojun Xiong](https://ymxbj.github.io/)<sup>*</sup> &nbsp;·&nbsp;
[Zhouhui Lian](mailto:lianzhouhui@pku.edu.cn)<sup>†</sup>

Wangxuan Institute of Computer Technology, Peking University

<sub><sup>*</sup> Equal contribution &nbsp;&nbsp; <sup>†</sup> Corresponding author</sub>

[Project page](https://xiazeqing.github.io/VecFontSDF/) ·
[arXiv](https://arxiv.org/abs/2303.12675)

<br>

<img src="./teaser.png" alt="VecFontSDF teaser" width="80%">

</div>

---

The official PyTorch implementation of the VecFontSDF paper. This paper proposes an end-to-end trainable method, VecFontSDF, to reconstruct and synthesize high-quality vector fonts using signed distance functions (SDFs). Specifically, based on the proposed SDF-based implicit shape representation, VecFontSDF learns to model each glyph as shape primitives enclosed by several parabolic curves, which can be precisely converted to quadratic Bézier curves that are widely used in vector font products. In this manner, most image generation methods can be easily extended to synthesize vector fonts. Qualitative and quantitative experiments conducted on a publicly available dataset demonstrate that our method obtains high-quality results on several tasks, including vector font reconstruction, interpolation, and few-shot vector font synthesis, markedly outperforming the state of the art.

## Table of contents

- [Installation](#installation)
- [Repository structure](#repository-structure)
- [Data preparation](#data-preparation)
- [Inference](#inference)
- [Training](#training)
- [Release plan](#release-plan)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## Installation

The code has been tested on Python 3.12, CUDA 12.8, PyTorch 2.7 with NVIDIA V100 GPUs.

```bash
git clone https://github.com/ymxbj/VecFontSDF.git
cd VecFontSDF

conda create -n vecfontsdf python=3.12 -y
conda activate vecfontsdf

# Install PyTorch matching your CUDA version, e.g.:
pip3 install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

pip3 install -r requirements.txt
```

## Repository structure

```
VecFontSDF/
├── options.py     — command-line arguments
├── dataloader.py  — single-glyph dataset
├── model.py       — ResNet-18 encoder + 2-layer SDF decoder
├── losses.py      — pseudo distance field + differentiable rasterization + losses
├── train.py       — iteration-based training loop
├── inference.py   — load a checkpoint and reconstruct an image / directory
├── data_prep/     — turn raw SVG fonts into the training-ready layout
│   ├── geometry.py            — Pos, StraightLine, QuadraticBezier primitives
│   ├── glyph.py               — Glyph / Contour: SVG parsing + signed distance
│   ├── cubic_to_quadratic.py  — rewrite cubic-Bezier SVGs as quadratic-only
│   ├── svg_to_grid_sdf.py     — per-pixel grid SDF generator
│   ├── svg_to_contour_sdf.py  — contour-aligned SDF sample generator
│   └── svg_to_png.py          — SVG → raster PNG (uses cairosvg)
└── README.md
```

## Data preparation

The training set is organized one directory per font, named with a 4-digit
zero-padded integer id (e.g. `0123`):

```
data/
├── font_list.txt                a python-eval-able list of ints, e.g. [0, 1, 2, ...]
├── img/
│   ├── 0000/                    font #0
│   │   ├── 0.png                glyph index 0 (= ASCII '0')
│   │   ├── 1.png                glyph index 1 (= ASCII '1')
│   │   └── ...                  62 files total: 0-9 + A-Z + a-z
│   └── 0001/
└── sdf/
    └── 0000/
        └── sdf/
            ├── 48_grid.npy      [H, W] grid SDF for '0' (ASCII 48)
            ├── 48_contour.npy   [M_c, 3] = (x, y, sdf) contour samples for '0'
            └── ...
```

Notes on the on-disk naming:
- Image files are named by **glyph index 0..61** (`0.png`, `1.png`, ...).
- SDF files are named by **ASCII codepoint** (`48_grid.npy` is `'0'`,
  `65_grid.npy` is `'A'`, ...).
- `grid.npy` is stored as `(col, row)` and is transposed by the dataloader
  to standard `(row, col)`. Positive values are outside the glyph, negative
  inside.
- `contour.npy` is a `(M_c, 3)` array; each row is
  `(x, y, signed_distance)` in pixel units (`x` is the column coordinate
  and `y` is the row coordinate, both in `[0, H]`).

### Generating SDFs from raw SVG fonts

If you start from a directory of SVG glyphs (one directory per font,
filenames `<codepoint>.svg`), the helpers under `data_prep/` produce
everything the dataloader expects. Each script auto-skips files that
already exist, so they are safe to rerun.

The SDF / dataloader pipeline only understands SVG path commands `M`, `L`,
and `Q` (move, line, and quadratic Bezier). If your input SVGs contain
cubic Bezier `C` segments, run the downgrade step first.

```bash
# 0. (optional) downgrade cubic-Bezier SVGs to quadratic-only
python3 -m VecFontSDF.data_prep.cubic_to_quadratic \
    --svg_root /path/to/cubic_svg --out_root /path/to/quadratic_svg --workers 8

# 1. rasterized glyph images (uses cairosvg)
python3 -m VecFontSDF.data_prep.svg_to_png \
    --svg_root /path/to/quadratic_svg --out_root ./data/img --image_size 128 --workers 8

# 2. per-pixel grid SDF
python3 -m VecFontSDF.data_prep.svg_to_grid_sdf \
    --svg_root /path/to/quadratic_svg --out_root ./data/sdf --image_size 128 --workers 8

# 3. contour-aligned SDF samples (default 4000 points per glyph)
python3 -m VecFontSDF.data_prep.svg_to_contour_sdf \
    --svg_root /path/to/quadratic_svg --out_root ./data/sdf --num_points 4000 --workers 8

# 4. write the font id list
ls /path/to/quadratic_svg | python3 -c "import sys; print([int(x) for x in sys.stdin.read().split()])" > ./data/font_list.txt
```

Step 0 is a best-effort downgrade — a cubic that genuinely cannot be
represented as a single quadratic (its left and right tangents do not
meet at the same point) will be reported as a failure; the script writes
no output for the offending glyph and lists it at the end.

## Inference

```bash
python3 -m VecFontSDF.inference \
    --ckpt experiments/vecfontsdf_recon/checkpoints/latest.pth \
    --input some_glyph.png \
    --render_size 256 \
    --save_params
```

Produces, in `experiments/vecfontsdf_recon/inference/`:

- `<stem>_recon.png` — input next to the reconstruction at `--render_size`.
- `<stem>_params.npy` (with `--save_params`) — `[N_p * N_a, 6]` array of
  parabolic curve parameters; the 6 columns are `(k, p, q, d, e, f)`.

`--input` may also be a directory, in which case all `.png` / `.jpg` files
in it are processed in a single pass.

## Training

```bash
python3 -m VecFontSDF.train \
    --img_path  /path/to/img \
    --sdf_path  /path/to/sdf \
    --font_list /path/to/font_list.txt \
    --train_split 1000 \
    --batch_size 64 --num_workers 8 \
    --n_iters 100000 \
    --experiment_name vecfontsdf_recon
```

Outputs land in `experiments/vecfontsdf_recon/`:

- `samples/` — periodic side-by-side comparisons of GT vs. reconstruction.
- `checkpoints/` — `vecfontsdf_*.pth` snapshots and a rolling `latest.pth`.
- `logs/` — TensorBoard event files (if `tensorboard` is installed).
- `opts.txt` — full dump of the run's hyperparameters; used by
  `inference.py` to restore the model architecture.

Run `python3 -m VecFontSDF.train --help` for the full argument list (loss
weights, primitive count, optimizer hyperparameters, validation cadence,
etc.).

To resume from a checkpoint, rerun the same training command with
`--resume <path-to-ckpt>` appended:

```bash
python3 -m VecFontSDF.train \
    --img_path  /path/to/img \
    --sdf_path  /path/to/sdf \
    --font_list /path/to/font_list.txt \
    --train_split 1000 \
    --batch_size 64 --num_workers 8 \
    --n_iters 100000 \
    --experiment_name vecfontsdf_recon \
    --resume experiments/vecfontsdf_recon/checkpoints/latest.pth
```

## Release plan

- [x] Training code
- [x] Inference code
- [x] Data preparation code (SVG → grid / contour SDF + raster PNG)
- [ ] Pre-trained checkpoints
- [ ] Training data

## Citation

If you find this work useful in your research, please cite the original paper:

```bibtex
@InProceedings{Xia_2023_CVPR,
  author    = {Xia, Zeqing and Xiong, Bojun and Lian, Zhouhui},
  title     = {VecFontSDF: Learning To Reconstruct and Synthesize High-Quality Vector Fonts via Signed Distance Functions},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2023},
  pages     = {1848-1857}
}
```

## License

This project is released under the [MIT License](./LICENSE).

## Contact

If you have any questions, please contact xiongbojun@pku.edu.cn.

## Acknowledgments

This codebase reimplements the reconstruction pipeline introduced by
Xia, Xiong, and Lian in their CVPR 2023 paper. We thank the authors for
their work. This work was supported by National Language Committee of
China (Grant No.: ZDI135-130), Center For Chinese Font Design and
Research, and Key Laboratory of Science, Technology and Standard in
Press Industry (Key Laboratory of Intelligent Press Media Technology).
