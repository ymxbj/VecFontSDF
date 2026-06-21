import os
from typing import List

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms


DEFAULT_CODEPOINTS: List[int] = (
    list(range(65, 91))     # A-Z  -> class labels 0..25
    + list(range(97, 123))  # a-z  -> class labels 26..51
)


def load_font_list(path: str) -> List[int]:
    """Read a font id list stored as a python-literal list of ints."""
    with open(path) as f:
        return list(eval(f.read()))


class GlyphReconDataset(data.Dataset):
    """Single-glyph reconstruction dataset.

    Each sample is one (font_id, glyph_index) pair. glyph_index is the internal
    0..61 index; the matching ASCII codepoint is looked up in DEFAULT_CODEPOINTS
    and used to name the SDF files. Note that on disk, image files are named by
    glyph index 0..61 while SDF files are named by ASCII codepoint.

    Each sample is a dict:
        image        [1, H, W]   raster glyph read as grayscale, values in [0, 1]
        class        [C]         one-hot character label (C = char_categories);
                                 the hot index equals glyph_idx
        grid_sdf     [H, W]      per-pixel signed distance, in (row, col) layout
                                 (positive outside the glyph, negative inside)
        contour_sdf  [M_c, 3]    each row = (x, y, signed_distance) in pixel
                                 units; x is the column coordinate and y is
                                 the row coordinate, both in [0, H], and
                                 signed_distance > 0 outside the glyph
        glyph_idx    int         0..61
        font_id      int
    """

    def __init__(self, img_root: str, sdf_root: str, font_ids: List[int],
                 image_size: int, codepoints: List[int] = DEFAULT_CODEPOINTS):
        super().__init__()
        self.img_root = img_root
        self.sdf_root = sdf_root
        self.codepoints = codepoints
        self.transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        self.index = [(fid, gi) for fid in font_ids for gi in range(len(codepoints))]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        font_id, glyph_idx = self.index[idx]
        codepoint = self.codepoints[glyph_idx]
        font_str = '%04d' % font_id

        img = Image.open(
            os.path.join(self.img_root, font_str, f'{glyph_idx}.png')
        ).convert('L')
        img = self.transforms(img)  # [1, H, W] in [0, 1]

        sdf_dir = os.path.join(self.sdf_root, font_str, 'sdf')
        grid_sdf = np.load(os.path.join(sdf_dir, f'{codepoint}_grid.npy'))
        grid_sdf = torch.from_numpy(grid_sdf).float()
        # On-disk layout is (col, row); transpose to standard (row, col).
        grid_sdf = grid_sdf.transpose(0, 1).contiguous()

        contour_sdf = np.load(os.path.join(sdf_dir, f'{codepoint}_contour.npy'))
        contour_sdf = torch.from_numpy(contour_sdf).float()  # [M_c, 3]

        # glyph_idx (0..51) is also the conditioning class label.
        clss = torch.zeros(len(self.codepoints), dtype=torch.float32)
        clss[glyph_idx] = 1.0

        return {
            'image': img,
            'grid_sdf': grid_sdf,
            'contour_sdf': contour_sdf,
            'class': clss,
            'glyph_idx': torch.tensor(glyph_idx, dtype=torch.long),
            'font_id': torch.tensor(font_id, dtype=torch.long),
        }


def build_loaders(opts):
    font_ids = load_font_list(opts.font_list)
    train_ids = font_ids[:opts.train_split]
    val_ids = font_ids[opts.train_split:]
    if not val_ids:
        # No validation split given — fall back to a slice of the training set.
        val_ids = train_ids[: max(1, len(train_ids) // 20)]

    train_set = GlyphReconDataset(opts.img_path, opts.sdf_path, train_ids,
                                  opts.image_size)
    val_set = GlyphReconDataset(opts.img_path, opts.sdf_path, val_ids,
                                opts.image_size)

    train_loader = data.DataLoader(
        train_set, batch_size=opts.batch_size, shuffle=True,
        num_workers=opts.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=opts.num_workers > 0,
    )
    val_loader = data.DataLoader(
        val_set, batch_size=opts.batch_size, shuffle=False,
        num_workers=opts.num_workers, pin_memory=True,
        persistent_workers=opts.num_workers > 0,
    )
    return train_loader, val_loader
