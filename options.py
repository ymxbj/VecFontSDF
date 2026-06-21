import argparse


def get_recon_parser() -> argparse.ArgumentParser:
    """CLI args for VecFontSDF glyph reconstruction."""
    p = argparse.ArgumentParser(description="VecFontSDF — glyph reconstruction")

    # ---------------- data ----------------
    p.add_argument('--img_path', type=str, default='./data/img',
                   help='per-font dir; each dir holds {0..61}.png (named by glyph index)')
    p.add_argument('--sdf_path', type=str, default='./data/sdf',
                   help='per-font dir; each dir has sdf/ with '
                        '{codepoint}_grid.npy and {codepoint}_contour.npy')
    p.add_argument('--font_list', type=str, default='./data/font_list.txt',
                   help='a python-eval-able list of integer font ids')
    p.add_argument('--train_split', type=int, default=1000,
                   help='first N fonts -> training, rest -> validation')
    p.add_argument('--char_categories', type=int, default=52,
                   help='number of glyph classes used for conditioning: '
                        'A-Z + a-z = 52 (label = index in A..Z,a..z)')
    p.add_argument('--image_size', type=int, default=128)
    p.add_argument('--sdf_points_num', type=int, default=4000,
                   help='number of contour SDF sampling points per glyph')

    # ---------------- model ----------------
    p.add_argument('--fc_channel', type=int, default=256,
                   help='hidden dim between the image+label feature and the '
                        'curve-parameter head')
    p.add_argument('--v_dim', type=int, default=16,
                   help='number of shape primitives')
    p.add_argument('--p_dim', type=int, default=6,
                   help='number of parabolic curves intersected per primitive')

    # ---------------- loss ----------------
    p.add_argument('--gamma', type=float, default=0.02,
                   help='differentiable rasterization range')
    p.add_argument('--w_image', type=float, default=1.0)
    p.add_argument('--w_grid', type=float, default=100.0)
    p.add_argument('--w_contour', type=float, default=1000.0)
    p.add_argument('--w_regular', type=float, default=1.0)
    p.add_argument('--w_k2', type=float, default=0.1)
    p.add_argument('--k2_target', type=float, default=1.3,
                   help='lower bound on k^2, keeps primitives non-degenerate')

    # ---------------- training ----------------
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--beta1', type=float, default=0.9)
    p.add_argument('--beta2', type=float, default=0.999)
    p.add_argument('--eps', type=float, default=1e-8)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--n_iters', type=int, default=100_000)
    p.add_argument('--val_every', type=int, default=2000)
    p.add_argument('--sample_every', type=int, default=500)
    p.add_argument('--ckpt_every', type=int, default=5000)
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--multi_gpu', action='store_true')
    p.add_argument('--resume', type=str, default=None,
                   help='resume from a .pth checkpoint')

    # ---------------- logging / output ----------------
    p.add_argument('--experiment_name', type=str, default='vecfontsdf')
    p.add_argument('--out_dir', type=str, default='./experiments')
    p.add_argument('--no_tboard', dest='tboard', action='store_false')
    p.set_defaults(tboard=True)

    return p
