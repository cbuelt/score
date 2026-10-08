"""Build the BTS/Eigen-split NYU depth benchmark at NATIVE 480x640 resolution.

Matches the protocol of Liu et al. (CVPR 2023, "Single Image Depth Prediction
Made Better: A Multivariate Gaussian Take"): NYU Depth V2 at 480x640, the
standard split with 24,231 train pairs and 654 Eigen test images, depth capped
at 10 m.

* train: sync pairs from the BTS filelist. Depth is the uint16 millimeter
  PNG; the ~29% projection holes (depth==0) of the full frame are stored
  nearest-filled, with the hole mask kept as ``valid`` (see below).
* test: the 654 Eigen-split images from nyu_depth_v2_labeled.mat with the
  official dense (colorization-filled) float depths. No holes.

Storage: depth as uint16 millimeters (0..10000, holes filled), images as
uint8. Layout: depth (n,480,640,1), image (n,480,640,3), chunked per image
for lazy access. The loader converts to meters/10 in [0,1].

A ``valid`` mask (uint8, 1 = measured depth, 0 = projection hole filled at
build time) is stored alongside, so pointwise metrics can be restricted to
measured pixels; the multivariate scores cannot mask at this dimension and use
the filled field.

``--crop`` builds the Eigen-crop variant ``nyu_bts_crop_{train,test}.h5``:
rows 45:471, columns 41:601 of the 480x640 frame (426x560, d = 238,560), the
region every NYU evaluation since Eigen et al. (2014) scores on, and the one
Liu et al. train on as well (their loader zeroes the ground truth outside it).
The Kinect border strip carries two thirds of the projection holes, so cropping
at build time removes most of the filled pixels instead of scoring them.

Run (CPU only):
    python src/data/build_nyu_bts.py --src <raw download> --out data/nyu_bts [--crop]
"""

import argparse
import multiprocessing as mp
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
from PIL import Image
from scipy.ndimage import distance_transform_edt

# Raw download (BTS sync set, filelist, splits.mat, nyu_depth_v2_labeled.mat)
# and output directory; set from the command line in __main__.
SRC = Path("data/nyu_bts_raw")
OUT = Path("data/nyu_bts")
MAX_DEPTH_MM = 10000
N_WORKERS = 8

# Eigen crop (Eigen, Puhrsch & Fergus 2014), as in the BTS / NeWCRFs / Liu et
# al. evaluation code: eval_mask[45:471, 41:601] = 1 on the 480x640 frame.
EIGEN_ROWS = slice(45, 471)
EIGEN_COLS = slice(41, 601)

# Set by main(): full frame or Eigen crop.
CROP = False
PREFIX = "nyu_bts"
H, W = 480, 640


def _window(a: np.ndarray) -> np.ndarray:
    return a[EIGEN_ROWS, EIGEN_COLS] if CROP else a


def _fill_holes(d_mm: np.ndarray) -> np.ndarray:
    """Nearest-neighbor fill of hole pixels (d==0)."""
    if (d_mm > 0).all():
        return d_mm
    _, idx = distance_transform_edt(d_mm == 0, return_indices=True)
    return d_mm[tuple(idx)]


def _train_item(line: str):
    rgb_rel, depth_rel, _ = line.split()
    rgb = _window(np.array(Image.open(SRC / "sync" / rgb_rel.lstrip("/"))))
    raw = _window(np.array(Image.open(SRC / "sync" / depth_rel.lstrip("/"))))
    valid = (raw > 0).astype(np.uint8)
    d_mm = np.minimum(_fill_holes(raw), MAX_DEPTH_MM).astype(np.uint16)
    return rgb, d_mm, valid


def _write(path: Path, n: int, iterator):
    with h5py.File(path, "w") as f:
        d_img = f.create_dataset(
            "image", (n, H, W, 3), dtype="uint8", chunks=(1, H, W, 3)
        )
        d_dep = f.create_dataset(
            "depth", (n, H, W, 1), dtype="uint16", chunks=(1, H, W, 1)
        )
        # Hole mask, gzip-compressed (blobby, compresses ~50x); read only by
        # the masked pointwise metrics, never by the training loader.
        d_val = f.create_dataset(
            "valid", (n, H, W, 1), dtype="uint8", chunks=(1, H, W, 1),
            compression="gzip", compression_opts=4,
        )
        n_holes = 0
        for i, (img, dep, val) in enumerate(iterator):
            assert img.shape == (H, W, 3) and dep.shape == (H, W), (
                img.shape,
                dep.shape,
            )
            d_img[i] = img
            d_dep[i, ..., 0] = dep
            d_val[i, ..., 0] = val
            n_holes += int((val == 0).sum())
            if (i + 1) % 1000 == 0:
                print(f"  {path.name}: {i + 1}/{n}", flush=True)
    print(
        f"wrote {path} ({n} samples, {H}x{W}, filled holes "
        f"{100 * n_holes / (n * H * W):.2f}% of pixels)",
        flush=True,
    )


def build_train():
    lines = (SRC / "nyudepthv2_train_files_with_gt.txt").read_text().split("\n")
    lines = [l for l in lines if l.strip()]
    print(f"train: {len(lines)} pairs", flush=True)
    with mp.Pool(N_WORKERS) as pool:
        _write(
            OUT / f"{PREFIX}_train.h5",
            len(lines),
            pool.imap(_train_item, lines, chunksize=8),
        )


def build_test():
    ndx = sio.loadmat(SRC / "splits.mat")["testNdxs"].ravel() - 1  # 1-based
    print(f"test: {len(ndx)} images", flush=True)
    with h5py.File(SRC / "nyu_depth_v2_labeled.mat") as f:
        images, depths = f["images"], f["depths"]

        def items():
            for i in ndx:
                rgb = _window(np.transpose(images[i], (2, 1, 0)))  # (3,W,H) -> (H,W,3)
                d_m = _window(np.transpose(depths[i]))  # (W,H) -> (H,W), float meters
                d_mm = np.clip(d_m * 1000.0, 0, MAX_DEPTH_MM).astype(np.uint16)
                yield rgb, d_mm, (d_mm > 0).astype(np.uint8)

        _write(OUT / f"{PREFIX}_test.h5", len(ndx), items())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--crop", action="store_true", help="build the Eigen-crop variant")
    ap.add_argument("--src", type=Path, default=SRC, help="raw NYU Depth v2 / BTS download")
    ap.add_argument("--out", type=Path, default=OUT, help="output directory (paths.nyu_bts_dir)")
    args = ap.parse_args()
    SRC, OUT = args.src, args.out
    if args.crop:
        CROP, PREFIX = True, "nyu_bts_crop"
        H = EIGEN_ROWS.stop - EIGEN_ROWS.start
        W = EIGEN_COLS.stop - EIGEN_COLS.start
    OUT.mkdir(parents=True, exist_ok=True)
    build_test()
    build_train()
    for split in ("test", "train"):
        with h5py.File(OUT / f"{PREFIX}_{split}.h5") as f:
            d = f["depth"]
            step = max(1, d.shape[0] // 100)
            sub = d[::step].astype(np.float32) / 1000.0
            print(
                f"{split}: n={d.shape[0]} depth_m[min={sub.min():.3f} "
                f"max={sub.max():.3f} mean={sub.mean():.3f}] "
                f"frac(==10m)={(sub >= 10.0).mean():.5f}",
                flush=True,
            )
    print("BUILD_DONE", flush=True)
