"""Calculate per-channel raster means and standard deviations."""

import argparse
from pathlib import Path

import numpy as np


def compute_mean_std(fnames, n_channels, show_histogram=False, bins=100):
    """Average per-band moments computed separately for each raster."""
    import rioxarray
    from matplotlib import pyplot as plt

    rasters_mean = []
    rasters_std = []
    all_bands = [[] for _ in range(n_channels)] if show_histogram else None
    for fname in fnames:
        raster = rioxarray.open_rasterio(fname, masked=True)
        try:
            if raster.shape[0] != n_channels:
                raise ValueError(
                    f"{fname}: expected {n_channels} channels; found {raster.shape[0]}"
                )
            data = raster.values
            rasters_mean.append(np.nanmean(data, axis=(1, 2)))
            rasters_std.append(np.nanstd(data, axis=(1, 2)))
            if show_histogram:
                for idx in range(n_channels):
                    all_bands[idx].append(data[idx].ravel())
        finally:
            raster.close()
    mean = np.mean(rasters_mean, axis=0)
    std = np.mean(rasters_std, axis=0)
    print("mean =", ", ".join(str(v) for v in mean))
    print("std =", ", ".join(str(v) for v in std))
    if show_histogram:
        for i, values in enumerate(all_bands):
            fig, ax = plt.subplots()
            v = np.concatenate(values)
            ax.hist(v[np.isfinite(v)], bins=bins)
            ax.set_title(f"Band {i + 1}")
        plt.show()
    return mean, std


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--column", default="input")
    p.add_argument("--channels", type=int, required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--histogram", action="store_true")
    args = p.parse_args()
    import pandas as pd

    df = pd.read_csv(args.csv)
    if "dset" in df:
        df = df[df.dset == args.split]
    if df.empty:
        raise ValueError("No data to calculate statistics")
    root = args.dataset_root
    compute_mean_std(
        [p if Path(p).is_absolute() else root / p for p in df[args.column]],
        args.channels,
        args.histogram,
    )


if __name__ == "__main__":
    main()
