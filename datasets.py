"""Geospatial patch sampling and Lightning data loading."""

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import rioxarray
import torch
from torch.utils.data import DataLoader, Dataset


class RasterDataset(Dataset):
    def __init__(
        self,
        df,
        n_samples_per_input,
        crop_len,
        ignore_index,
        transforms=None,
        norms=None,
        seed=1,
    ):
        self.df = df.copy()
        self.n_samples_per_input = n_samples_per_input
        self.crop_len = crop_len
        self.ignore_index = ignore_index
        self.transforms = transforms
        self.norms = norms or {}
        # Seed a dedicated MT19937 random generator for patch coordinates.
        self.rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(seed)))
        self.samples = []
        self.rasters = []
        required = {"label"}
        if not (required <= set(self.df.columns)):
            raise ValueError(f"CSV must contain {required}; found {list(df.columns)}")
        cols = [x for x in self.df.columns if x != "dset"]
        for row in self.df[cols].itertuples(index=False, name=None):
            self.rasters.append(
                {
                    key: rioxarray.open_rasterio(path, masked=True)
                    for key, path in zip(cols, row)
                }
            )

    def prepare_samples(self):
        self.samples.clear()
        half = self.crop_len / 2
        for scene in self.rasters:
            label = scene["label"]
            xmin, xmax = float(label.x.min()), float(label.x.max())
            ymin, ymax = float(label.y.min()), float(label.y.max())
            if xmax - xmin <= self.crop_len or ymax - ymin <= self.crop_len:
                raise ValueError(
                    "crop_len exceeds a label raster extent; adjust the INI crop_len"
                )
            count = 0
            for _ in range(self.n_samples_per_input * 100):
                if count >= self.n_samples_per_input:
                    break
                east = self.rs.uniform(xmin + half, xmax - half)
                north = self.rs.uniform(ymin + half, ymax - half)
                sub = (
                    label.sel(
                        x=slice(east - half, east + half),
                        y=slice(north + half, north - half),
                    )
                    if float(label.y[0]) > float(label.y[-1])
                    else label.sel(
                        x=slice(east - half, east + half),
                        y=slice(north - half, north + half),
                    )
                )
                if sub.size == 0 or np.isnan(sub.values).mean() >= 0.30:
                    continue
                self.samples.append((scene, east, north))
                count += 1
            if count == 0:
                raise RuntimeError(
                    "No valid crops found. Check raster bounds/crop_len and labels."
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scene, east, north = self.samples[idx]
        half = self.crop_len / 2
        result = {}
        for key, raster in scene.items():
            ys = (
                slice(north + half, north - half)
                if float(raster.y[0]) > float(raster.y[-1])
                else slice(north - half, north + half)
            )
            crop = raster.sel(x=slice(east - half, east + half), y=ys).values.copy()
            if key == "label":
                crop = (
                    np.nan_to_num(crop, nan=self.ignore_index)
                    .squeeze(axis=0)
                    .astype(np.int64)
                )
                result[key] = torch.from_numpy(crop)
            else:
                # Replace invalid pixels with each channel's patch-local mean
                # before normalization.
                # Fall back to the fold mean if an entire channel is invalid.
                tensor = torch.from_numpy(crop.astype("float32"))
                for channel in range(tensor.shape[0]):
                    valid = torch.isfinite(tensor[channel])
                    if not valid.all():
                        fallback = (
                            float(self.norms[key][0][channel])
                            if key in self.norms
                            else 0.0
                        )
                        replacement = (
                            tensor[channel][valid].mean() if valid.any() else fallback
                        )
                        tensor[channel][~valid] = replacement
                if key in self.norms:
                    means, stds = self.norms[key]
                    mean_tensor = torch.tensor(means, dtype=tensor.dtype).view(-1, 1, 1)
                    std_tensor = torch.tensor(stds, dtype=tensor.dtype).view(-1, 1, 1)
                    tensor = torch.nan_to_num((tensor - mean_tensor) / std_tensor)
                result[key] = tensor
        return result


class GeoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df,
        n_samples_per_input,
        crop_len,
        norms,
        ignore_index,
        seed,
        batch_size,
        **kwargs,
    ):
        super().__init__()
        self.df = df
        self.n_samples_per_input = n_samples_per_input
        self.crop_len = crop_len
        self.norms = norms
        self.ignore_index = ignore_index
        self.seed = seed
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage not in (None, "fit"):
            return
        for split, attr, increment in [("train", "train_ds", 0), ("val", "val_ds", 1)]:
            df = self.df.loc[self.df.dset == split].drop(columns="dset")
            if df.empty:
                raise ValueError(f"No {split} rows in training CSV")
            setattr(
                self,
                attr,
                RasterDataset(
                    df,
                    self.n_samples_per_input,
                    self.crop_len,
                    self.ignore_index,
                    norms=self.norms,
                    seed=self.seed,
                ),
            )

    def train_dataloader(self):
        self.train_ds.prepare_samples()
        # Patch coordinates are randomized before batches are assembled.
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

    def val_dataloader(self):
        self.val_ds.prepare_samples()
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=0)
