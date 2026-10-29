"""Single entry point for SA single-band, early, late and gated experiments."""

import argparse
import json

from runtime import (
    Settings,
    cli,
    device_name,
    load_config,
    rebase_csv_paths,
    route_late_inputs,
    set_seed,
)


def train(args):
    cfg = load_config(args)
    settings = Settings(args, cfg)
    settings.assert_environment()

    import platform

    import numpy as np
    import pandas as pd
    import pytorch_lightning as pl
    import torch
    import torchvision
    from datasets import GeoDataModule
    from losses import count_training_classes, inverse_sqrt_class_weights
    from models import DeepLabLikeMultiStreamDict
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

    device = device_name(
        "cpu" if settings.mode == "dev" and args.device == "auto" else args.device
    )
    set_seed(settings.seed)
    dev = settings.mode == "dev"
    if dev:
        print(
            "Development mode: first fold by default, one epoch, one sample per scene, no ImageNet download."
        )
    for fold in settings.folds:
        # Keep one continuous random-number stream across the configured folds.
        print(
            f"===== {settings.experiment} / {fold} / {settings.mode} / seed {settings.seed} ====="
        )
        output = settings.output(fold)
        output.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(settings.csv(fold))
        if "dset" not in df.columns:
            raise ValueError("CSV needs dset column with train and val values")
        df = rebase_csv_paths(df, settings.dataset_root, settings.default_dataset_root)
        if settings.experiment.startswith("single_"):
            if "input" not in df.columns and "input-1" in df.columns:
                df = df.rename(columns={"input-1": "input"})
            df = df[["dset", "input", "label"]]
        elif settings.model_experiment == "late":
            df = route_late_inputs(df, settings.input_bands)
        else:
            df = df[["dset", "input-1", "input-2", "label"]]
        mean, std = settings.stats(fold)
        if settings.experiment.startswith("single_"):
            norms = {"input": (mean, std)}
        else:
            norms = {"input-1": (mean[:2], std[:2]), "input-2": (mean[2:], std[2:])}
        batch_size = 1 if dev else cfg.getint("train", "batch_size")
        crop_len = cfg.getfloat("datamodule", "crop_len")
        samples = 1 if dev else cfg.getint("datamodule", "n_samples_per_input")
        max_epochs = 1 if dev else cfg.getint("train", "max_epochs")
        dm = GeoDataModule(
            df,
            samples,
            crop_len,
            norms,
            cfg.getint("train", "ignore_index"),
            settings.seed,
            batch_size,
        )
        counts, weights = None, None
        if settings.class_weighted:
            counts = count_training_classes(
                df.loc[df.dset == "train", "label"],
                cfg.getint("model", "num_classes"),
                cfg.getint("train", "ignore_index"),
            )
            weights = inverse_sqrt_class_weights(counts)
            print(
                f"Training class counts: {counts.tolist()} | CE weights: {weights.tolist()}"
            )
        model = DeepLabLikeMultiStreamDict(
            experiment=settings.model_experiment,
            num_classes=cfg.getint("model", "num_classes"),
            pretrained=cfg.getboolean("model", "pretrained") and not dev,
            class_weights=weights,
            lr=cfg.getfloat("train", "lr"),
            reduce_lr_patience=cfg.getint("train", "reduce_lr_patience"),
            ignore_index=cfg.getint("train", "ignore_index"),
            gated_switch_epoch=cfg.getint("model", "gated_switch_epoch", fallback=15),
            gated_target_band=settings.gated_target_band,
            ablation=settings.ablation,
        )
        checkpoint = ModelCheckpoint(
            dirpath=output,
            filename="best_weights",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
        )
        stopper = EarlyStopping(
            monitor="val_loss",
            patience=cfg.getint("train", "patience"),
            min_delta=1e-4,
            mode="min",
        )
        trainer = pl.Trainer(
            accelerator="gpu" if device == "cuda" else device,
            devices=1,
            max_epochs=max_epochs,
            min_epochs=1 if dev else cfg.getint("train", "min_epochs"),
            gradient_clip_val=1.0,
            log_every_n_steps=1,
            default_root_dir=str(output),
            callbacks=[checkpoint, stopper],
            logger=CSVLogger(str(output), name="training"),
            deterministic=True,
            reload_dataloaders_every_n_epochs=cfg.getint(
                "train", "reload_every_n_epochs"
            ),
            enable_progress_bar=True,
        )
        trainer.fit(model, datamodule=dm)
        if checkpoint.best_model_path:
            saved = torch.load(
                checkpoint.best_model_path, map_location="cpu", weights_only=False
            )
            model.load_state_dict(saved["state_dict"])
        torch.save(
            {
                "state_dict": model.cpu().state_dict(),
                "experiment": settings.experiment,
                "input_bands": list(settings.input_bands),
                "seed": settings.seed,
                "fold": fold,
                "gated_target_band": settings.gated_target_band,
                "ablation": settings.ablation,
                "class_weighted": settings.class_weighted,
                "class_weights": weights.tolist() if weights is not None else None,
            },
            output / "model.pth",
        )
        # Store a compact run manifest for provenance and eval reproducibility.
        manifest = {
            "experiment": settings.experiment,
            "input_bands": list(settings.input_bands),
            "fold": fold,
            "seed": settings.seed,
            "mode": settings.mode,
            "device": device,
            "gated_target_band": settings.gated_target_band,
            "ablation": settings.ablation,
            "class_weighted": settings.class_weighted,
            "class_counts": counts.tolist() if counts is not None else None,
            "class_weights": weights.tolist() if weights is not None else None,
            "csv": str(settings.csv(fold)),
            "epochs": max_epochs,
            "batch_size": batch_size,
            "crop_len": crop_len,
            "samples_per_input": samples,
            "normalization": {"mean": mean, "std": std},
            "versions": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "pytorch": torch.__version__,
                "torchvision": torchvision.__version__,
                "lightning": pl.__version__,
                "numpy": np.__version__,
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
            },
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        }
        (output / "run.json").write_text(json.dumps(manifest, indent=2))
        with (output / "train.cfg").open("w") as f:
            cfg.write(f)
        print(f"Saved {output / 'model.pth'}")


def main():
    parser = cli(argparse.ArgumentParser(description=__doc__))
    train(parser.parse_args())


if __name__ == "__main__":
    main()
