"""Evaluate a configured segmentation experiment and summarize scene metrics."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from runtime import Settings, cli, device_name, load_config, set_seed


def f1_metrics_from_cm(cm):
    """Compute F1 for a confusion matrix with explicit absent-class handling.

    Per-class F1 is undefined when its class has no true pixels (NaN).
    The scene macro F1 averages only classes represented in ground truth.
    Weighted F1 uses ground-truth support. This prevents a scene with no
    example of a rare class from counting as a zero-F1 observation for it.
    """
    cm = np.asarray(cm, dtype=np.float64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("Expected a square confusion matrix")
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = np.diag(cm)
    observed = support > 0
    raw_f1 = np.divide(
        2 * tp,
        support + predicted,
        out=np.zeros_like(tp),
        where=(support + predicted) != 0,
    )
    return {
        "weighted_f1": float(raw_f1 @ support / support.sum())
        if support.sum()
        else float("nan"),
        "macro_f1": float(raw_f1[observed].mean()) if observed.any() else float("nan"),
        "per_class_f1": np.where(observed, raw_f1, np.nan),
        "support": support.astype(np.int64),
    }


def format_from_cm(cm, class_names=None):
    """Format pixel-pooled metrics; omit ground-truth-absent classes from macro F1."""
    cm = np.asarray(cm)
    scores = f1_metrics_from_cm(cm)
    tp = np.diag(cm).astype(float)
    support = scores["support"].astype(float)
    predicted = cm.sum(axis=0).astype(float)
    observed = support > 0
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
    union = support + predicted - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union != 0)
    total = support.sum()
    weighted = lambda a: float((a * support).sum() / total) if total else float("nan")
    macro = lambda a: float(a[observed].mean()) if observed.any() else float("nan")
    acc = float(tp.sum() / total) if total else float("nan")
    micro_iou = (
        float(tp.sum() / (total + predicted.sum() - tp.sum()))
        if total
        else float("nan")
    )
    names = class_names if class_names is not None else [str(i) for i in range(len(tp))]
    lines = ["              precision    recall  f1-score   support", ""]
    for k in range(len(tp)):
        name = str(names[k])
        f1_value = scores["per_class_f1"][k]
        f1_text = f"{f1_value:.2f}" if np.isfinite(f1_value) else "N/A"
        lines.append(
            f"{name:>14}    {precision[k]:.2f}      {recall[k]:.2f}      {f1_text}    {int(support[k])}"
        )
    lines.extend(
        [
            "",
            f"    accuracy                           {acc:.2f}    {int(total)}",
            f"   macro avg       {macro(precision):.2f}      {macro(recall):.2f}      {scores['macro_f1']:.2f}    {int(total)}",
            f"weighted avg       {weighted(precision):.2f}      {weighted(recall):.2f}      {scores['weighted_f1']:.2f}    {int(total)}",
            "",
            "Jaccard Index:",
            f"micro: {micro_iou:.4f}",
            f"macro: {macro(iou):.4f}",
            f"weighted: {weighted(iou):.4f}",
            "",
            "Macro F1 uses ground-truth-present classes; absent class F1 is N/A.",
            "Confusion Matrix (row-normalized):",
            np.array2string(cm / np.maximum(support[:, None], 1), precision=3),
        ]
    )
    return "\n".join(lines) + "\n"


def weighted_f1_from_cm(cm):
    """Compute the support-weighted F1 score for one confusion matrix."""
    return f1_metrics_from_cm(cm)["weighted_f1"]


def summarize_f1(values):
    """Return n, mean, sample SD for finite scores; no invented zeros for absent classes."""
    valid = np.asarray(values, dtype=float)
    valid = valid[np.isfinite(valid)]
    if not len(valid):
        return 0, float("nan"), float("nan")
    return (
        len(valid),
        float(valid.mean()),
        float(valid.std(ddof=1)) if len(valid) > 1 else float("nan"),
    )


def write_f1_statistics(scenes, pooled_cm, average_dir, class_names=None):
    """Write scene-level and pooled F1 summaries without needing raster dependencies.

    scenes: sequence of (fold, scene, confusion_matrix) records.
    Outputs: metrics_average.txt, scene_f1.csv, scene_weighted_f1.csv,
    and f1_summary.csv. Scene-level averages give every scene equal weight.
    """
    average_dir = Path(average_dir)
    average_dir.mkdir(parents=True, exist_ok=True)
    count = len(pooled_cm)
    names = class_names if class_names is not None else [str(k) for k in range(count)]
    scored = [
        (fold, scene, f1_metrics_from_cm(matrix)) for fold, scene, matrix in scenes
    ]
    metric_values = {
        "weighted_f1": [r["weighted_f1"] for _, _, r in scored],
        "macro_f1": [r["macro_f1"] for _, _, r in scored],
    }
    for k in range(count):
        metric_values[f"f1_class_{k}"] = [r["per_class_f1"][k] for _, _, r in scored]
    pooled = f1_metrics_from_cm(pooled_cm)
    pooled_values = {
        "weighted_f1": pooled["weighted_f1"],
        "macro_f1": pooled["macro_f1"],
    }
    for k in range(count):
        pooled_values[f"f1_class_{k}"] = pooled["per_class_f1"][k]

    def csv_value(v):
        return f"{v:.8f}" if np.isfinite(v) else ""

    with (average_dir / "scene_f1.csv").open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(
            ["fold", "scene", "weighted_f1", "macro_f1"]
            + [f"f1_class_{k}" for k in range(count)]
            + [f"support_class_{k}" for k in range(count)]
        )
        for fold, scene, r in scored:
            writer.writerow(
                [fold, scene, csv_value(r["weighted_f1"]), csv_value(r["macro_f1"])]
                + [csv_value(v) for v in r["per_class_f1"]]
                + r["support"].tolist()
            )

    with (average_dir / "scene_weighted_f1.csv").open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["fold", "scene", "weighted_f1"])
        for fold, scene, r in scored:
            writer.writerow([fold, scene, csv_value(r["weighted_f1"])])

    with (average_dir / "f1_summary.csv").open("w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(
            [
                "metric",
                "class_name",
                "n_scenes",
                "scene_mean",
                "scene_sd",
                "pooled_f1",
                "pooled_support",
            ]
        )
        for key, values in metric_values.items():
            n, mean, sd = summarize_f1(values)
            k = int(key.split("_")[-1]) if key.startswith("f1_class_") else None
            writer.writerow(
                [
                    key,
                    names[k] if k is not None else "",
                    n,
                    csv_value(mean),
                    csv_value(sd),
                    csv_value(pooled_values[key]),
                    int(pooled["support"][k])
                    if k is not None
                    else int(pooled["support"].sum()),
                ]
            )

    with (average_dir / "metrics_average.txt").open("w") as out:
        _, weighted_mean, weighted_sd = summarize_f1(metric_values["weighted_f1"])
        _, macro_mean, macro_sd = summarize_f1(metric_values["macro_f1"])
        out.write(f"Scene-level mean weighted F1: {weighted_mean:.6f}\n")
        out.write(f"Scene-level weighted F1 standard deviation: {weighted_sd:.6f}\n")
        out.write(f"Scene-level mean macro F1: {macro_mean:.6f}\n")
        out.write(f"Scene-level macro F1 standard deviation: {macro_sd:.6f}\n\n")
        out.write(f"Scene-level F1 across {len(scored)} evaluated scenes (one seed):\n")
        for key, values in metric_values.items():
            n, mean, sd = summarize_f1(values)
            label = (
                key
                if not key.startswith("f1_class_")
                else f"{key} ({names[int(key.split('_')[-1])]})"
            )
            mean_text = f"{mean:.6f}" if np.isfinite(mean) else "N/A"
            sd_text = f"{sd:.6f}" if np.isfinite(sd) else "N/A"
            out.write(
                f"  {label}: mean={mean_text}, sample_SD={sd_text}, n_scenes={n}\n"
            )
        out.write(
            "Absent ground-truth classes are excluded (N/A) from scene macro and per-class means.\n"
        )
        out.write(
            "Scene mean weights each scene equally; pooled scores weight the underlying pixels.\n"
        )
        out.write(
            "These are descriptive statistics for one seed, NOT independent-seed confidence intervals.\n\n"
        )
        out.write(
            f"Pooled pixel-based metrics over {len(scored)} evaluated scenes:\n\n"
        )
        out.write(format_from_cm(pooled_cm, names))


def checkpoint_class_weights(checkpoint, class_weighted, num_classes, checkpoint_path):
    """Verify the CE variant matches the requested evaluation and retrieve its weights."""
    import torch

    if checkpoint.get("class_weighted", False) != class_weighted:
        raise ValueError(
            f"Checkpoint class-weighted setting differs from evaluation flag: {checkpoint_path}"
        )
    values = checkpoint.get("class_weights")
    if not class_weighted:
        if values is not None:
            raise ValueError(
                f"Unexpected class weights for standard CE: {checkpoint_path}"
            )
        return None
    if values is None or len(values) != num_classes:
        raise ValueError(f"Checkpoint missing valid class weights: {checkpoint_path}")
    weights = torch.as_tensor(values, dtype=torch.float32)
    if not torch.isfinite(weights).all() or (weights < 0).any() or not weights.any():
        raise ValueError(f"Checkpoint has invalid class weights: {checkpoint_path}")
    return weights


def verify_training_normalization(output_dir, means, stds):
    """Reject evaluating a checkpoint with statistics different from its training run."""
    path = Path(output_dir) / "run.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing training manifest for normalization check: {path}"
        )
    data = json.loads(path.read_text())
    trained = data.get("normalization", {})
    if trained.get("mean") != list(means) or trained.get("std") != list(stds):
        raise ValueError(
            f"Evaluation normalization differs from training at {path}; "
            "match [datamodule] mean/std to the training config."
        )


def save_confusion(matrix, path, names=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    cm = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(
        cm, display_labels=names or [str(x) for x in range(len(cm))]
    ).plot(ax=ax, values_format=".2f", cmap="Blues", colorbar=False)
    ax.set_title("Normalized confusion matrix")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def normalize(data, means, stds):
    import torch

    mean = torch.tensor(means, dtype=torch.float32).reshape(1, -1, 1, 1)
    std = torch.tensor(stds, dtype=torch.float32).reshape(1, -1, 1, 1)
    return torch.nan_to_num(
        (torch.from_numpy(data.astype(np.float32))[None] - mean) / std
    )


def inference(model, imgs, settings, band, device):
    """Run full-scene inference using the configured sensor(s).

    Normalize complete rasters and replace nonfinite normalized values with zero,
    run one model forward pass, and apply softmax across the class dimension.
    This function deliberately has no tiling option.
    """
    import torch

    means, stds = settings
    if model.experiment.startswith("single_"):
        tensors = [normalize(imgs[0], means, stds).to(device)]
    else:
        tensors = [
            normalize(image, means[2 * i : 2 * i + 2], stds[2 * i : 2 * i + 2]).to(
                device
            )
            for i, image in enumerate(imgs)
        ]
    with torch.inference_mode():
        return torch.softmax(model(tensors, band=band), dim=1)[0].cpu().numpy()


def save_raster(src, data, path, nodata):
    import rioxarray  # registers the rio extension
    import xarray as xr

    result = xr.DataArray(
        data,
        dims=("band", "y", "x"),
        coords={
            "band": np.arange(1, data.shape[0] + 1),
            "y": src.y.values,
            "x": src.x.values,
        },
    )
    if src.rio.crs is not None:
        result.rio.write_crs(src.rio.crs, inplace=True)
    result.rio.write_transform(src.rio.transform(), inplace=True)
    result.rio.write_nodata(nodata, inplace=True)
    result.rio.to_raster(path)


def evaluation_sources(settings, band):
    """Select data sources to read and forward to each encoder."""
    indices = [0 if b == "alos2" else 1 for b in settings.input_bands]
    if settings.experiment == "gated" and band != "both":
        required = [0 if band == "alos2" else 1]
    else:
        required = sorted(set(indices))
    return indices, required


def evaluate(args):
    cfg = load_config(args, evaluation=True)
    settings = Settings(args, cfg)
    settings.assert_environment()
    import rioxarray
    import torch
    from models import DeepLabLikeMultiStreamDict
    from sklearn.metrics import confusion_matrix

    device = device_name(
        "cpu" if settings.mode == "dev" and args.device == "auto" else args.device
    )
    set_seed(settings.seed)
    n_classes = cfg.getint("model", "num_classes")
    pooled_cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    evaluated_scenes = 0
    scene_results = []
    band = settings.eval_band
    class_names = [
        x.strip()
        for x in cfg.get("eval", "class_names", fallback="").split(",")
        if x.strip()
    ]
    if len(class_names) != n_classes:
        class_names = None
    scenes_per_fold = 1 if settings.mode == "dev" else 2
    for fold in settings.folds:
        output = settings.output(fold)
        eval_dir = output / f"evaluation_{band}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        (eval_dir / "metrics.txt").write_text(
            ""
        )  # Reset report for this evaluation run.
        saved_path = output / "model.pth"
        if not saved_path.exists():
            raise FileNotFoundError(
                f"Missing trained model: {saved_path}; run main.py first"
            )
        weights = torch.load(saved_path, map_location="cpu", weights_only=False)
        if weights.get("experiment") != settings.experiment:
            raise ValueError(f"Checkpoint experiment mismatch: {saved_path}")
        if settings.model_experiment == "late":
            # Input routing is part of the checkpoint identity: the same
            # architecture can use L+C, L+L, or C+C.
            if weights.get("input_bands") != list(settings.input_bands):
                raise ValueError(f"Checkpoint sensor routing mismatch: {saved_path}")
        if settings.experiment == "gated":
            if weights.get("gated_target_band", "alos2") != settings.gated_target_band:
                raise ValueError(f"Checkpoint gated target mismatch: {saved_path}")
            if weights.get("ablation", "none") != settings.ablation:
                raise ValueError(f"Checkpoint ablation mismatch: {saved_path}")
        class_weights = checkpoint_class_weights(
            weights, settings.class_weighted, n_classes, saved_path
        )
        model = DeepLabLikeMultiStreamDict(
            experiment=settings.model_experiment,
            num_classes=n_classes,
            pretrained=False,
            class_weights=class_weights,
            lr=cfg.getfloat("train", "lr"),
            reduce_lr_patience=cfg.getint("train", "reduce_lr_patience"),
            ignore_index=cfg.getint("train", "ignore_index"),
            gated_switch_epoch=cfg.getint("model", "gated_switch_epoch", fallback=15),
            gated_target_band=settings.gated_target_band,
            ablation=settings.ablation,
        )
        model.load_state_dict(weights["state_dict"])
        model.eval().to(device)
        alos2, s1, labels = settings.scenes(fold)
        means, stds = settings.stats(fold)
        verify_training_normalization(output, means, stds)
        for idx in range(scenes_per_fold):
            # A duplicated-band late model reads only its selected source, then
            # sends the same normalized sensor data to two independent encoders.
            source_indices, required = evaluation_sources(settings, band)
            paths = (alos2[idx], s1[idx])
            rasters = {
                j: rioxarray.open_rasterio(paths[j], masked=True) for j in required
            }
            src = rasters[required[0]]
            primary = paths[required[0]]
            imgs = []
            used = []
            for j in source_indices:
                if j in rasters:
                    img = rasters[j].values.astype("float32")
                    if img.shape[-2:] != src.shape[-2:]:
                        raise ValueError(
                            "Input raster dimensions do not match; pre-align rasters"
                        )
                    used.append(img)
                else:
                    # Gated target-only inference skips the unused sensor.
                    img = np.full((2,) + src.shape[-2:], np.nan, dtype="float32")
                imgs.append(img)
            mask = np.zeros(src.shape[-2:], dtype=bool)
            for img in used:
                mask |= np.isnan(img).any(axis=0)
            # Singles pull the proper two statistics for their one selected modality.
            norm_stats = (means, stds)
            probs = inference(model, imgs, norm_stats, band, device)
            probs[:, mask] = np.nan
            pred = np.nan_to_num(probs, nan=-1).argmax(axis=0).astype("uint8")
            pred[mask] = 241
            stem = primary.stem
            if cfg.getboolean("eval", "output_probabilities", fallback=True):
                save_raster(
                    src, probs.astype("float32"), eval_dir / f"pred-{stem}.tif", np.nan
                )
            save_raster(src, pred[None], eval_dir / f"class-{stem}.tif", 241)
            label = rioxarray.open_rasterio(labels[idx], masked=True)
            true = label.values[0]
            if true.shape != pred.shape:
                raise ValueError(
                    f"Label and input shapes differ for {primary}: {true.shape} vs {pred.shape}"
                )
            valid = ~(
                mask | np.isnan(true) | (true == cfg.getint("eval", "ignore_index"))
            )
            correct = np.full_like(pred, 241)
            correct[valid] = (pred[valid] == true[valid]).astype("uint8")
            save_raster(src, correct[None], eval_dir / f"correct-prim-{stem}.tif", 241)
            if valid.any():
                yt, yp = true[valid].astype(int), pred[valid].astype(int)
                # Reject unexpected label codes instead of silently excluding
                # them from the confusion matrix and inflating F1.
                if ((yt < 0) | (yt >= n_classes)).any():
                    unexpected = np.unique(yt[(yt < 0) | (yt >= n_classes)])
                    raise ValueError(
                        f"Unexpected ground-truth class codes in {labels[idx]}: "
                        f"{unexpected.tolist()}"
                    )
                matrix = confusion_matrix(yt, yp, labels=list(range(n_classes)))
                pooled_cm += matrix
                evaluated_scenes += 1
                scene_results.append((fold, stem, matrix))
                with (eval_dir / "metrics.txt").open("a") as f:
                    f.write(f"\n{stem} vs {labels[idx].stem}:\n")
                    f.write(format_from_cm(matrix, class_names))
                save_confusion(
                    matrix, eval_dir / f"confusion_matrix-{stem}.pdf", class_names
                )
            for raster in rasters.values():
                raster.close()
            label.close()
            print(f"Evaluated {fold}: {stem}, {int(valid.sum())} labeled pixels")
        with (eval_dir / "evaluate.cfg").open("w") as f:
            cfg.write(f)

    if evaluated_scenes:
        cm = pooled_cm
        average_dir = (
            settings.data_root
            / "SA"
            / "metrics"
            / settings.experiment
            / settings.mode
            / f"seed_{settings.seed}"
        )
        if settings.ablation != "none":
            average_dir = (
                average_dir / f"ablation_{settings.ablation.replace('-', '_')}"
            )
        if settings.class_weighted:
            average_dir = average_dir / "class_weighted"
        average_dir = average_dir / f"evaluation_{band}"
        average_dir.mkdir(parents=True, exist_ok=True)
        write_f1_statistics(scene_results, cm, average_dir, class_names)
        save_confusion(cm, average_dir / "confusion_matrix_average.pdf", class_names)
        print(f"Averaged metrics: {average_dir / 'metrics_average.txt'}")
    else:
        print("No valid labeled pixels found: metrics could not be calculated.")


def main():
    parser = cli(argparse.ArgumentParser(description=__doc__))
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
