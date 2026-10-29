"""Configuration, model, metric, and CLI tests."""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from evaluate import evaluation_sources, format_from_cm, weighted_f1_from_cm
from models import DeepLabLikeMultiStreamDict, GatedFusion
from runtime import (
    REPO_ROOT,
    Settings,
    after_results,
    cli,
    load_config,
    rebase_csv_paths,
    route_late_inputs,
)


@pytest.mark.parametrize(
    "stem,experiment",
    (
        ("single_alos", "single_alos2"),
        ("single_s1", "single_s1"),
        ("early", "early"),
        ("late", "late"),
        ("late_alos2_alos2", "late_alos2_alos2"),
        ("late_s1_s1", "late_s1_s1"),
        ("gated_alos2", "gated"),
        ("gated_s1", "gated"),
    ),
)
@pytest.mark.parametrize("evaluation", (False, True))
def test_configs_and_env_paths(stem, experiment, evaluation, monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_ROOT", "results")
    monkeypatch.setenv("DATASET_ROOT", str(tmp_path / "Belgica_Bank"))
    name = ("eval_" if evaluation else "") + "SA_" + stem + ".ini"
    if evaluation and stem == "single_alos":
        name = "eval_SA_single_alos2.ini"
    args = cli(argparse.ArgumentParser()).parse_args(
        ["--config-file", name, "--dev", "--seed", "42"]
    )
    cfg = load_config(args, evaluation=evaluation)
    settings = Settings(args, cfg)
    assert settings.experiment == experiment
    assert settings.folds == ["jan"]
    assert settings.mode == "dev"
    assert (
        settings.csv("jan")
        == tmp_path
        / "Belgica_Bank"
        / f"BB_IO-SA-{'alos2' if stem == 'single_alos' else 's1' if stem == 'single_s1' else 'alos2-s1'}-jan.csv"
    )
    assert (
        settings.output("jan")
        == REPO_ROOT
        / "results"
        / "SA"
        / settings.output_dirs[0].split("SA/")[1]
        / "dev"
        / "seed_42"
    )
    assert settings.scenes("jan")[0][0].is_relative_to(tmp_path / "Belgica_Bank")
    assert str(settings.output("jan")).endswith("dev/seed_42")
    mean, std = settings.stats("jan")
    assert len(mean) == len(std) == (2 if experiment.startswith("single") else 4)
    assert len(settings.scenes("jan")[2]) == 2
    if stem == "gated_s1":
        assert settings.gated_target_band == "s1"
        assert settings.eval_band == ("s1" if evaluation else "s1")


@pytest.mark.parametrize(
    "stem,bands",
    (
        ("late_alos2_alos2", ("alos2", "alos2")),
        ("late_s1_s1", ("s1", "s1")),
        ("late", ("alos2", "s1")),
    ),
)
def test_late_routing_preserves_independent_encoder_capacity(
    stem, bands, monkeypatch, tmp_path
):
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "output"))
    monkeypatch.setenv("DATASET_ROOT", str(tmp_path / "dataset"))
    args = cli(argparse.ArgumentParser()).parse_args(["-c", f"SA_{stem}.ini"])
    settings = Settings(args, load_config(args))
    assert settings.model_experiment == "late"
    assert settings.input_bands == bands
    df = pd.DataFrame(
        {
            "dset": ["train", "val"],
            "input-1": ["L1", "L2"],
            "input-2": ["C1", "C2"],
            "label": ["Y1", "Y2"],
        }
    )
    routed = route_late_inputs(df, settings.input_bands)
    assert routed["input-1"].tolist() == (
        ["L1", "L2"] if bands[0] == "alos2" else ["C1", "C2"]
    )
    assert routed["input-2"].tolist() == (
        ["L1", "L2"] if bands[1] == "alos2" else ["C1", "C2"]
    )
    for fold in settings.folds:
        mean, std = settings.stats(fold)
        assert len(mean) == len(std) == 4
        if bands[0] == bands[1]:
            assert mean[:2] == mean[2:]
            assert std[:2] == std[2:]
    model = DeepLabLikeMultiStreamDict(
        experiment=settings.model_experiment, pretrained=False
    )
    # Both encoders have independent trainable weights, even for identical inputs.
    assert (
        model.streams[0]["conv1"].weight.data_ptr()
        != model.streams[1]["conv1"].weight.data_ptr()
    )
    assert sum(p.numel() for p in model.parameters()) == sum(
        p.numel()
        for p in DeepLabLikeMultiStreamDict(
            experiment="late", pretrained=False
        ).parameters()
    )
    eval_args = cli(argparse.ArgumentParser()).parse_args(["-c", f"eval_SA_{stem}.ini"])
    eval_settings = Settings(eval_args, load_config(eval_args, evaluation=True))
    assert settings.output("jan") == eval_settings.output("jan")
    assert settings.stats("jan") == eval_settings.stats("jan")
    assert settings.input_bands == eval_settings.input_bands
    source_indices, required = evaluation_sources(eval_settings, "both")
    expected_indices = [0 if b == "alos2" else 1 for b in bands]
    assert source_indices == expected_indices
    assert required == sorted(set(expected_indices))
    if stem != "late":
        assert (
            settings.output("jan")
            != tmp_path / "output" / "SA" / "alos2-s1-jan-late" / "full" / "seed_0"
        )


def test_same_band_routes_keep_matching_patches():
    df = pd.DataFrame(
        {"dset": ["train"], "input-1": ["L"], "input-2": ["C"], "label": ["Y"]}
    )
    l = route_late_inputs(df, ("alos2", "alos2"))
    c = route_late_inputs(df, ("s1", "s1"))
    assert l.loc[0, "input-1"] == l.loc[0, "input-2"] == "L"
    assert c.loc[0, "input-1"] == c.loc[0, "input-2"] == "C"
    with pytest.raises(ValueError, match="missing"):
        route_late_inputs(df.drop(columns="input-2"), ("s1", "s1"))


def test_no_env_uses_configured_windows_paths(monkeypatch):
    monkeypatch.delenv("OUTPUT_ROOT", raising=False)
    monkeypatch.delenv("DATASET_ROOT", raising=False)
    args = cli(argparse.ArgumentParser()).parse_args(
        ["--config-file", "SA_single_alos.ini"]
    )
    cfg = load_config(args)
    settings = Settings(args, cfg)
    assert settings.seed == 0
    assert settings.mode == "full"
    assert len(settings.folds) == 5
    assert (
        str(settings.csv("jan")).replace("\\", "/")
        == "D:/jema2085/data/Belgica_Bank/results/BB_IO-SA-alos2-jan.csv"
    )
    assert (
        str(settings.output("jan"))
        .replace("\\", "/")
        .endswith("Belgica_Bank/results/SA/alos2-jan/full/seed_0")
    )


def test_cli_requires_config_and_has_focused_options():
    evaluator = cli(argparse.ArgumentParser())
    trainer = cli(argparse.ArgumentParser())
    with pytest.raises(SystemExit):
        evaluator.parse_args([])
    expected = {
        "-h",
        "--help",
        "-c",
        "--config-file",
        "--dev",
        "--ablation",
        "--seed",
        "--device",
        "--class-weighted",
    }
    for parser in (trainer, evaluator):
        options = {opt for action in parser._actions for opt in action.option_strings}
        assert options == expected
        args = parser.parse_args(["--config-file", "SA_early.ini"])
        assert args.seed == 0 and not args.dev


def test_absolute_paths_rebased(monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "output"))
    monkeypatch.setenv("DATASET_ROOT", str(tmp_path / "dataset"))
    args = cli(argparse.ArgumentParser()).parse_args(["-c", "SA_late.ini"])
    cfg = load_config(args)
    cfg["io"]["run_dirs"] = cfg["io"]["run_dirs"].replace(
        "results/SA/", "D:/jema2085/data/Belgica_Bank/results/SA/"
    )
    cfg["io"]["csv_files"] = cfg["io"]["csv_files"].replace(
        "results/BB_", "D:/jema2085/data/Belgica_Bank/results/BB_"
    )
    settings = Settings(args, cfg)
    assert settings.csv("jan") == tmp_path / "dataset" / "BB_IO-SA-alos2-s1-jan.csv"
    assert (
        settings.output("jan")
        == tmp_path / "output" / "SA" / "alos2-s1-jan-late" / "full" / "seed_0"
    )
    assert after_results("C:/foo/results/SA/hi") == Path("SA/hi")


def test_windows_raster_csv_paths(tmp_path):
    source = pd.DataFrame(
        {
            "dset": ["train"],
            "input": ["D:\\jema2085\\data\\Belgica_Bank\\denoised_resampled\\test.tif"],
            "label": ["labels_rasterized/SA/label.tif"],
        }
    )
    mapped = rebase_csv_paths(source, tmp_path)
    assert mapped["input"][0] == str(tmp_path / "denoised_resampled/test.tif")
    assert mapped["label"][0] == str(tmp_path / "labels_rasterized/SA/label.tif")
    assert mapped["dset"][0] == "train"


def test_averaged_confusion_text():
    cm = np.array([[2, 1], [1, 4]], dtype=int)
    summary = format_from_cm(cm)
    assert "Jaccard Index:" in summary
    assert "micro:" in summary
    assert "Confusion Matrix" in summary


def test_gated_centering_math():
    torch.manual_seed(0)
    fusion = GatedFusion(channels=1)
    with torch.no_grad():
        fusion.gate[0].weight.fill_(0)
        fusion.gate[0].bias.fill_(1.0986122886681098)  # sigmoid = 0.75
    l = torch.ones((1, 1, 2, 2))
    c = torch.zeros_like(l)
    assert torch.allclose(fusion(l, c), torch.full_like(l, 0.71875), atol=1e-6)
    uncentered = GatedFusion(channels=1, gate_centering=False)
    with torch.no_grad():
        uncentered.gate[0].weight.zero_()
        uncentered.gate[0].bias.fill_(1.0986122886681098)
    assert torch.allclose(uncentered(l, c), torch.full_like(l, 0.75), atol=1e-6)


@pytest.mark.parametrize(
    "experiment", ("single_alos2", "single_s1", "early", "late", "gated")
)
def test_model_forward_and_checkpoint(experiment, tmp_path):
    torch.set_num_threads(2)
    model = DeepLabLikeMultiStreamDict(experiment=experiment, pretrained=False)
    model.eval()
    images = [
        torch.randn(1, 2, 128, 128)
        for _ in range(1 if experiment.startswith("single") else 2)
    ]
    with torch.no_grad():
        logits = model(images)
        assert logits.shape == (1, 4, 128, 128)
        if experiment == "gated":
            for band in ("alos2", "s1"):
                assert model(images, band=band).shape == logits.shape
    torch.save(model.state_dict(), tmp_path / "weights.pth")
    loaded = DeepLabLikeMultiStreamDict(experiment=experiment, pretrained=False)
    loaded.load_state_dict(torch.load(tmp_path / "weights.pth", weights_only=True))


def test_scene_weighted_f1_matches_sklearn():
    from sklearn.metrics import f1_score

    true = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    pred = np.array([0, 1, 1, 0, 1, 2, 0, 2, 2])
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(true, pred):
        cm[t, p] += 1
    assert weighted_f1_from_cm(cm) == pytest.approx(
        f1_score(true, pred, average="weighted")
    )


def test_sixteen_configs():
    files = sorted((REPO_ROOT / "configs").glob("*.ini"))
    assert len(files) == 16
    assert len([v for v in files if v.name.startswith("eval_")]) == 8
    assert all("[experiment]" in p.read_text() for p in files)


@pytest.mark.parametrize("ablation", ("none", "no-curriculum", "no-gate-centering"))
def test_ablations_and_output_isolation(ablation, monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("DATASET_ROOT", str(tmp_path / "dataset"))
    argv = ["-c", "SA_gated_s1.ini", "--ablation", ablation]
    args = cli(argparse.ArgumentParser()).parse_args(argv)
    setting = Settings(args, load_config(args))
    assert setting.ablation == ablation
    directory = setting.output("jan")
    if ablation == "none":
        assert directory.name == "seed_0"
    else:
        assert directory.name == "ablation_" + ablation.replace("-", "_")
        assert directory.parent.name == "seed_0"
    model = DeepLabLikeMultiStreamDict(
        experiment="gated", pretrained=False, gated_target_band="s1", ablation=ablation
    )
    assert model.gated_fusion.gate_centering == (ablation != "no-gate-centering")
    # Using a later epoch, no-curriculum still uses two encoders when training.
    from types import SimpleNamespace

    model._trainer = SimpleNamespace(current_epoch=20)
    model.train()
    inputs = [torch.randn(1, 2, 128, 128), torch.randn(1, 2, 128, 128)]
    with torch.no_grad():
        output = model(inputs)
    assert output.shape == (1, 4, 128, 128)
    # hook the fusion to prove it is used at epoch >= 15 only for no-curriculum
    calls = []
    hook = model.gated_fusion.register_forward_hook(lambda *a: calls.append(True))
    with torch.no_grad():
        model(inputs)
    hook.remove()
    assert bool(calls) == (ablation == "no-curriculum")
    # The validation/inference target is still single-band in all variants.
    model.eval()
    calls = []
    hook = model.gated_fusion.register_forward_hook(lambda *a: calls.append(True))
    with torch.no_grad():
        model(inputs, band="s1")
    hook.remove()
    assert not calls


def test_ablation_non_gated_rejected():
    args = cli(argparse.ArgumentParser()).parse_args(
        ["-c", "SA_early.ini", "--ablation", "no-curriculum"]
    )
    with pytest.raises(ValueError, match="only defined for gated"):
        Settings(args, load_config(args))


def test_full_scene_inference(monkeypatch):
    from evaluate import inference

    args = cli(argparse.ArgumentParser()).parse_args(["-c", "eval_SA_single_alos2.ini"])

    class Identity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        experiment = "single_alos2"
        num_classes = 2

        def forward(self, x, band="both"):
            self.calls += 1
            one = x[0][:, 0:1]
            return torch.cat([one, -one], dim=1)

    arr = np.full((2, 130, 150), 2.0, dtype=np.float32)
    arr[0, 0, 0] = np.nan
    model = Identity()
    probabilities = inference(model, [arr], ([1.0, 1.0], [1.0, 1.0]), "both", "cpu")
    assert model.calls == 1
    assert probabilities.shape == (2, 130, 150)
    assert probabilities[0, 0, 0] == pytest.approx(0.5)
    assert probabilities[0, 2, 2] == pytest.approx(
        float(torch.softmax(torch.tensor([1.0, -1.0]), dim=0)[0])
    )


@pytest.mark.parametrize(
    "train_config, eval_config",
    [
        ("SA_single_alos.ini", "eval_SA_single_alos2.ini"),
        ("SA_single_s1.ini", "eval_SA_single_s1.ini"),
        ("SA_early.ini", "eval_SA_early.ini"),
        ("SA_late.ini", "eval_SA_late.ini"),
        ("SA_gated_alos2.ini", "eval_SA_gated_alos2.ini"),
        ("SA_gated_s1.ini", "eval_SA_gated_s1.ini"),
    ],
)
def test_training_evaluation_normalization_identical(train_config, eval_config):
    """Train and eval must use precisely the same fold/channel statistics."""
    from runtime import numbers

    parser = cli(argparse.ArgumentParser())
    train_cfg = load_config(parser.parse_args(["-c", train_config]))
    eval_cfg = load_config(
        cli(argparse.ArgumentParser()).parse_args(["-c", eval_config]), evaluation=True
    )
    for key in ("mean", "std"):
        assert numbers(train_cfg, "datamodule", key) == numbers(
            eval_cfg, "datamodule", key
        )
    training_args = parser.parse_args(["-c", train_config])
    evaluation_args = cli(argparse.ArgumentParser()).parse_args(["-c", eval_config])
    train_settings = Settings(training_args, train_cfg)
    eval_settings = Settings(evaluation_args, eval_cfg)
    for fold in train_settings.folds:
        assert train_settings.stats(fold) == eval_settings.stats(fold)


def test_macro_per_class_f1_all_classes_present_matches_sklearn():
    from evaluate import f1_metrics_from_cm
    from sklearn.metrics import f1_score

    true = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    pred = np.array([0, 1, 0, 1, 2, 2, 0, 2])
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(true, pred):
        cm[t, p] += 1
    metric = f1_metrics_from_cm(cm)
    assert metric["macro_f1"] == pytest.approx(f1_score(true, pred, average="macro"))
    assert metric["weighted_f1"] == pytest.approx(
        f1_score(true, pred, average="weighted")
    )
    assert metric["per_class_f1"] == pytest.approx(f1_score(true, pred, average=None))


def test_absent_class_excluded_from_scene_macro_and_summary_csv(tmp_path):
    import csv

    from evaluate import f1_metrics_from_cm, write_f1_statistics

    # First scene is perfect class 0 only: class 1 has no true support.
    # Second scene has poor class 1 predictions.
    cm_a = np.array([[5, 0], [0, 0]])
    cm_b = np.array([[1, 1], [2, 0]])
    m_a = f1_metrics_from_cm(cm_a)
    m_b = f1_metrics_from_cm(cm_b)
    assert m_a["macro_f1"] == pytest.approx(1.0)
    assert np.isnan(m_a["per_class_f1"][1])
    assert m_b["macro_f1"] == pytest.approx((0.4 + 0.0) / 2)
    write_f1_statistics(
        [("jan", "scene_a", cm_a), ("feb", "scene_b", cm_b)],
        cm_a + cm_b,
        tmp_path,
        ["Class 0", "Class 1"],
    )
    with (tmp_path / "scene_f1.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["f1_class_1"] == ""
    assert rows[0]["support_class_1"] == "0"
    assert float(rows[0]["macro_f1"]) == pytest.approx(1.0)
    assert float(rows[1]["macro_f1"]) == pytest.approx(0.2)
    with (tmp_path / "f1_summary.csv").open() as f:
        table = {row["metric"]: row for row in csv.DictReader(f)}
    assert table["macro_f1"]["n_scenes"] == "2"
    assert float(table["macro_f1"]["scene_mean"]) == pytest.approx(0.6)
    assert table["f1_class_1"]["n_scenes"] == "1"
    assert float(table["f1_class_1"]["scene_mean"]) == pytest.approx(0.0)
    assert table["f1_class_1"]["class_name"] == "Class 1"
    assert table["f1_class_1"]["pooled_support"] == "2"
    assert (tmp_path / "scene_weighted_f1.csv").is_file()
    report = (tmp_path / "metrics_average.txt").read_text()
    assert "macro_f1" in report and "f1_class_1 (Class 1)" in report
    assert "pooled" in report.lower()


def test_eval_rejects_mismatched_training_normalization(tmp_path):
    import json

    from evaluate import verify_training_normalization

    with pytest.raises(FileNotFoundError, match="training manifest"):
        verify_training_normalization(tmp_path, [0.0], [1.0])
    (tmp_path / "run.json").write_text(
        json.dumps({"normalization": {"mean": [1.0, 2.0], "std": [3.0, 4.0]}})
    )
    verify_training_normalization(tmp_path, [1.0, 2.0], [3.0, 4.0])
    with pytest.raises(ValueError, match="differs from training"):
        verify_training_normalization(tmp_path, [1.0, 2.1], [3.0, 4.0])


@pytest.mark.parametrize("stem", ("SA_late.ini", "eval_SA_late.ini"))
def test_class_weighted_flag_and_output_separation(stem, monkeypatch, tmp_path):
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("DATASET_ROOT", str(tmp_path / "dataset"))
    parser = cli(argparse.ArgumentParser())
    standard = parser.parse_args(["--config-file", stem])
    weighted = parser.parse_args(["--config-file", stem, "--class-weighted"])
    assert not standard.class_weighted and weighted.class_weighted
    evaluation = stem.startswith("eval_")
    standard_settings = Settings(standard, load_config(standard, evaluation=evaluation))
    weighted_settings = Settings(weighted, load_config(weighted, evaluation=evaluation))
    assert (
        weighted_settings.output("jan")
        == standard_settings.output("jan") / "class_weighted"
    )


def test_only_cross_entropy_and_weighted_cross_entropy_available():
    from losses import MaskedCrossEntropyLoss
    from models import DeepLabLikeMultiStreamDict

    model = DeepLabLikeMultiStreamDict(experiment="single_alos2", pretrained=False)
    assert isinstance(model.loss_fn, MaskedCrossEntropyLoss)
    assert model.loss_fn.weight is None
    weights = torch.tensor([0.5, 2.5, 1.0, 0.3])
    weighted = DeepLabLikeMultiStreamDict(
        experiment="single_alos2", pretrained=False, class_weights=weights
    )
    assert torch.equal(weighted.loss_fn.weight, weights)


def test_weighted_cross_entropy_ignored_pixels_and_gradients():
    import torch.nn.functional as F
    from losses import MaskedCrossEntropyLoss

    logits = torch.tensor(
        [[[[2.0, -1.0]], [[-2.0, 0.7]], [[1.0, 2.2]]]], requires_grad=True
    )
    labels = torch.tensor([[[0, 4]]])
    weights = torch.tensor([1.0, 2.0, 3.0])
    criterion = MaskedCrossEntropyLoss(weight=weights, ignore_index=4)
    loss = criterion(logits, labels)
    assert float(loss.detach()) == pytest.approx(
        float(
            F.cross_entropy(logits[..., :1], labels[..., :1], weight=weights).detach()
        )
    )
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[..., 1]).item() == 0
    all_invalid = criterion(logits, torch.full_like(labels, 4))
    assert torch.isfinite(all_invalid) and float(all_invalid.detach()) == 0.0


def test_training_only_class_counts_and_inverse_sqrt_weights(tmp_path):
    import rasterio
    from losses import count_training_classes, inverse_sqrt_class_weights
    from rasterio.transform import from_origin

    data = np.array([[0, 0, 0, 1], [0, 0, 1, 2], [0, 4, 4, 3]], dtype=np.uint8)
    label_path = tmp_path / "label.tif"
    with rasterio.open(
        label_path,
        "w",
        driver="GTiff",
        width=4,
        height=3,
        count=1,
        dtype="uint8",
        nodata=4,
        transform=from_origin(0, 3, 1, 1),
    ) as dst:
        dst.write(data, 1)
    counts = count_training_classes([label_path], 4, ignore_index=4)
    assert counts.tolist() == [6, 2, 1, 1]
    weights = inverse_sqrt_class_weights(counts)
    assert weights.shape == (4,)
    assert float(weights[0]) < float(weights[1]) < float(weights[2])
    assert weights.tolist()[-1] == pytest.approx(float(weights[2]))
    assert weights.mean().item() == pytest.approx(1.0)
    unobserved = inverse_sqrt_class_weights([6, 2, 1, 0])
    assert unobserved[3] == 0
    with pytest.raises(ValueError, match="No labeled training pixels"):
        inverse_sqrt_class_weights([0, 0, 0, 0])


def test_eval_requires_matching_weighted_checkpoint():
    from evaluate import checkpoint_class_weights

    regular = {"class_weighted": False, "class_weights": None}
    weighted = {"class_weighted": True, "class_weights": [0.5, 2.0, 0.7, 1.2]}
    assert checkpoint_class_weights(regular, False, 4, "model.pth") is None
    assert torch.equal(
        checkpoint_class_weights(weighted, True, 4, "model.pth"),
        torch.tensor(weighted["class_weights"]),
    )
    with pytest.raises(ValueError, match="differs from evaluation flag"):
        checkpoint_class_weights(weighted, False, 4, "model.pth")
    with pytest.raises(ValueError, match="differs from evaluation flag"):
        checkpoint_class_weights(regular, True, 4, "model.pth")
    with pytest.raises(ValueError, match="missing valid class weights"):
        checkpoint_class_weights({"class_weighted": True}, True, 4, "model.pth")
    with pytest.raises(ValueError, match="invalid class weights"):
        checkpoint_class_weights(
            {"class_weighted": True, "class_weights": [float("nan"), 1, 1, 1]},
            True,
            4,
            "model.pth",
        )


def test_ini_files_do_not_select_objective_or_sampler():
    for config in (REPO_ROOT / "configs").glob("*.ini"):
        contents = config.read_text().lower()
        assert "[loss]" not in contents
        assert "class_weights" not in contents
        for term in ("focal", "tversky", "dice", "class_balanced", "targeted"):
            assert term not in contents


def test_weighted_loss_differs_from_standard_ce_and_state_dict_roundtrip():
    from losses import MaskedCrossEntropyLoss

    logits = torch.tensor([[[[1.0, -2.0]], [[-1.0, 2.0]]]])
    target = torch.tensor([[[0, 1]]])
    regular = MaskedCrossEntropyLoss(ignore_index=4)
    weighted = MaskedCrossEntropyLoss(weight=torch.tensor([1.0, 5.0]), ignore_index=4)
    weighted_value = float(weighted(logits, target))
    assert weighted_value != pytest.approx(float(regular(logits, target)))
    restored = MaskedCrossEntropyLoss(weight=torch.ones(2), ignore_index=4)
    restored.load_state_dict(weighted.state_dict())
    assert torch.allclose(restored.weight, weighted.weight)
    assert float(restored(logits, target)) == pytest.approx(weighted_value)
