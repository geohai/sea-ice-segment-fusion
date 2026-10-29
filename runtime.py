"""Configuration-only experiment selection, portable paths, and reproducibility."""

import configparser
import os
import platform
import random
import re
from pathlib import Path

import numpy as np

FOLDS = ("jan", "feb", "mar", "nov", "dec")
EXPERIMENT_INPUT_BANDS = {
    "single_alos2": ("alos2",),
    "single_s1": ("s1",),
    "early": ("alos2", "s1"),
    "late": ("alos2", "s1"),
    "late_alos2_alos2": ("alos2", "alos2"),
    "late_s1_s1": ("s1", "s1"),
    "gated": ("alos2", "s1"),
}
EXPERIMENTS = tuple(EXPERIMENT_INPUT_BANDS)
REPO_ROOT = Path(__file__).resolve().parent


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch/Lightning, and request deterministic kernels."""
    import pytorch_lightning as pl
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    pl.seed_everything(seed, workers=True)


def cli(parser):
    parser.add_argument(
        "-c",
        "--config-file",
        required=True,
        help="Path to an experiment INI, or filename in configs/",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="One-epoch Mac/CPU-friendly smoke test; omitted = full run",
    )
    parser.add_argument(
        "--ablation",
        choices=("none", "no-curriculum", "no-gate-centering"),
        default="none",
        help="Gated experiments only; match training/evaluation to one ablation",
    )
    parser.add_argument(
        "--class-weighted",
        action="store_true",
        help="Use inverse-square-root class-weighted cross-entropy instead of standard CE",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    return parser


def load_config(args, evaluation=False):
    candidate = Path(os.path.expandvars(os.path.expanduser(args.config_file)))
    if not candidate.is_file():
        candidate = REPO_ROOT / "configs" / args.config_file
    if not candidate.is_file():
        raise FileNotFoundError(f"Configuration not found: {args.config_file}")
    cfg = configparser.ConfigParser(interpolation=None)
    if not cfg.read(candidate):
        raise OSError(f"Unable to read {candidate}")
    is_eval = cfg.getboolean("eval", "enabled", fallback=False)
    if is_eval != evaluation:
        kind = "evaluation" if evaluation else "training"
        raise ValueError(f"Expected a {kind} INI: {candidate}")
    return cfg


def entries(cfg, section, key):
    return [
        line.strip()
        for line in cfg.get(section, key).splitlines()
        if line.strip() and not line.lstrip().startswith(("#", ";"))
    ]


def numbers(cfg, section, key):
    return [float(x.strip()) for x in cfg.get(section, key).split(",")]


def route_late_inputs(df, input_bands):
    """Route the two independent late-fusion encoders to their selected SAR inputs.

    Both encoder columns can point to the same raster, without sharing encoder
    weights or changing the number of channels entering the decoder.
    """
    source_columns = {"alos2": "input-1", "s1": "input-2"}
    missing = {"dset", "input-1", "input-2", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"Dual-stream training CSV is missing {sorted(missing)}")
    if len(input_bands) != 2 or any(b not in source_columns for b in input_bands):
        raise ValueError(f"Expected two SAR inputs, got {input_bands!r}")
    routed = df[["dset", "label"]].copy()
    for index, band in enumerate(input_bands, start=1):
        routed[f"input-{index}"] = df[source_columns[band]]
    return routed[["dset", "input-1", "input-2", "label"]]


def configured_root(name):
    """An exported relative root is relative to the repository, not the shell CWD."""
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    expanded = Path(os.path.expandvars(os.path.expanduser(value.strip())))
    return (
        expanded.resolve()
        if expanded.is_absolute()
        else (REPO_ROOT / expanded).resolve()
    )


def after_results(raw):
    """Take everything after the 'results/' component, even from Windows paths."""
    path = str(raw).strip().replace("\\", "/")
    match = re.search(r"(?:^|/)results/(.*)$", path, re.IGNORECASE)
    if not match or not match.group(1):
        raise ValueError(f"Expected path containing results/: {raw!r}")
    return Path(match.group(1))


def relative_dataset_path(raw, configured_root):
    """Relocate absolute paths from the configured dataset directory."""
    path = str(raw).strip().replace("\\", "/")
    prefix = str(configured_root).rstrip("/\\").replace("\\", "/")
    if path.lower().startswith(prefix.lower() + "/"):
        path = path[len(prefix) + 1 :]
    elif re.match(r"^[A-Za-z]:/", path) or Path(path).is_absolute():
        raise ValueError(
            f"Cannot rebase dataset raster path {raw!r}; expected {prefix}"
        )
    return Path(path)


class Settings:
    def __init__(self, args, cfg):
        self.platform = platform.system().lower()
        self.mode = "dev" if args.dev else "full"
        self.seed = args.seed
        self.class_weighted = args.class_weighted
        self.experiment = cfg.get("experiment", "name")
        self.ablation = args.ablation
        if self.ablation != "none" and self.experiment != "gated":
            raise ValueError("Ablations are only defined for gated experiments")
        if self.experiment not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment {self.experiment}")
        self.model_experiment = (
            "late"
            if self.experiment in ("late_alos2_alos2", "late_s1_s1")
            else self.experiment
        )
        self.input_bands = EXPERIMENT_INPUT_BANDS[self.experiment]
        self.folds = [FOLDS[0]] if args.dev else list(FOLDS)
        self.device = args.device
        self.cfg = cfg
        self.args = args
        self.output_root_env = configured_root("OUTPUT_ROOT")
        self.dataset_root_env = configured_root("DATASET_ROOT")
        self.default_dataset_root = Path(cfg.get("paths", "default_dataset_root"))
        self.default_output_root = Path(cfg.get("paths", "default_output_root"))
        self.dataset_root = self.dataset_root_env or self.default_dataset_root
        self.data_root = self.output_root_env or self.default_output_root
        self.fold_names = entries(cfg, "folds", "names")
        self.csv_paths = entries(cfg, "io", "csv_files")
        self.output_dirs = entries(cfg, "io", "run_dirs")
        if not (len(self.fold_names) == len(self.csv_paths) == len(self.output_dirs)):
            raise ValueError("folds.names, io.csv_files and io.run_dirs must align")
        if set(self.fold_names) != set(FOLDS):
            raise ValueError(f"Expected the five folds {FOLDS}")
        self.training = not cfg.getboolean("eval", "enabled", fallback=False)
        self.gated_target_band = cfg.get("model", "gated_target_band", fallback="alos2")
        self.eval_band = cfg.get("eval", "band", fallback="both")
        if self.experiment == "gated":
            if self.gated_target_band not in ("alos2", "s1"):
                raise ValueError("[model] gated_target_band must be alos2 or s1")
            if self.eval_band not in ("both", "alos2", "s1"):
                raise ValueError("[eval] band must be both, alos2 or s1")
        elif self.eval_band != "both":
            raise ValueError("Only gated evaluation supports [eval] band != both")

    def fold_index(self, fold):
        return self.fold_names.index(fold)

    def csv(self, fold):
        suffix = after_results(self.csv_paths[self.fold_index(fold)])
        # DATASET_ROOT contains CSV files directly, while the fallback has a
        # results/ subdirectory for CSV files.
        return (
            self.dataset_root_env / suffix
            if self.dataset_root_env is not None
            else self.default_dataset_root / "results" / suffix
        )

    def output(self, fold):
        suffix = after_results(self.output_dirs[self.fold_index(fold)])
        base = self.data_root / suffix / self.mode / f"seed_{self.seed}"
        if self.ablation != "none":
            base = base / f"ablation_{self.ablation.replace('-', '_')}"
        return base / "class_weighted" if self.class_weighted else base

    def stats(self, fold):
        idx = self.fold_index(fold)
        means, stds = (
            numbers(self.cfg, "datamodule", "mean"),
            numbers(self.cfg, "datamodule", "std"),
        )
        channels = 2 if self.experiment.startswith("single_") else 4
        if len(means) != channels * len(self.fold_names) or len(stds) != len(means):
            raise ValueError(
                f"Expected {channels * len(self.fold_names)} normalization values"
            )
        lo = channels * idx
        return means[lo : lo + channels], stds[lo : lo + channels]

    def scenes(self, fold):
        idx = 2 * self.fold_index(fold)

        def get(key):
            values = entries(self.cfg, "test", key)
            if len(values) != len(self.fold_names) * 2:
                raise ValueError(f"[test] {key} must list two scenes per fold")
            return [
                self.dataset_root / relative_dataset_path(v, self.default_dataset_root)
                for v in values[idx : idx + 2]
            ]

        return get("test_alos2"), get("test_s1"), get("test_labels")

    def assert_environment(self):
        # A Windows drive path is not a valid local dataset root on macOS/Linux.
        if self.platform != "windows" and (
            self.output_root_env is None or self.dataset_root_env is None
        ):
            raise RuntimeError(
                "Configured paths use Windows drives. On macOS/Linux, set "
                "OUTPUT_ROOT and DATASET_ROOT before running."
            )


def device_name(requested):
    import torch

    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def rebase_csv_paths(df, dataset_root, configured_root="D:/jema2085/data/Belgica_Bank"):
    """Map CSV input/label file paths onto the dataset root on both platforms."""
    df = df.copy()
    for col in df.columns:
        if not (col.startswith("input") or col.startswith("label")):
            continue

        def rebase(raw):
            path = str(raw).strip().replace("\\", "/")
            root = str(configured_root).rstrip("/").replace("\\", "/")
            if path.lower().startswith(root.lower() + "/"):
                path = path[len(root) + 1 :]
            elif re.match(r"^[A-Za-z]:/", path):
                raise ValueError(
                    f"Unknown Windows drive path in CSV: {path}; expected {root}"
                )
            elif Path(path).is_absolute():
                return path  # Already an absolute raster path; leave untouched.
            return str(dataset_root / path)

        df[col] = df[col].map(rebase)
    return df
