"""ResNet18 encoders and DeepLabHead segmentation with configurable SAR fusion.

Single-sensor models accept two channels. Early fusion accepts four channels.
Late fusion concatenates features from two encoders. Gated fusion learns
per-pixel feature weights and supports target-sensor inference.
"""

from typing import List  # noqa: UP035

import pytorch_lightning as pl
import torch
from losses import MaskedCrossEntropyLoss
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


class GatedFusion(nn.Module):
    """Combine L-band and C-band features using a learned spatial gate."""

    def __init__(self, channels: int = 256, gate_centering: bool = True):
        super().__init__()
        self.gate_centering = gate_centering
        self.gate = nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.Sigmoid())

    def forward(self, l: Tensor, c: Tensor) -> Tensor:
        g = self.gate(torch.cat([l, c], dim=1))
        if self.gate_centering:
            # Apply the fixed gate-centering adjustment.
            g = g - 0.5 * (g - 0.5).square() * torch.sign(g - 0.5)
        return g * l + (1 - g) * c


class DeepLabLikeMultiStreamDict(pl.LightningModule):
    def __init__(
        self,
        experiment="single_alos2",
        num_classes=4,
        pretrained=True,
        class_weights=None,
        lr=1e-5,
        reduce_lr_patience=5,
        ignore_index=4,
        gated_switch_epoch=15,
        gated_target_band="alos2",
        ablation="none",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.experiment = experiment
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.lr = lr
        self.reduce_lr_patience = reduce_lr_patience
        if ablation not in ("none", "no-curriculum", "no-gate-centering"):
            raise ValueError(f"Unknown ablation {ablation}")
        if ablation != "none" and experiment != "gated":
            raise ValueError("Ablations require a gated model")
        self.ablation = ablation
        self.gated_switch_epoch = gated_switch_epoch
        self.gated_target_band = gated_target_band
        if gated_target_band not in ("alos2", "s1"):
            raise ValueError("gated_target_band must be alos2 or s1")

        if experiment not in ("single_alos2", "single_s1", "early", "late", "gated"):
            raise ValueError(f"Invalid fusion strategy: {experiment}")
        n_streams = 2 if experiment in ("late", "gated") else 1
        n_channels = 4 if experiment == "early" else 2
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        streams = []
        for _ in range(n_streams):
            backbone = resnet18(weights=weights)
            # Initialize the input projection for two or four SAR channels.
            backbone.conv1 = nn.Conv2d(
                n_channels, 64, 7, stride=2, padding=3, bias=False
            )
            streams.append(IntermediateLayerGetter(backbone, {"layer3": "out"}))
        self.streams = nn.ModuleList(streams)
        self.gated_fusion = (
            GatedFusion(gate_centering=ablation != "no-gate-centering")
            if experiment == "gated"
            else nn.Identity()
        )
        head_channels = 512 if experiment == "late" else 256
        self.classifier = DeepLabHead(head_channels, num_classes)
        # Use fixed-size pooling for variable-resolution inference.
        pool = self.classifier[0].convs[-1][0]
        if isinstance(pool, nn.AdaptiveAvgPool2d):
            self.classifier[0].convs[-1][0] = nn.AvgPool2d(2, 2)

        self.loss_fn = MaskedCrossEntropyLoss(
            weight=class_weights, ignore_index=ignore_index
        )

    def forward(self, x: List[Tensor], band: str = "both") -> Tensor:
        if self.experiment in ("single_alos2", "single_s1"):
            if len(x) != 1:
                raise ValueError("Single sensor model requires one image")
            feature = self.streams[0](x[0])["out"]
        elif self.experiment == "early":
            if len(x) != 2:
                raise ValueError("Early fusion requires L- and C-band images")
            feature = self.streams[0](torch.cat(x, dim=1))["out"]
        else:
            if len(x) != 2:
                raise ValueError("Dual stream requires L- and C-band images")
            if (
                self.experiment == "gated"
                and self.training
                and self.ablation != "no-curriculum"
                and self.current_epoch >= self.gated_switch_epoch
            ):
                band = (
                    self.gated_target_band
                )  # Curriculum: fuse initially, then target only.
            if self.experiment == "gated" and band in ("alos2", "s1"):
                stream_idx = 0 if band == "alos2" else 1
                feature = self.streams[stream_idx](x[stream_idx])["out"]
            else:
                l = self.streams[0](x[0])["out"]
                c = self.streams[1](x[1])["out"]
                feature = (
                    torch.cat([l, c], dim=1)
                    if self.experiment == "late"
                    else self.gated_fusion(l, c)
                )
        return F.interpolate(
            self.classifier(feature),
            size=x[0].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, min_lr=1e-8, patience=self.reduce_lr_patience
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }

    def _step(self, batch, stage):
        if self.experiment.startswith("single_"):
            x = [batch["input"]]
        else:
            x = [batch["input-1"], batch["input-2"]]
        y = batch["label"]
        # Validate gated models using their configured target sensor.
        val_band = (
            self.gated_target_band
            if stage == "val" and self.experiment == "gated"
            else "both"
        )
        logits = self(x, band=val_band)
        loss = self.loss_fn(logits, y)
        predicted = logits.argmax(dim=1)
        mask = y != self.ignore_index
        valid = mask.sum()
        if valid.item() > 0:
            acc = (predicted[mask] == y[mask]).float().mean()
            self.log(
                f"{stage}_accuracy",
                acc,
                on_step=False,
                on_epoch=True,
                batch_size=y.size(0),
            )
        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=(stage == "val"),
            batch_size=y.size(0),
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")
