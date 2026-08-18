"""Splatfacto with a learned, achromatic appearance code per capture sequence."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Type, Union

import torch
from torch import nn
from torch.nn import Parameter

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig

from space_3dgs.cull_strategy import CullAfterRefineStrategy


@dataclass
class SequenceAppearanceModelConfig(SplatfactoModelConfig):
    """Configuration for sequence-conditioned appearance correction."""

    _target: Type = field(default_factory=lambda: SequenceAppearanceModel)
    num_sequences: int = 0
    """Number of capture sequences (size of the appearance-code table). 0 means
    "take it from the dataparser": SequenceNerfstudio publishes the list of
    sequence names in the dataparser metadata."""
    appearance_embedding_dim: int = 8
    """Size of each learned sequence code."""
    max_log_gain: float = 0.25
    """Limits exposure gain to exp(+/- max_log_gain)."""
    max_bias: float = 0.10
    """Limits additive brightness correction in normalized image units."""
    appearance_reg_mult: float = 1e-4
    """Keeps sequence codes near the shared neutral appearance."""
    cull_stop_iter: int = 0
    """If > 0, keep opacity/scale culling active (without splitting or opacity
    resets) between stop_split_at and this iteration, so Gaussians that fade
    out late in training are still removed."""


class SequenceAppearanceModel(SplatfactoModel):
    """Applies one learned achromatic exposure correction per capture sequence.

    A short embedding is decoded to a global gain and bias. The same correction
    is applied to all three channels, so the model cannot invent colour casts
    (which matters on grayscale or near-grayscale imagery) while still
    absorbing per-sequence exposure differences.

    Evaluation cameras use their attached sequence ID. Viewer/interpolated
    cameras without an ID use the mean embedding across all sequences.
    """

    config: SequenceAppearanceModelConfig

    def populate_modules(self) -> None:
        super().populate_modules()
        if self.config.cull_stop_iter > 0:
            self.strategy = CullAfterRefineStrategy(
                **vars(self.strategy), cull_stop_iter=self.config.cull_stop_iter
            )
            self.strategy_state = self.strategy.initialize_state(scene_scale=1.0)

        self.num_sequences = self.config.num_sequences
        if self.num_sequences <= 0:
            names = (self.kwargs.get("metadata") or {}).get("sequence_names")
            if not names:
                raise ValueError(
                    "num_sequences=0 needs the SequenceNerfstudio dataparser "
                    "(no 'sequence_names' in the dataparser metadata); set "
                    "--pipeline.model.num_sequences explicitly instead"
                )
            self.num_sequences = len(names)

        self.appearance_embedding = nn.Embedding(
            self.num_sequences,
            self.config.appearance_embedding_dim,
        )
        self.appearance_decoder = nn.Linear(self.config.appearance_embedding_dim, 2)

        # Start at a neutral correction while retaining a gradient path from
        # each sequence code through the decoder.
        nn.init.zeros_(self.appearance_embedding.weight)
        nn.init.normal_(self.appearance_decoder.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.appearance_decoder.bias)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        groups = super().get_param_groups()
        groups["appearance"] = list(self.appearance_embedding.parameters()) + list(
            self.appearance_decoder.parameters()
        )
        return groups

    def _appearance_code(self, camera: Cameras) -> torch.Tensor:
        if camera.metadata is not None and "sequence_id" in camera.metadata:
            sequence_id = torch.as_tensor(
                camera.metadata["sequence_id"],
                device=self.appearance_embedding.weight.device,
                dtype=torch.long,
            ).reshape(-1)
            if sequence_id.numel() != 1:
                raise ValueError(
                    f"Expected one sequence ID per rendered camera, got {sequence_id.tolist()}"
                )
            if not 0 <= int(sequence_id.item()) < self.num_sequences:
                raise ValueError(
                    f"Sequence ID {int(sequence_id.item())} is outside "
                    f"[0, {self.num_sequences - 1}]"
                )
            return self.appearance_embedding(sequence_id)

        # A free viewer camera is not tied to one sequence. Average the learned
        # codes to produce a stable, neutral-looking shared appearance.
        return self.appearance_embedding.weight.mean(dim=0, keepdim=True)

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        outputs = super().get_outputs(camera)
        if "rgb" not in outputs:
            return outputs

        raw = self.appearance_decoder(self._appearance_code(camera)).reshape(-1)
        log_gain = self.config.max_log_gain * torch.tanh(raw[0])
        bias = self.config.max_bias * torch.tanh(raw[1])
        outputs["rgb"] = torch.clamp(outputs["rgb"] * torch.exp(log_gain) + bias, 0.0, 1.0)
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        losses = super().get_loss_dict(outputs, batch, metrics_dict=metrics_dict)
        if self.config.appearance_reg_mult > 0:
            losses["appearance_reg"] = (
                self.config.appearance_reg_mult
                * self.appearance_embedding.weight.square().mean()
            )
        return losses
