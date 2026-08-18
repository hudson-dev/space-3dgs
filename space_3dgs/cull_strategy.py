"""gsplat densification strategy that keeps culling after splitting stops."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Union

import torch

from gsplat.strategy import DefaultStrategy


@dataclass
class CullAfterRefineStrategy(DefaultStrategy):
    """DefaultStrategy, plus opacity/scale culling after ``refine_stop_iter``.

    The stock strategy stops everything at ``refine_stop_iter``: splitting,
    duplication, opacity resets — and culling. Over a long schedule that leaves
    the final third of training with no way to remove Gaussians, so
    low-opacity haze that is still present when splitting stops survives to
    the export. This subclass keeps ``_prune_gs`` (opacity < prune_opa,
    oversized scales) running every ``refine_every`` steps until
    ``cull_stop_iter``.
    """

    cull_stop_iter: int = 0
    """Keep pruning until this iteration; 0 reproduces DefaultStrategy."""

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
    ) -> None:
        if step < self.refine_stop_iter:
            super().step_post_backward(params, optimizers, state, step, info, packed=packed)
            return
        if step < self.cull_stop_iter and step % self.refine_every == 0:
            n_prune = self._prune_gs(params, optimizers, state, step)
            if self.verbose and n_prune > 0:
                print(
                    f"Step {step}: {n_prune} GSs pruned post-refinement. "
                    f"Now having {len(params['means'])} GSs."
                )
