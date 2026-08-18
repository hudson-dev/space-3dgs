"""Nerfstudio dataparser that attaches a stable capture-sequence ID to each camera.

The sequence is read from the image filename: ``<sequence>_<digits>.<ext>``
(e.g. ``flight_left_1618944193000.png`` -> ``flight_left``). Images whose names
do not follow that pattern all share one sequence, so a plain unstructured
capture simply gets a single appearance code.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.nerfstudio_dataparser import Nerfstudio
from nerfstudio.utils.rich_utils import CONSOLE

DEFAULT_SEQUENCE = "default"


def sequence_name(image_path: Path) -> str:
    """Extract ``<sequence>`` from ``<sequence>_<digits>.<ext>``."""
    stem = image_path.stem
    prefix, separator, suffix = stem.rpartition("_")
    if not separator or not suffix.isdigit() or not prefix:
        return DEFAULT_SEQUENCE
    return prefix


class SequenceNerfstudio(Nerfstudio):
    """Adds ``sequence_id`` camera metadata while preserving the base parser."""

    def _sequence_mapping(self) -> Dict[str, int]:
        data = self.config.data
        transforms_path = data if data.suffix == ".json" else data / "transforms.json"
        meta = json.loads(transforms_path.read_text())
        names = sorted({sequence_name(Path(frame["file_path"])) for frame in meta["frames"]})
        return {name: index for index, name in enumerate(names)}

    def _generate_dataparser_outputs(self, split: str = "train") -> DataparserOutputs:
        outputs = super()._generate_dataparser_outputs(split=split)
        mapping = self._sequence_mapping()
        ids = torch.tensor(
            [mapping[sequence_name(path)] for path in outputs.image_filenames],
            dtype=torch.long,
        ).unsqueeze(-1)

        metadata = dict(outputs.cameras.metadata or {})
        metadata["sequence_id"] = ids
        outputs.cameras.metadata = metadata
        outputs.metadata["sequence_names"] = [
            name for name, _ in sorted(mapping.items(), key=lambda item: item[1])
        ]

        if split == "train":
            CONSOLE.log(f"Capture-sequence appearance IDs: {mapping}")
            if DEFAULT_SEQUENCE in mapping and len(mapping) > 1:
                CONSOLE.log(
                    "[yellow]Some image names lack a '<sequence>_<digits>' pattern; "
                    f"they share the '{DEFAULT_SEQUENCE}' sequence."
                )
        return outputs
