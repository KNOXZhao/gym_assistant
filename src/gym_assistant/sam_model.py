"""Wrapper utilities for working with SAM 2.1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


class SamModelNotAvailableError(RuntimeError):
    """Raised when SAM 2.1 cannot be imported."""


@dataclass
class Sam2Config:
    checkpoint: Path
    device: str = "cuda"
    config_file: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    points_per_side: int = 32
    pred_iou_thresh: float = 0.86
    stability_score_thresh: float = 0.92
    crop_n_layers: int = 0
    crop_nms_thresh: float = 0.7
    box_nms_thresh: float = 0.7


class Sam2Segmenter:
    """Thin wrapper around the SAM 2 automatic mask generator."""

    def __init__(self, config: Sam2Config) -> None:
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2
        except Exception as exc:  # pragma: no cover - import error handling
            raise SamModelNotAvailableError(
                "Unable to import the SAM 2 package. Install it with 'pip install git+https://github.com/facebookresearch/sam2.git'"
            ) from exc

        model = build_sam2(
            config_file=config.config_file,
            ckpt_path=str(config.checkpoint),
            device=config.device,
        )
        self._generator = SAM2AutomaticMaskGenerator(
            model,
            points_per_side=config.points_per_side,
            pred_iou_thresh=config.pred_iou_thresh,
            stability_score_thresh=config.stability_score_thresh,
            crop_n_layers=config.crop_n_layers,
            crop_nms_thresh=config.crop_nms_thresh,
            box_nms_thresh=config.box_nms_thresh,
        )

    def generate(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Generate segmentation masks for an image in BGR format."""
        image_rgb = image_bgr[..., ::-1]
        return self._generator.generate(image_rgb)
