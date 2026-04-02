"""Imagination output: decode internal visual activations into an image.

For debugging and visualization: "what is the brain imagining right now?"
Triggered when memory_long activates visual region internally (without external input).

Reads visual region activations and reconstructs a low-resolution image.
"""

from __future__ import annotations

import math

import brain_core

from brain.utils.config import REGIONS

_VIS_START = REGIONS["visual"][0]  # 10000

# Sub-region offsets
_LOW_START = 0
_LOW_END = 5000
_MID_START = 5000
_MID_END = 10000
_HIGH_START = 10000
_HIGH_END = 15000
_SPATIAL_START = 15000
_SPATIAL_END = 20000


class ImaginationOutput:
    """Decodes internal visual activations into a visualization.

    Usage:
        imag = ImaginationOutput()
        result = imag.read()
        # result["image"] is a 2D list of floats (grayscale)
        # result["spatial_focus"] is (x, y) center of attention
    """

    def __init__(self, width: int = 32, height: int = 32):
        self.width = width
        self.height = height

    def read(self) -> dict:
        """Read visual region and construct imagination image.

        Returns:
            dict with keys:
              image: 2D list (height × width) of floats [0, 1]
              spatial_focus: (x, y) normalized position or None
              visual_activation: overall visual activity level
              active_neurons: count of active visual neurons
              sub_activations: dict of sub-region activation levels
        """
        visual_activation = brain_core.get_visual_activation()

        # Read all visual activations
        acts = brain_core.read_visual_activations()

        # Reconstruct image from low-level neurons
        image = [[0.0] * self.width for _ in range(self.height)]
        self._fill_from_low_level(acts, image)

        # Read spatial focus
        spatial_focus = self._read_spatial_focus(acts)

        # Sub-region activation levels
        low_peaks = brain_core.get_peak_visual_neurons(10, "low")
        mid_peaks = brain_core.get_peak_visual_neurons(10, "mid")
        high_peaks = brain_core.get_peak_visual_neurons(10, "high")
        spatial_peaks = brain_core.get_peak_visual_neurons(10, "spatial")

        return {
            "image": image,
            "spatial_focus": spatial_focus,
            "visual_activation": visual_activation,
            "active_neurons": len(acts),
            "sub_activations": {
                "low": sum(a for _, a in low_peaks),
                "mid": sum(a for _, a in mid_peaks),
                "high": sum(a for _, a in high_peaks),
                "spatial": sum(a for _, a in spatial_peaks),
            },
        }

    def _fill_from_low_level(
        self,
        acts: list[tuple[int, float]],
        image: list[list[float]],
    ) -> None:
        """Map low-level visual activations back to image pixels."""
        w = self.width
        h = self.height
        grid_size = (w - 2) * (h - 2)  # matches encoding grid

        for gid, act in acts:
            local = gid - _VIS_START
            if _LOW_START <= local < _LOW_END:
                idx = local - _LOW_START
                # Reverse the encoding: idx = r * (w-2) + c
                if grid_size > 0:
                    pixel_idx = idx % grid_size
                    r = pixel_idx // (w - 2) + 1
                    c = pixel_idx % (w - 2) + 1
                    if 0 <= r < h and 0 <= c < w:
                        image[r][c] = max(image[r][c], act)

    def _read_spatial_focus(
        self,
        acts: list[tuple[int, float]],
    ) -> tuple[float, float] | None:
        """Read spatial sub-region to determine focus position."""
        x_sum = 0.0
        x_weight = 0.0
        y_sum = 0.0
        y_weight = 0.0

        for gid, act in acts:
            local = gid - _VIS_START
            if _SPATIAL_START <= local < _SPATIAL_END:
                offset = local - _SPATIAL_START
                if offset < 2500:
                    # X position neurons
                    x_sum += (offset / 2499.0) * act
                    x_weight += act
                else:
                    # Y position neurons
                    y_offset = offset - 2500
                    y_sum += (y_offset / 2499.0) * act
                    y_weight += act

        if x_weight > 0.01 and y_weight > 0.01:
            return (x_sum / x_weight, y_sum / y_weight)
        return None

    def is_imagining(self) -> bool:
        """Check if the brain has significant internal visual activity."""
        return brain_core.get_visual_activation() > 0.1
