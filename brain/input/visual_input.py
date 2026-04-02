"""Visual input pipeline: convert images/frames into visual region activations.

Processes image data into hierarchical visual features:
  - Low-level (10000–14999): edges, colors, orientation
  - Mid-level (15000–19999): shapes, textures, contours
  - High-level (20000–24999): objects, scenes (via trace matching)
  - Spatial (25000–29999): position, movement

Supports numpy arrays (images) as input.
"""

from __future__ import annotations

import hashlib
import math

import brain_core

from brain.utils.config import REGIONS

_VIS_START = REGIONS["visual"][0]  # 10000

# Sub-region offsets (local to visual)
_LOW_START = 0       # 10000
_LOW_END = 5000      # 14999
_MID_START = 5000    # 15000
_MID_END = 10000     # 19999
_HIGH_START = 10000  # 20000
_HIGH_END = 15000    # 24999
_SPATIAL_START = 15000  # 25000
_SPATIAL_END = 20000    # 29999

# Target image dimensions for processing
_TARGET_WIDTH = 32
_TARGET_HEIGHT = 32


class VisualInput:
    """Encodes images/frames into visual region neuron activations.

    Usage:
        encoder = VisualInput()
        result = encoder.encode(frame)  # frame: (H, W) or (H, W, C) array
    """

    def __init__(self, boost: float = 0.8):
        self.boost = boost

    def encode(self, frame) -> dict:
        """Encode an image frame into brain activations.

        Args:
            frame: numpy-like array of shape (H, W) or (H, W, C).
                   Values expected in [0, 255] or [0.0, 1.0].

        Returns:
            dict with keys: neurons_activated, low_count, mid_count, spatial_count
        """
        pixels = self._normalize(frame)
        signals: list[tuple[int, float]] = []

        # Low-level: edge-like features from pixel intensity gradients
        low_signals = self._extract_low_level(pixels)
        signals.extend(low_signals)

        # Mid-level: shape features from local patterns
        mid_signals = self._extract_mid_level(pixels)
        signals.extend(mid_signals)

        # Spatial: position encoding from center-of-mass of intensity
        spatial_signals = self._extract_spatial(pixels)
        signals.extend(spatial_signals)

        # Activate visual neurons
        neurons = [gid for gid, _ in signals]
        if neurons:
            activated = brain_core.boost_visual(neurons, self.boost)
        else:
            activated = 0

        # Also inject the actual activation values for finer control
        for gid, val in signals:
            brain_core.inject_activations([(gid, val * self.boost)])

        return {
            "neurons_activated": activated,
            "low_count": len(low_signals),
            "mid_count": len(mid_signals),
            "spatial_count": len(spatial_signals),
            "total_signals": len(signals),
        }

    def _normalize(self, frame) -> list[list[float]]:
        """Convert frame to 2D list of float values in [0, 1]."""
        # Handle various input types
        if hasattr(frame, 'shape'):
            # numpy array
            if len(frame.shape) == 3:
                # Color → grayscale via luminance
                if frame.shape[2] == 3:
                    gray = [
                        [float(frame[r][c][0]) * 0.299 +
                         float(frame[r][c][1]) * 0.587 +
                         float(frame[r][c][2]) * 0.114
                         for c in range(frame.shape[1])]
                        for r in range(frame.shape[0])
                    ]
                else:
                    gray = [
                        [float(frame[r][c][0]) for c in range(frame.shape[1])]
                        for r in range(frame.shape[0])
                    ]
            else:
                gray = [
                    [float(frame[r][c]) for c in range(frame.shape[1])]
                    for r in range(frame.shape[0])
                ]
        elif isinstance(frame, list):
            gray = [[float(v) for v in row] for row in frame]
        else:
            return [[0.0] * _TARGET_WIDTH for _ in range(_TARGET_HEIGHT)]

        # Normalize to [0, 1]
        max_val = max(max(row) for row in gray) if gray and gray[0] else 1.0
        if max_val > 1.0:
            gray = [[v / 255.0 for v in row] for row in gray]

        # Resize to target dims using nearest neighbor
        h = len(gray)
        w = len(gray[0]) if gray else 0
        if h == 0 or w == 0:
            return [[0.0] * _TARGET_WIDTH for _ in range(_TARGET_HEIGHT)]

        resized = []
        for r in range(_TARGET_HEIGHT):
            row = []
            src_r = int(r * h / _TARGET_HEIGHT)
            for c in range(_TARGET_WIDTH):
                src_c = int(c * w / _TARGET_WIDTH)
                row.append(gray[min(src_r, h - 1)][min(src_c, w - 1)])
            resized.append(row)
        return resized

    def _extract_low_level(self, pixels: list[list[float]]) -> list[tuple[int, float]]:
        """Extract edge/gradient features → low-level visual neurons."""
        signals = []
        h = len(pixels)
        w = len(pixels[0]) if pixels else 0

        for r in range(1, h - 1):
            for c in range(1, w - 1):
                # Horizontal gradient (Sobel-like)
                gx = abs(pixels[r][c + 1] - pixels[r][c - 1])
                # Vertical gradient
                gy = abs(pixels[r + 1][c] - pixels[r - 1][c])
                edge = min(1.0, (gx + gy) / 2.0)

                if edge > 0.05:
                    # Map (r, c) to low-level neuron range
                    idx = r * (w - 2) + c
                    gid = _VIS_START + _LOW_START + (idx % (_LOW_END - _LOW_START))
                    signals.append((gid, edge))

        return signals

    def _extract_mid_level(self, pixels: list[list[float]]) -> list[tuple[int, float]]:
        """Extract shape/texture features → mid-level visual neurons."""
        signals = []
        h = len(pixels)
        w = len(pixels[0]) if pixels else 0
        block_size = 4

        for br in range(0, h - block_size + 1, block_size):
            for bc in range(0, w - block_size + 1, block_size):
                # Compute block statistics
                block_sum = 0.0
                block_var = 0.0
                vals = []
                for dr in range(block_size):
                    for dc in range(block_size):
                        v = pixels[br + dr][bc + dc]
                        vals.append(v)
                        block_sum += v
                mean = block_sum / len(vals)
                for v in vals:
                    block_var += (v - mean) ** 2
                block_var /= len(vals)

                # High variance → texture; low variance + high mean → uniform shape
                texture = min(1.0, block_var * 10.0)
                uniformity = max(0.0, mean - block_var * 5.0)

                if texture > 0.05:
                    block_idx = (br // block_size) * (w // block_size) + (bc // block_size)
                    gid = _VIS_START + _MID_START + (block_idx % (_MID_END - _MID_START))
                    signals.append((gid, texture))

                if uniformity > 0.3:
                    block_idx = (br // block_size) * (w // block_size) + (bc // block_size)
                    gid = _VIS_START + _MID_START + ((block_idx + 2500) % (_MID_END - _MID_START))
                    signals.append((gid, uniformity))

        return signals

    def _extract_spatial(self, pixels: list[list[float]]) -> list[tuple[int, float]]:
        """Extract spatial position/layout → spatial visual neurons."""
        signals = []
        h = len(pixels)
        w = len(pixels[0]) if pixels else 0

        # Center of mass of intensity
        total = 0.0
        cx = 0.0
        cy = 0.0
        for r in range(h):
            for c in range(w):
                v = pixels[r][c]
                total += v
                cx += c * v
                cy += r * v

        if total > 0.01:
            cx /= total
            cy /= total
            # Normalize to [0, 1]
            norm_x = cx / max(w - 1, 1)
            norm_y = cy / max(h - 1, 1)

            # Encode position as gaussian bump in spatial sub-region
            x_center = int(norm_x * 2499)
            y_center = int(norm_y * 2499)

            for offset in range(-30, 31):
                dist = abs(offset)
                act = math.exp(-0.5 * dist * dist / (15.0 * 15.0))
                if act > 0.01:
                    x_gid = _VIS_START + _SPATIAL_START + max(0, min(x_center + offset, 2499))
                    y_gid = _VIS_START + _SPATIAL_START + max(0, min(y_center + offset + 2500, 4999))
                    signals.append((x_gid, act))
                    if y_gid < _VIS_START + _SPATIAL_END:
                        signals.append((y_gid, act))

        return signals

    def encode_raw(self, neuron_activations: list[tuple[int, float]]) -> int:
        """Directly inject pre-computed visual neuron activations.

        For cases where feature extraction is done externally.
        """
        neurons = [gid for gid, _ in neuron_activations]
        if not neurons:
            return 0
        return brain_core.boost_visual(neurons, self.boost)
