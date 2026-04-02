"""Download and prepare datasets from HuggingFace for brain learning.

Supports:
  - ag_news:          short news text, 4 classes (World/Sports/Business/Sci-Tech)
  - cifar10:          32x32 color images, 10 classes
  - speech_commands:  1-second spoken word audio clips, 35 classes
  - imdb:             movie reviews, 2 classes (pos/neg)
  - common_voice:     multi-language speech audio
"""

from __future__ import annotations

import math
from typing import Any, Generator

import numpy as np


def _ensure_datasets():
    """Import and return the datasets library, with a helpful error if missing."""
    try:
        import datasets
        return datasets
    except ImportError:
        raise ImportError(
            "HuggingFace `datasets` library required.\n"
            "Install: pip install datasets Pillow soundfile"
        )


# ── Dataset loaders ──────────────────────────────────────────────────────────

def load_text_dataset(
    name: str = "ag_news",
    split: str = "train",
    max_samples: int = 100,
) -> list[dict[str, Any]]:
    """Load a text dataset. Returns list of {text, label, label_name}."""
    ds_lib = _ensure_datasets()

    if name == "ag_news":
        ds = ds_lib.load_dataset("ag_news", split=f"{split}[:{max_samples}]")
        label_names = ["World", "Sports", "Business", "Sci/Tech"]
        return [
            {
                "text": row["text"],
                "label": row["label"],
                "label_name": label_names[row["label"]],
            }
            for row in ds
        ]

    elif name == "imdb":
        ds = ds_lib.load_dataset("imdb", split=f"{split}[:{max_samples}]")
        label_names = ["negative", "positive"]
        return [
            {
                "text": row["text"][:500],  # truncate long reviews
                "label": row["label"],
                "label_name": label_names[row["label"]],
            }
            for row in ds
        ]

    else:
        raise ValueError(f"Unknown text dataset: {name}")


def load_image_dataset(
    name: str = "cifar10",
    split: str = "train",
    max_samples: int = 100,
) -> list[dict[str, Any]]:
    """Load an image dataset. Returns list of {image: np.ndarray, label, label_name}."""
    ds_lib = _ensure_datasets()

    if name == "cifar10":
        ds = ds_lib.load_dataset("cifar10", split=f"{split}[:{max_samples}]")
        label_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ]
        results = []
        for row in ds:
            img = row["img"]
            # PIL Image → numpy array (H, W, C), uint8
            arr = np.array(img, dtype=np.uint8)
            results.append({
                "image": arr,
                "label": row["label"],
                "label_name": label_names[row["label"]],
            })
        return results

    else:
        raise ValueError(f"Unknown image dataset: {name}")


def load_audio_dataset(
    name: str = "speech_commands",
    split: str = "train",
    max_samples: int = 100,
) -> list[dict[str, Any]]:
    """Load an audio dataset. Returns list of {audio: np.ndarray, sample_rate, label, label_name}."""
    ds_lib = _ensure_datasets()

    if name == "speech_commands":
        ds = ds_lib.load_dataset(
            "google/speech_commands", "v0.02",
            split=f"{split}[:{max_samples}]",
        )
        results = []
        for row in ds:
            audio = row["audio"]
            samples = np.array(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]
            results.append({
                "audio": samples,
                "sample_rate": sr,
                "label": row.get("label", -1),
                "label_name": row.get("label", "unknown") if isinstance(row.get("label"), str) else str(row.get("label", "?")),
            })
        return results

    else:
        raise ValueError(f"Unknown audio dataset: {name}")


def load_multimodal_batch(
    text_name: str = "ag_news",
    image_name: str = "cifar10",
    audio_name: str = "speech_commands",
    max_samples: int = 50,
) -> list[dict[str, Any]]:
    """Load aligned samples from text + image + audio datasets.

    Each sample gets one of each modality (zipped by index).
    """
    texts = load_text_dataset(text_name, max_samples=max_samples)
    images = load_image_dataset(image_name, max_samples=max_samples)
    audios = load_audio_dataset(audio_name, max_samples=max_samples)

    n = min(len(texts), len(images), len(audios), max_samples)
    results = []
    for i in range(n):
        results.append({
            "text": texts[i]["text"],
            "text_label": texts[i]["label_name"],
            "image": images[i]["image"],
            "image_label": images[i]["label_name"],
            "audio": audios[i]["audio"],
            "audio_sample_rate": audios[i]["sample_rate"],
            "audio_label": audios[i].get("label_name", "?"),
        })
    return results


# ── Image-to-video helper ────────────────────────────────────────────────────

def images_to_video_frames(
    images: list[np.ndarray],
    frames_per_image: int = 3,
) -> Generator[tuple[np.ndarray, str], None, None]:
    """Convert a list of images into a pseudo-video stream.

    Yields (frame, label) tuples. Between images, yields interpolated
    crossfade frames for smoother transitions.
    """
    for i, img in enumerate(images):
        for _ in range(frames_per_image):
            yield img, f"frame_{i}"
        # Crossfade to next image
        if i + 1 < len(images):
            next_img = images[i + 1]
            # Simple blend
            blend = (img.astype(np.float32) + next_img.astype(np.float32)) / 2.0
            yield blend.astype(np.uint8), f"transition_{i}_{i+1}"
