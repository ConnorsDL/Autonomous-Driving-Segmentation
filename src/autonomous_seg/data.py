"""Dataset utilities for autonomous driving semantic segmentation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class SegmentationDataset(Dataset):
    """Dataset that pairs RGB images with segmentation masks.

    The dataset assumes the following folder structure::

        dataset/
            images/
                xxx.png
                yyy.png
            masks/
                xxx.png
                yyy.png

    Each mask should be stored as either a grayscale image where pixel values
    correspond to class indices or as a paletted image. When ``class_map`` is
    provided the dataset will remap pixel values to contiguous indices.
    """

    def __init__(
        self,
        root: os.PathLike,
        image_transforms: Optional[Callable] = None,
        mask_transforms: Optional[Callable] = None,
        class_map: Optional[Dict[int, int]] = None,
        mask_suffix: str = "",
        image_suffix: str = "",
    ) -> None:
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "masks"
        self.image_paths = sorted(
            [p for p in self.images_dir.glob(f"*{image_suffix}.png")]
            + [p for p in self.images_dir.glob(f"*{image_suffix}.jpg")]
            + [p for p in self.images_dir.glob(f"*{image_suffix}.jpeg")]
        )
        if not self.image_paths:
            raise FileNotFoundError(
                f"No images were found in {self.images_dir}. Ensure the dataset is prepared correctly."
            )

        self.mask_paths = []
        for image_path in self.image_paths:
            candidate_exts = [image_path.suffix, ".png", ".jpg", ".jpeg"]
            mask_path = None
            for ext in candidate_exts:
                potential = self.masks_dir / f"{image_path.stem}{mask_suffix}{ext}"
                if potential.exists():
                    mask_path = potential
                    break
            if mask_path is None:
                raise FileNotFoundError(
                    f"Mask for image {image_path.name} was not found. Expected one of: "
                    + ", ".join(str(self.masks_dir / f"{image_path.stem}{mask_suffix}{ext}") for ext in candidate_exts)
                )
            self.mask_paths.append(mask_path)

        self.image_transforms = image_transforms or transforms.ToTensor()
        self.mask_transforms = mask_transforms
        self.class_map = class_map or {}

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.image_transforms:
            image = self.image_transforms(image)

        mask_array = np.array(mask, dtype=np.int64)
        if mask_array.ndim == 3:  # paletted image expands to shape (H, W, 3)
            mask_array = mask_array[..., 0]

        mask_tensor = torch.from_numpy(mask_array)

        if self.class_map:
            mask_tensor = mask_tensor.clone()
            for original, remapped in self.class_map.items():
                mask_tensor[mask_tensor == original] = remapped

        if self.mask_transforms:
            mask_tensor = self.mask_transforms(mask_tensor)

        return image, mask_tensor


@dataclass
class DataConfig:
    """Configuration options for dataset loading."""

    train_dir: Path
    val_dir: Optional[Path] = None
    image_size: Tuple[int, int] = (512, 512)
    batch_size: int = 4
    num_workers: int = 4
    class_map: Optional[Dict[int, int]] = None
    augmentations: bool = True


def _build_transforms(image_size: Tuple[int, int], augmentations: bool) -> Tuple[Callable, Callable]:
    resize = transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR)
    mask_resize = transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST)

    augment: List[Callable] = []
    if augmentations:
        augment.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ]
        )

    image_transform = transforms.Compose(
        augment
        + [
            resize,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def mask_transform(mask: torch.Tensor) -> torch.Tensor:
        mask_image = Image.fromarray(mask.numpy().astype("uint8"))
        mask_resized = mask_resize(mask_image)
        return torch.from_numpy(np.array(mask_resized, dtype=np.int64))

    return image_transform, mask_transform


def create_dataloaders(config: DataConfig) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create training and validation dataloaders according to ``config``."""

    image_transform, mask_transform = _build_transforms(config.image_size, config.augmentations)

    train_dataset = SegmentationDataset(
        config.train_dir,
        image_transforms=image_transform,
        mask_transforms=mask_transform,
        class_map=config.class_map,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_loader: Optional[DataLoader] = None
    if config.val_dir is not None and config.val_dir.exists():
        val_dataset = SegmentationDataset(
            config.val_dir,
            image_transforms=image_transform,
            mask_transforms=mask_transform,
            class_map=config.class_map,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader
