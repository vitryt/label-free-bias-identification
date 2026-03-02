import os
import re
import torch
from PIL import Image, ImageFile
from pathlib import Path
from typing import Optional, List, Tuple
from torch.utils.data import Dataset
from torchvision import transforms

# For the ResNet models below
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet18, ResNet18_Weights

class UrbanCarsDataset(Dataset):
    """
    UrbanCars dataset loader

    File structure: /data/urbancars/bg-0.95_co_occur_obj-0.95/train/ --> 8 subcategories
                    /data/urbancars/bg-0.5_co_occur_obj-0.5/test/ --> 8 subcategories
                    /data/urbancars/bg-0.5_co_occur_obj-0.5/val/ --> 8 subcategories

    Labels:
        y as target label --> 0 = urban, 1 = country
        a_bg as spurious background --> 0 = urban, 1 = country
        a_co as spurious co-occuring object --> 0 = urban, 1 = country
        g for group encoding

    Setting the "primary" spurious attribute "a" to background (i.e., a_bg)
    This is stored into batch[2] of our 2x2 evaluation matrix in model_utils.py
    Storing co-occurrence object labels as "self.a_co"
    """

    label_map = {"urban": 0, "country": 1}
    class_names = ["urban", "country"]
    spurious_names = ["urban_bg", "country_bg"]
    co_occur_names = ["urban_object", "country_object"]

    subdir_pattern = re.compile(
        r"obj-(?P<obj>urban|country)_bg-(?P<bg>urban|country)_co_occur_obj-(?P<co>urban|country)"
    )

    image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        return_group: bool = False,
        image_size: int = 224,
    ):
        self.root_dir = Path(root_dir).expanduser().resolve()

        split_key = str(split).lower()
        if split_key not in {"train", "val", "test"}:
            raise ValueError("split must be one of: 'train', 'val', 'test'")

        self.split_dir = self._find_split_dir(split_key)

        if transform is None:
            resize_side = int(image_size * 256 / 224)  # 256 when image_size=224
            transform = transforms.Compose([
                transforms.Resize(resize_side),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

        self.transform = transform
        self.mask_transform = transforms.Compose([
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
        self.return_group = return_group

        # Data lists
        self.image_paths: List[Path] = []
        self.mask_paths: List[Optional[Path]] = []
        self.y: List[int] = []                     # Target label = urban is 0 / country is 1
        self.a_bg: List[int] = []                  # Spurious attribute = background type
        self.a_co: List[int] = []                  # Spurious attribute = co-occurring object type
        self.a: List[int] = []                     # "Primary" spurious attribute (setting to background by default)
        self.g: List[int] = []

        ImageFile.LOAD_TRUNCATED_IMAGES = True # Specified so we can handle any corrupt images

        self._load_samples()

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No samples found for split '{split_key}' in {self.split_dir}"
            )

    def _find_split_dir(self, split_key: str) -> Path:
        """
        Finding the split directory within the root folder, same structure as Whac-A-Mole
        """
        
        direct = self.root_dir / split_key
        if direct.is_dir() and self._has_urbancars_subdirs(direct):
            return direct

        if self.root_dir.is_dir():
            for ratio_dir in sorted(self.root_dir.iterdir()):
                if not ratio_dir.is_dir():
                    continue
                candidate = ratio_dir / split_key
                if candidate.is_dir() and self._has_urbancars_subdirs(candidate):
                    return candidate

        if self.root_dir.is_dir():
            for subdir in sorted(self.root_dir.iterdir()):
                if not subdir.is_dir():
                    continue
                for ratio_dir in sorted(subdir.iterdir()):
                    if not ratio_dir.is_dir():
                        continue
                    candidate = ratio_dir / split_key
                    if candidate.is_dir() and self._has_urbancars_subdirs(candidate):
                        return candidate

        raise FileNotFoundError(
            f"Could not find UrbanCars '{split_key}' split directory.\n"
            f"Searched under: {self.root_dir}\n"
            f"Expected structure:\n"
            f"  <root>/bg-X_co_occur_obj-X/{split_key}/obj-*_bg-*_co_occur_obj-*/"
        )

    def _has_urbancars_subdirs(self, directory: Path) -> bool:
        """Check if a directory contains UrbanCars-style subdirectories."""
        for child in directory.iterdir():
            if child.is_dir() and self.subdir_pattern.match(child.name):
                return True
        return False

    def _find_mask(self, img_path: Path) -> Optional[Path]:
        """Find mask file that belongs to image from mask"""
        stem = img_path.stem
        parent = img_path.parent

        for suffix in ["_mask", "_seg", "_segmentation"]:
            for ext in [".png", ".jpg", ".bmp"]:
                mask_path = parent / f"{stem}{suffix}{ext}"
                if mask_path.exists():
                    return mask_path

        mask_dir = parent / "mask"
        if mask_dir.is_dir():
            for ext in [".png", ".jpg", ".bmp"]:
                mask_path = mask_dir / f"{stem}{ext}"
                if mask_path.exists():
                    return mask_path

        return None

    def _is_mask_file(self, path: Path) -> bool:
        """Checking if a file is a mask instead of the actual image"""
        name_lower = path.stem.lower()
        return any(tag in name_lower for tag in ["_mask", "_seg", "_segmentation"])

    def _load_samples(self):
        """Enumerating all images from the 8 subdirectories and extracting the needed labels"""
        for subdir in sorted(self.split_dir.iterdir()):
            if not subdir.is_dir():
                continue

            match = self.subdir_pattern.match(subdir.name)
            if not match:
                continue

            y = self.label_map[match.group("obj")]
            a_bg = self.label_map[match.group("bg")]
            a_co = self.label_map[match.group("co")]
            g = 4 * y + 2 * a_bg + a_co

            for img_path in sorted(subdir.iterdir()):
                if not img_path.is_file():
                    continue
                if img_path.suffix.lower() not in self.image_ext:
                    continue
                if self._is_mask_file(img_path):
                    continue

                mask_path = self._find_mask(img_path)

                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)
                self.y.append(y)
                self.a_bg.append(a_bg)
                self.a_co.append(a_co)
                self.a.append(a_bg)     # Note: primary spurious attribute = background (see above)
                self.g.append(g)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """
        Returns:
            Without the return_group (image, y, a, mask)
            With the return_group (image, y, a, g, mask)
        """
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        mask_path = self.mask_paths[idx]
        if mask_path is not None and mask_path.exists():
            mask = Image.open(mask_path).convert("L")
            mask = self.mask_transform(mask)
        else:
            # If a blank mask (all foreground) when no mask is available
            mask = torch.ones(1, img.shape[1], img.shape[2])

        y = self.y[idx]
        a = self.a[idx]

        if self.return_group:
            g = self.g[idx]
            return img, y, a, g, mask
        return img, y, a, mask

# Dummy Usage:

if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "data/UrbanCars"
    print(f"TESTING! UrbanCars dataloader  (root={root})")
    print("=" * 55)

    for split in ["train", "val", "test"]:
        try:
            ds = UrbanCarsDataset(root_dir=root, split=split, return_group=True)
            print(f"\n  {split:>5s} samples : {len(ds)}")
            # Group counts
            from collections import Counter
            g_counts = Counter(ds.g)
            for g_val in sorted(g_counts):
                print(f"    group {g_val}: {g_counts[g_val]}")
            # Sample
            img, y, a, g, mask = ds[0]
            print(f"    sample shape: img={img.shape}, mask={mask.shape}, y={y}, a={a}, g={g}")
        except FileNotFoundError as e:
            print(f"\n  {split:>5s}: not found ({e})")
    print("Done.")

# Including the model for ResNet here as we have done in other dataloaders
# Torch model: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html

class UrbanCarsResNet50(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# Torch model: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html

class UrbanCarsResNet18(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)