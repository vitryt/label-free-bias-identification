import os
import csv
from PIL import Image, ImageFile
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset
from torchvision import transforms

class WaterBirdsDataset(Dataset):
    """
    Loading Waterbirds dataset.

    File structure:
        <root_dir>/data/Waterbirds/waterbird_complete95_forest2water2/
            ├── Bird image subfolders
            └── metadata.csv

    Dataset containts the following columns: img_id, img_filename, y, split, place
    With:
        y (target label)           --> 0 = landbird, 1 = waterbird       
        place (spurious attribute) --> 0 = land background, 1 = water    
        split                      --> 0 = train, 1 = val, 2 = test
    """

    class_names = ["landbird", "waterbird"]
    spurious_names = ["land", "water"]

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        return_group: bool = False,
        image_size: int = 224,
    ):
        """
        root_dir (str): path to Waterbirds dataset files
        split {"train","val","test"}: specifying which split to load
        transform: torchvision transforms or None
        return_group (bool): if set true, the getitem function returns single integer "g" encoding pair (y, a)
                            --> utilised for per-group metrics and to make it easier to index counts and weights
        image_size (int): by default 224
        """
        self.root_dir = Path(root_dir).expanduser().resolve() # Resolve and normalise root path

        folder_loc = self.root_dir / "waterbird_complete95_forest2water2"

        if folder_loc.exists():
            self.dataset_dir = folder_loc
        else:
            raise FileNotFoundError(
                f"Could not find the Waterbirds dataset files.\n Tried {folder_loc}"
            )
        
        folder_loc = self.root_dir / "segmentations"

        if folder_loc.exists():
            self.segmentation_dir = folder_loc
        else:
            raise FileNotFoundError(
                "Could not find the CUB segmentation files.\n"
            )

        self.metadata_path = self.dataset_dir / "metadata.csv"
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"metadata.csv not found at: {self.metadata_path}")

        split_key = str(split).lower() # Specify dataset split based on input
        if split_key not in {"train", "val", "test"}:
            raise ValueError("split must be one of: 'train', 'val', 'test'")
        self.split_id = 0 if split_key == "train" else (1 if split_key in {"val"} else 2)

        if transform is None: # By default the transform uses ImageNet preprocessing for the ResNet-style model
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
                transforms.Resize(resize_side),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ])
        self.return_group = return_group

        self.image_paths = []   # Path to Waterbird images
        self.mask_paths = []    # Path to the CUB masks
        self.y = []             # Target labels
        self.a = []             # Spurious attribute i.e. land or water
        self.g = []             # Encoding class and background to single integer fro group information

        ImageFile.LOAD_TRUNCATED_IMAGES = True # Load truncatd images without error

        with open(self.metadata_path, "r", newline = "") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["split"]) != self.split_id:
                    continue

                img_path = self.dataset_dir / row["img_filename"] # Get absolute file path
                if not img_path.exists():
                    alt = self.dataset_dir / "images" / row["img_filename"]
                    if alt.exists():
                        img_path = alt
                    else:
                        raise FileNotFoundError(
                            f"Image not found:\n  {img_path}\n  (alternatively, tried {alt})"
                        )
                
                seg_path = self.segmentation_dir / row["img_filename"]
                seg_path = str(seg_path).replace(".jpg", ".png")
                if not os.path.exists(seg_path):
                    raise FileNotFoundError(
                        f"Segmentation not found:\n  {seg_path}\n"
                    )

                y = int(row["y"])       # Landbird (0) and Waterbird (1)
                a = int(row["place"])   # Land (0) and Water (1)
                g = 2 * y + a           # Group information

                self.image_paths.append(img_path)
                self.mask_paths.append(seg_path)
                self.y.append(y)
                self.a.append(a)
                self.g.append(g)

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"Error: no samples found for the split: '{split_key}' (id={self.split_id}). "
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """
        (image, y, a) or (image, y, a, g) --> if group selected
            image: Tensor [3, H, W] after transforms
            y (int): Landbird (0), Waterbird (1)
            a (int): Land background (0), Water background (1)
            g (int): 2*y + a in {0,1,2,3}
        """
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        img = Image.open(img_path).convert("RGB") # Loading images as RGB and apply transforms
        img = self.transform(img)

        mask = Image.open(mask_path).convert("L") # Loading images as RGB and apply transforms
        mask = self.mask_transform(mask)

        y = self.y[idx]
        a = self.a[idx]

        if self.return_group:
            g = self.g[idx]
            return img, y, a, g, mask
        return img, y, a, mask


# Dummy use:
# if __name__ == "__main__":
#     print("Let's go!")
#     ds_train = WaterBirdsDataset(root_dir=".", split="train", return_group=True)
#     print("Train samples:", len(ds_train))
#     x, y, a, g = ds_train[0]
#     print("One sample:", x.shape, y, a, g)

###########################
# Expected dummy results:
# 4795 training samples
# Size = RGB (3), 224x224
# Labels: 1 (Waterbird, y), 1 (Background, a), 3 (Group encoding, g)
###########################




import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

# Torch model: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html

class WaterbirdsResNet50(nn.Module):
    def __init__(self, num_classes=2, pretrained=True): # Also loading weights trained on ImageNet-1K
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def get_grad_cam_target_layer(self):
        """Last conv block — standard GradCAM target for ResNets."""
        return self.backbone.layer4[-1]

    def forward(self, x):
        return self.backbone(x)


from torchvision.models import resnet18, ResNet18_Weights

# Torch model: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
# Original ResNet paper: https://arxiv.org/abs/1512.03385

class WaterbirdsResNet18(nn.Module):
    def __init__(self, num_classes=2, pretrained=True): # Pre-trained weights on ImageNet-1K
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def get_grad_cam_target_layer(self):
        """Last conv block — standard GradCAM target for ResNets."""
        return self.backbone.layer4[-1]

    def forward(self, x):
        return self.backbone(x)