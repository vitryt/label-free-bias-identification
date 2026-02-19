from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

# Torch model: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html

class CelebAResNet50(nn.Module):
    def __init__(self, pretrained = True, multi_label = False, num_attributes = 40): # Also loading weights trained on ImageNet-1K
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet50(weights=weights)
        in_features = self.backbone.fc.in_features

        out_dim = num_attributes if multi_label else 2 # If multi_label passed False, sing binary classification of one target attribute 
        self.backbone.fc = nn.Linear(in_features, out_dim)

        self.multi_label = multi_label
        self.out_dim = out_dim

    def forward(self, x):
        return self.backbone(x)
    




import torch
from PIL import Image, ImageFile
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from torch.utils.data import Dataset
from torchvision import transforms

class CelebADataset(Dataset):
    """
    Loading CelebA dataset.

    File structure:
        <root_dir>/data/CelebA/.../
            ├── img_align_celeba
            └── list_attr_celeba.txt
            └── list_eval_partition.txt

    If multi_label = False -> returns y as int in range {0,1} for target_attribute
    if multi_label = True -> returns y as Tensor[40] in range {0,1} for all the target attributes
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        target_attribute: str = "Blond_Hair",           # Specify the target attribute to examine (case: multi_label = False)
        spurious_attribute: str = "Male",               # Specify the spurious attribute to examine
        transform: Optional[transforms.Compose] = None,
        return_group: bool = False,
        image_size: int = 224,
        multi_label: bool = False, # Change to toggle to using all attributes
    ):
        """
        root_dir (str): path to CelebA dataset files
        split {"train","val","test"}: specifying which split to load
        transform: torchvision transforms or None
        return_group (bool): if set true, the getitem function returns single integer "g" encoding pair (y, a)
                            --> utilised for per-group metrics and to make it easier to index counts and weights
        image_size (int): by default 224
        multi_label (bool): specifies whether to examine one target attribute or all 40
        """
        self.root_dir = Path(root_dir).expanduser().resolve() # Resolve and normalise root path

        folder_loc = self.root_dir
        fallback = self.root_dir 
        fallback_again = self.root_dir # If CelebA folder provided directly

        def is_this_celeba(d: Path) -> bool:
            return ((d/"img_align_celeba").is_dir() and (d/"list_attr_celeba.txt").is_file() and (d/"list_eval_partition.txt").is_file())

        if is_this_celeba(folder_loc):
            self.dataset_dir = folder_loc
        elif is_this_celeba(fallback):
            self.dataset_dir = fallback
        elif is_this_celeba(fallback_again):
            self.dataset_dir = fallback_again
        else:
            raise FileNotFoundError("Could not find the CelebA dataset files")
            
        self.attribute_path = self.dataset_dir/"list_attr_celeba.txt"
        self.partition_path = self.dataset_dir/"list_eval_partition.txt"
        self.images_dir = self.dataset_dir/"img_align_celeba"

        if not self.attribute_path.is_file():
            raise FileNotFoundError("Couldn't find the list_attr_celeba.txt file")
        if not self.partition_path.is_file():
            raise FileNotFoundError("Couldn't find the list_eval_partition.txt file")
        if not self.images_dir.is_dir():
            raise FileNotFoundError("Couldn't find the im_align_celeba folder with the images")

        split_key = str(split).lower() 
        if split_key not in {"train", "val", "valid", "test"}: # Specify dataset split based on input
            raise ValueError("split must be one of: 'train', 'val'/'valid', 'test'")
        self.split_id = 0 if split_key == "train" else (1 if split_key in {"val", "valid"} else 2)

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
        self.return_group = return_group
        self.multi_label = multi_label

        self.target_attribute = target_attribute
        self.spurious_attribute = spurious_attribute
        self.class_names = [f"not_{target_attribute}", target_attribute]         # Labeling stuff e.g. not blond hair, and blond hair
        self.spurious_names = [f"not_{spurious_attribute}", spurious_attribute]  # Labeling stuff e.g. not male, and male

        ImageFile.LOAD_TRUNCATED_IMAGES = True # Load truncatd images without error

        with open(self.attribute_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        if len(lines) < 3:
            raise RuntimeError(f"Attribute file issue :()")

        attribute_names = lines[1].split()
        if target_attribute not in attribute_names:
            raise ValueError(f"The target_attribute of {target_attribute} was not found as a CelebA attribute")
        if spurious_attribute not in attribute_names:
            raise ValueError(f"The spurious_attribute of {spurious_attribute} was not found as a CelebA attribute")

        y_idx = attribute_names.index(target_attribute)
        a_idx = attribute_names.index(spurious_attribute)

        attribute_map: Dict[str, List[int]] = {}
        for line in lines[2:]:
            parts = line.split()
            if len(parts) < 1 + len(attribute_names):
                continue
            fname = parts[0]
            vals = parts[1:]

            # y = 1 if int(vals[y_idx]) == 1 else 0
            # a = 1 if int(vals[a_idx]) == 1 else 0
            # attribute_map[fname] = (y, a)

            vec = [1 if int(v) == 1 else 0 for v in vals]
            attribute_map[fname] = vec
            
        if len(attribute_map) == 0:
            raise RuntimeError(f"0 attributes in map!")

        
        self.image_paths: List[Path] = []
        self.filenames: List[str] = []
        self.a: List[int] = []
        self.g: List[int] = []

        self.y: List[int] = []           # Case: multi_label is false (1 attributes)
        self.y_all: List[List[int]] = [] # Case: multi_label is true (40 attributes)

        with open(self.partition_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) != 2:
                    continue
                fname, split_id_str = parts[0], parts[1]
                if int(split_id_str) != self.split_id:
                    continue

                if fname not in attribute_map:
                    raise RuntimeError(f"Partition file not found :(")

                img_path = self.images_dir/fname
                if not img_path.exists():
                    raise FileNotFoundError(f"Image not found: {img_path}")

                vec = attribute_map[fname]
                y_target = int(vec[y_idx])
                # y, a = attribute_map[fname]
                a = int(vec[a_idx])
                g = 2 * y_target + a

                self.image_paths.append(img_path)
                self.filenames.append(fname)
                self.a.append(a)
                self.g.append(g)

                if self.multi_label:
                    self.y_all.append(vec)
                else:
                    self.y.append(y_target)

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No samples found for split")
        
        if self.multi_label:
            self.y_all_tensor = torch.tensor(self.y_all, dtype = torch.float32)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """
        Returning:
            (image, y, a) or (image, y, a, g)
        """
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB") # Loading images as RGB and apply transforms
        img = self.transform(img)

        # y = self.y[idx]
        a = self.a[idx]

        if self.multi_label:
            y = self.y_all_tensor[idx] # Tensor[40] for all attributes
        else:
            y = self.y[idx]

        if self.return_group:
            g = self.g[idx]
            return img, y, a, g
        return img, y, a


# Dummy use:
if __name__ == "__main__":
    print("CelebA check...................................")

    ds_train = CelebADataset(
        root_dir="..",               
        split="train",
        target_attribute="Blond_Hair",
        spurious_attribute="Male",
        return_group=True,
        image_size=224,
        multi_label= True
    )

    print("Train samples:", len(ds_train))
    print("Target attribute:", ds_train.target_attribute)
    print("Spurious attribute:", ds_train.spurious_attribute)
    print("Class names:", ds_train.class_names)
    print("Spurious names:", ds_train.spurious_names)

    x, y, a, g = ds_train[0] # Retrieving a sample 
    # print("Example sample:", x.shape, y, a, g)
    print("Example sample:", x.shape, y.shape, y.dtype, a, g)  # y being torch.Size([40])
    print("Ende...................................")