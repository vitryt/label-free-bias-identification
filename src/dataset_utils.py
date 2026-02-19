from torch.utils.data import Dataset
from src.colour_mnist import get_biased_mnist_dataloader # Coloured MNIST
from src.waterbird import WaterBirdsDataset # Waterbirds
from src.celeba import CelebADataset # CelebA

from torch.utils.data import DataLoader


def get_dataloaders(dataset_name, data_path, batch_size, train_correlation=0.95, test_correlation=0, split=["train", "val", "test"], seeds=(42, 42)):
    split_seed, shuffle_seed = seeds
    train_dataloader = validation_dataloader = test_dataloader = None
    if dataset_name == "MNIST":
        if "train" in split or "val" in split:
            train_dataloader, validation_dataloader = get_biased_mnist_dataloader(
                root = data_path, 
                batch_size = batch_size, 
                data_label_correlation = train_correlation,
                train = True, 
                validation = 1/10, 
                split_gen_seed = split_seed, 
                shuffle_seed = shuffle_seed
            )
        if "test" in split:
            test_dataloader = get_biased_mnist_dataloader(
            root = data_path, 
            batch_size = batch_size, 
            data_label_correlation = test_correlation,
            train = False, 
            shuffle_seed = shuffle_seed
        )
    elif dataset_name == "Waterbirds":
        transform = None  # Using default
        if "train" in split:
            train_dataloader = DataLoader(WaterBirdsDataset(data_path, "train", transform), batch_size = batch_size, shuffle = True)
        if "val" in split:
            validation_dataloader = DataLoader(WaterBirdsDataset(data_path, "val", transform), batch_size = batch_size)
        if "test" in split:
            test_dataloader = DataLoader(WaterBirdsDataset(data_path, "test", transform), batch_size = batch_size)
    elif dataset_name == "CelebA":
        transform = None  # Using default
        if "train" in split:
            train_dataloader = DataLoader(CelebADataset(data_path, "train", transform=transform), batch_size = batch_size, shuffle = True)
        if "val" in split:
            validation_dataloader = DataLoader(CelebADataset(data_path, "val", transform=transform), batch_size = batch_size)
        if "test" in split:
            test_dataloader = DataLoader(CelebADataset(data_path, "test", transform=transform), batch_size = batch_size)
    res = []
    for split_label, dataloader in (("train", train_dataloader), ("val", validation_dataloader), ("test", test_dataloader)):
        if split_label in split:
            res.append(dataloader)
    return res if len(res) > 1 else res[0]