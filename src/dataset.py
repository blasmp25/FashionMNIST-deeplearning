import torch
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

def create_dataloaders(batch_size=32):

    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        transform=ToTensor(),
        download=True
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        transform=ToTensor(),
        download=True
    )

    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False
    )
    
    class_names = train_data.classes
    
    return train_dataloader, test_dataloader, class_names
