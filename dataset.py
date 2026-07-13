import matplotlib
matplotlib.use('Agg') # Force headless mode to prevent PySide6 GUI crashes

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np

def get_data_loaders(batch_size=64, resize_for_tl=False):
    # Dynamically configure transformations based on the chosen model target
    if resize_for_tl:
        # 1. Setup ImageNet normalization statistics for Transfer Learning (MobileNetV2)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        train_transform = transforms.Compose([
            transforms.Resize((128, 128)), # Resizes to 128x128 to balance performance and accuracy. If we use 224x224, we can maybe achieve 1-3% more accuracy but the training will take more time and we will use more RAM/VRAM. It works also for anything above 128 especially that we are using Global Average Pooling (GAP).
            transforms.RandomHorizontalFlip(), # Flips images randomly to teach the model that a truck facing left is still a truck.
            transforms.ToTensor(), # Converts pixels to decimals between 0.0 and 1.0.
            transforms.Normalize(mean, std) # Color shifts the pixels to match the ImageNet distribution baseline.
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        # 2. Setup standard CIFAR-10 normalization statistics for Custom CNN from scratch
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4), # Pads images with a small border and crops them randomly to simulate different camera angles.
            transforms.RandomHorizontalFlip(), # Basic flip augmentation. It takes more computational time.
            transforms.ToTensor(), # Converts pixels to decimals between 0.0 and 1.0.
            transforms.Normalize(mean, std) # Color shifts the pixels to match the native CIFAR-10 baseline.
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    # Load the raw dataset TWICE with different transform assignments
    # The 'download=True' argument here handles pulling directly from PyTorch's official servers
    train_base = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    val_base = datasets.CIFAR10(root='./data', train=True, download=True, transform=test_transform) # <-- Clean images!
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    
    # Generate a fixed, reproducible shuffle of indices
    dataset_length = len(train_base)
    indices = torch.randperm(dataset_length).tolist()
    
    train_size = int(0.8 * dataset_length)
    
    # Extract Subsets using the correct transform maps
    train_dataset = Subset(train_base, indices[:train_size])
    val_dataset = Subset(val_base, indices[train_size:]) # <-- Slices the clean dataset!
    
    # Construct final DataLoader processing streams
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader


def visualize_samples(dataloader):
    """Visualizes a batch of images and saves them directly to our directory."""
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    
    images = images.numpy()
    fig = plt.figure(figsize=(10, 4))
    
    for idx in range(8):
        ax = fig.add_subplot(2, 4, idx+1, xticks=[], yticks=[])
        img = np.transpose(images[idx], (1, 2, 0))
        # Approximate denormalization back to visible RGB channels for previewing
        img = img * np.array([0.2023, 0.1994, 0.2010]) + np.array([0.4914, 0.4822, 0.4465])
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        # takes the specific label tensor (e.g., Tensor(3)) and converts it into a plain Python integer (3).
        # It then passes that integer to the classes list, which maps 3 directly to the text string "cat", placing it as the title of that specific subplot.
        ax.set_title(classes[labels[idx].item()])
    plt.tight_layout()
    
    plt.savefig('dataset_samples.png')
    print("Dataset preview grid generated and saved as 'dataset_samples.png'!")
    plt.close()


if __name__ == '__main__':
    print("Testing dataset downloading and preprocessing pipeline...")
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=8, resize_for_tl=False)
    print("Data loaders created successfully!")
    visualize_samples(train_loader)