import os

from torchvision import transforms
import torchvision
from torch import Tensor
import torch


def load_images() -> tuple[Tensor, Tensor]:
    """
    Loads and preprocesses brain MRI images and generates corresponding labels.

    The function processes images from two directories:
    - images/yes/: Contains MRI images with tumors (labeled as 1)
    - images/no/: Contains MRI images without tumors (labeled as 0)

    Each image is preprocessed by:
    - Converting to grayscale (if multi-channel)
    - Resizing to 218x180 pixels
    - Normalizing pixel values to [0,1] range
    - Converting to torch.float32 dtype

    Returns:
        tuple[Tensor, Tensor]: A tuple containing:
            - Images tensor of shape (N, 218, 180) where N is total number of images
            - Labels tensor of shape (N,) with binary labels (1 for tumor, 0 for no tumor)
    """

    def preprocess_image(img_file_path: str) -> Tensor:
        """
        Preprocesses a single image file for the brain MRI dataset.

        The function performs the following preprocessing steps:
        - For RGB (3-channel) images: converts to grayscale
        - For RGBA (4-channel) images: drops alpha channel and converts to grayscale
        - For single channel images: keeps as-is
        - Resizes image to 218x180 pixels
        - Converts to float32 dtype
        - Normalizes pixel values to [0,1] range

        Args:
            img_file_path (str): Path to the image file to preprocess

        Returns:
            Tensor: Preprocessed image tensor of shape (218, 180)
        """
        transforms_multi_channels = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((250, 200)),
                transforms.ConvertImageDtype(torch.float32),
            ]
        )

        transforms_1_channel = transforms.Compose(
            [transforms.Resize((250, 200)), transforms.ConvertImageDtype(torch.float32)]
        )

        img: Tensor = torchvision.io.decode_image(img_file_path)
        if img.shape[0] == 3:
            return transforms_multi_channels(img) / 255
        elif img.shape[0] == 4:
            return transforms_multi_channels(img[:3, :, :]) / 255
        else:
            return transforms_1_channel(img) / 255

    flip_transform = transforms.RandomHorizontalFlip(p=1.0)

    yes_images: Tensor = torch.stack(
        [
            preprocess_image(f"images/yes/{file_name}")
            for file_name in os.listdir("images/yes")
        ]
        + [
            flip_transform(preprocess_image(f"images/yes/{file_name}"))
            for file_name in os.listdir("images/yes")
        ]
    )
    yes_targets: Tensor = torch.ones(yes_images.shape[0])
    no_images: Tensor = torch.stack(
        [
            preprocess_image(f"images/no/{file_name}")
            for file_name in os.listdir("images/no")
        ]
        + [
            flip_transform(preprocess_image(f"images/no/{file_name}"))
            for file_name in os.listdir("images/no")
        ]
    )
    no_targets: Tensor = torch.zeros(no_images.shape[0])

    return torch.cat([yes_images, no_images]), torch.cat([yes_targets, no_targets])
