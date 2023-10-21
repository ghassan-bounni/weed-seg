import os
from PIL import Image

from torch.utils.data import Dataset, DataLoader

import torchvision.transforms
from data.augmentations import create_transforms


class StemDetectionDataset(Dataset):
    """
    Custom Dataset for Object Detection

    Attributes
    ----------
    images_dir : str
        The path to the images directory.

    masks_dir : str
        The path to the masks directory.

    transform : callable
        The transformation function to apply to the images.

    images : list
        The list of image names.

    masks : list
        The list of mask folder names.
    """

    def __init__(
        self, data_dir: str, transform: torchvision.transforms.Compose = None
    ) -> None:
        """
        Initializes the dataset.

        Parameters
        ----------
        data_dir : str
            The path to the data directory.

        transform : torchvision.transforms.Compose, optional
            The transformation function to apply to the images.
        """
        self.images_dir = os.path.join(data_dir, "images")
        self.masks_dir = os.path.join(data_dir, "masks")
        self.images = [
            os.path.join(self.images_dir, img_name)
            for img_name in os.listdir(self.images_dir)
        ]
        self.masks = [
            os.path.join(self.masks_dir, masks) for masks in os.listdir(self.masks_dir)
        ]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.images[idx])
        image = Image.open(img_name)

        masks = [
            Image.open(os.path.join(self.masks[idx], mask))
            for mask in os.listdir(self.masks[idx])
        ]

        if self.transform:
            image = self.transform(image)

        return image, masks


