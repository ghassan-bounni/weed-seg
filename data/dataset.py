import os
from PIL import Image
import random
import numpy

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
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

    image_transform : callable
        The transformation function to apply to the images.

    mask_transform : callable
        The transformation function to apply to the mask.

    images : list
        The list of image names.

    masks : list
        The list of mask folder names.
    """

    def __init__(
        self,
        data_dir: str,
        seed: int,
        image_transform: transforms.Compose = None,
        mask_transform: transforms.Compose = None,
    ) -> None:
        """
        Initializes the dataset.

        Parameters
        ----------
        data_dir : str
            The path to the data directory.

        image_transform : torchvision.transforms.Compose, optional
            The transformation function to apply to the images.

        mask_transform : torchvision.transforms.Compose, optional
            The transformation function to apply to the images.
        """
        self.seed = seed
        self.images_dir = os.path.join(data_dir, "images")
        self.masks_dir = os.path.join(data_dir, "masks")
        self.images = [
            os.path.join(self.images_dir, img_name)
            for img_name in os.listdir(self.images_dir)
        ]
        self.masks = [
            os.path.join(self.masks_dir, masks) for masks in os.listdir(self.masks_dir)
        ]
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path)
        seed = self.seed + idx

        masks = (
            [
                Image.open(os.path.join(self.masks[idx], mask))
                for mask in os.listdir(self.masks[idx])
            ]
            if os.path.isdir(self.masks[idx])
            else Image.open(self.masks[idx])
        )

        if self.image_transform:
            random.seed(seed)
            numpy.random.seed(seed)
            image = self.image_transform(image)

        if self.mask_transform:
            random.seed(seed)
            numpy.random.seed(seed)
            masks = self.mask_transform(masks)

        return image, masks, img_path.split("/")[-1].split(".")[0]


def create_dataloader(data_path, transforms_dict, batch_size, shuffle, seed):
    image_transforms = create_transforms(transforms_dict["image_transforms"])
    mask_transforms = create_transforms(transforms_dict["mask_transforms"])
    train_dataset = StemDetectionDataset(
        data_path, seed, image_transforms, mask_transforms
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_dataloader
