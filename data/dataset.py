import os
import numpy
import random
from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from data.augmentations import create_transforms
from scripts.preprocessing import create_binary_masks


class StemDetectionDataset(Dataset):
    """
    Custom Dataset for Plant Segmentation and Stem Detection.
    """

    def __init__(
        self,
        data_dir: str,
        seed: int,
        num_classes: int,
        image_transform: transforms.Compose = None,
        mask_transform: transforms.Compose = None,
    ) -> None:
        """
        Initializes the dataset.

        Parameters
        ----------
        data_dir : str
            The path to the data directory.

        seed : int
            The seed to use for the random number generator.

        num_classes : int
            The number of classes in the dataset.

        image_transform : torchvision.transforms.Compose, optional
            The transformation function to apply to the images.

        mask_transform : torchvision.transforms.Compose, optional
            The transformation function to apply to the images.
        """
        self.seed = seed
        self.num_classes = num_classes
        self.images_dir = os.path.join(data_dir, "images")
        self.masks_dir = os.path.join(data_dir, "masks")
        self.images = [
            os.path.join(self.images_dir, img_name)
            for img_name in os.listdir(self.images_dir)
        ]

        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path)

        masks_path = img_path.split(".")[0] + ".png"
        mask = Image.open(masks_path)

        seed = numpy.random.randint(self.seed)
        if self.image_transform:
            random.seed(seed)
            numpy.random.seed(seed)
            image = self.image_transform(image)

        if self.mask_transform:
            random.seed(seed)
            numpy.random.seed(seed)
            mask = self.mask_transform(mask)

        masks = create_binary_masks(mask, self.num_classes)

        return image, masks, img_path.split("/")[-1].split(".")[0]


def create_dataloader(
    data_path, transforms_dict, batch_size, shuffle, seed, num_classes
):
    image_transforms = create_transforms(transforms_dict["image_transforms"])
    mask_transforms = create_transforms(transforms_dict["mask_transforms"])
    train_dataset = StemDetectionDataset(
        data_path, seed, num_classes, image_transforms, mask_transforms
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_dataloader
