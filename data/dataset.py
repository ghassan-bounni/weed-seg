import os
import cv2

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
        num_classes: int,
        transform=None,
    ) -> None:
        """
        Initializes the dataset.

        Parameters
        ----------
        data_dir : str
            The path to the data directory.

        num_classes : int
            The number of classes in the dataset.

        transform : albumentations.Compose, optional
            The transformation function to apply to the images.
        """
        self.num_classes = num_classes
        self.images_dir = os.path.join(data_dir, "images")
        self.masks_dir = os.path.join(data_dir, "masks")
        self.images = os.listdir(self.images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks_path = os.path.join(
            self.masks_dir, self.images[idx].split(".")[0] + ".png"
        )
        mask = cv2.imread(masks_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        masks = create_binary_masks(mask, self.num_classes)

        return image, masks, img_path.split("/")[-1].split(".")[0]


def create_dataloader(
    data_path, transforms_dict, batch_size, shuffle, num_classes, num_workers
):
    train_dataset = StemDetectionDataset(
        data_path, num_classes, create_transforms(transforms_dict)
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )
    return train_dataloader
