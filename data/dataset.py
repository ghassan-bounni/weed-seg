import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def load_yolo_annotations(annotation_path: str) -> list:
    """
    Loads the annotations in YOLO format.

    Parameters
    ----------
    annotation_path : str
        The path to the annotation file.

    Returns
    -------
    list
    """
    annotations = []
    with open(annotation_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:  # YOLO format has 5 values per line
                class_label, x_center, y_center, width, height = map(float, parts)
                # Convert YOLO format to (x_min, y_min, x_max, y_max) format
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                x_max = x_center + width / 2
                y_max = y_center + height / 2
                annotations.append([x_min, y_min, x_max, y_max, class_label])
    return annotations


class CustomObjectDetectionDataset(Dataset):
    """
    Custom Dataset for Object Detection

    Attributes
    ----------
    images_dir : str
        The path to the images directory.

    annotations_dir : str
        The path to the annotations directory.

    transform : callable
        The transformation function to apply to the images.

    images : list
        The list of image names.

    annotations : list
        The list of annotation names.
    """

    def __init__(
        self, data_dir: str, data_format: str, transform: transforms.Compose = None
    ) -> None:
        """
        Initializes the dataset.

        Parameters
        ----------
        data_dir : str
            The path to the data directory.

        data_format : str
            The format of the annotations.

        transform : transforms.Compose, optional
            The transformation function to apply to the images.
        """
        self.images_dir = os.path.join(data_dir, "images")
        self.annotations_dir = os.path.join(data_dir, "annotations")
        self.format = data_format
        self.transform = transform
        self.images = os.listdir(self.images_dir)
        self.annotations = os.listdir(self.annotations_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.images[idx])
        image = Image.open(img_name)

        annotation_name = os.path.join(self.annotations_dir, self.annotations[idx])

        if self.format == "yolo":
            annotations = load_yolo_annotations(annotation_name)

            if self.transform:
                image = self.transform(image)

            return image, annotations
        else:
            raise ValueError(f"Format {self.format} not supported")
