import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2


def create_transforms(transform_configs: dict) -> A.Compose:
    """
    Composes a list of transforms from the configuration dictionary.

    Parameters
    ----------
    transform_configs : dict
        The configuration dictionary for the transforms.

    Returns
    -------
    albumentations.Compose
    """
    transforms_list = []
    for transform_config in transform_configs:
        transform_name, transform_params = next(iter(transform_config.items()))
        transform_params = transform_params or {}  # Handle null values

        if hasattr(A, transform_name):
            transform_class = getattr(A, transform_name)

            if transform_name == "Resize":
                interpolation_name = transform_params.get(
                    "interpolation", "INTER_LINEAR"
                )
                if hasattr(cv2, interpolation_name):
                    transform_params["interpolation"] = getattr(cv2, interpolation_name)
                else:
                    raise ValueError(
                        f"Interpolation mode {interpolation_name} not found in cv2"
                    )
            transforms_list.append(transform_class(**transform_params))
        else:
            raise ValueError(f"Transform {transform_name} not found in albumentations")

    transforms_list.append(ToTensorV2())

    return A.Compose(transforms_list)
