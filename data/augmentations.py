import torchvision.transforms as transforms


def create_transforms(transform_configs: dict) -> transforms.Compose:
    """
    Composes a list of transforms from the configuration dictionary.

    Parameters
    ----------
    transform_configs : dict
        The configuration dictionary for the transforms.

    Returns
    -------
    transforms.Compose
    """
    transforms_list = []
    for transform_config in transform_configs:
        transform_name, transform_params = next(iter(transform_config.items()))
        if transform_params is None:
            transform_params = {}  # Handle null values

        if hasattr(transforms, transform_name):
            transform_class = getattr(transforms, transform_name)
            transforms_list.append(transform_class(**transform_params))
        else:
            raise ValueError(
                f"Transform {transform_name} not found in torchvision.transforms"
            )

    return transforms.Compose(transforms_list)
