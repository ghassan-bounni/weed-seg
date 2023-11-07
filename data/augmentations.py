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

            if transform_name == "Resize":
                interpolation_name = transform_params.get("interpolation", "BILINEAR")
                if hasattr(transforms.InterpolationMode, interpolation_name):
                    transform_params["interpolation"] = getattr(
                        transforms.InterpolationMode, interpolation_name
                    )
                else:
                    raise ValueError(
                        f"Interpolation mode {interpolation_name} "
                        f"not found in torchvision.transforms.InterpolationMode"
                    )
            transforms_list.append(transform_class(**transform_params))
        else:
            raise ValueError(
                f"Transform {transform_name} not found in torchvision.transforms"
            )

    return transforms.Compose(transforms_list)
