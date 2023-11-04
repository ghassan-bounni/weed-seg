import yaml


def load_config(config_path: str) -> dict:
    """
    Loads the configuration file.

    Parameters
    ----------
    config_path : str
        The path to the configuration file.

    Returns
    -------
    dict
        The configuration dictionary.
    """
    with open(config_path, "r") as config_file:
        return yaml.safe_load(config_file)
