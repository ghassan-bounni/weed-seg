import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from data.dataset import CustomObjectDetectionDataset
from data.augmentations import create_transforms

from tqdm import tqdm
import logging
from utils.utils import load_model, load_criterion


def test(
    model_config: dict,
    data_config: dict,
    checkpoint_path: str,
    device: torch.device,
    logger: logging.Logger = logging.getLogger(__name__),
    verbose: bool = False,
):
    """
    Evaluates the model.

    Parameters
    ----------
    model_config : dict
        The model configuration.

    data_config : dict
        The data configuration.

    checkpoint_path : str
        The path to the model checkpoint.

    device : torch.device
        The device to run the training on.

    logger : logging.Logger
        The logger to use.

    verbose : bool
        Whether to print the logs or not.
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
    )

    model = load_model(cfg=model_config)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    criterion = load_criterion(loss_fn=model.args["loss_fn"])

    test_transforms = create_transforms(data_config["test"]["transforms"])
    # test_dataset = CustomObjectDetectionDataset(
    #     data_config["test"]["path"], "yolo", test_transforms
    # )

    test_dataset = MNIST(
        data_config["test"]["path"],
        train=False,
        download=True,
        transform=test_transforms,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=model.args["batch_size"])

    if verbose:
        logger.info("Starting evaluation...")

    model.eval()
    with torch.inference_mode():
        accuracy = 0

        with tqdm(test_dataloader, ncols=100, unit="batch") as pbar:
            for batch_idx, (inputs, targets) in enumerate(pbar):
                # Move the data to the device
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Forward pass
                output = model(inputs)

                # Calculate the loss
                loss = criterion(output, targets)

                # TODO: Add metrics

                accuracy += (output.argmax(1) == targets).float().mean()
                # Update the progress bar
                pbar.set_postfix(
                    {"batch": batch_idx, "loss": loss.item()}, refresh=True
                )
        accuracy /= len(test_dataloader)
        if verbose:
            logger.info(f"Accuracy: {accuracy}")
        # TODO: Add logging
