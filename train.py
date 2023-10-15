import torch
from torch.utils.data import DataLoader
from data.dataset import CustomObjectDetectionDataset
from data.augmentations import create_transforms

from tqdm import tqdm
from utils.logger import LOGGER
from utils.utils import load_model, load_optimizer, load_criterion


def train(
    model_config: dict,
    data_config: dict,
    device: torch.device,
    verbose: bool = False,
):
    """
    Trains the model.

    Parameters
    ----------
    model_config : dict
        The model configuration.

    data_config : dict
        The data configuration.

    device : torch.device
        The device to run the training on.

    verbose : bool
        Whether to print the logs or not.
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
    )

    model = load_model(cfg=model_config)
    model.to(device)

    optimizer = load_optimizer(
        optimizer_name=model.args["optimizer"],
        model_params=model.parameters(),
        lr=model.args["learning_rate"],
    )

    criterion = load_criterion(loss_fn=model.args["loss_fn"])

    train_transforms = create_transforms(data_config["train"]["transforms"])
    train_dataset = CustomObjectDetectionDataset(
        data_config["train"]["path"], "yolo", train_transforms
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=model.args["batch_size"], shuffle=True
    )

    valid_transforms = create_transforms(data_config["valid"]["transforms"])
    valid_dataset = CustomObjectDetectionDataset(
        data_config["valid"]["path"], "yolo", valid_transforms
    )
    valid_dataloader = DataLoader(valid_dataset, batch_size=model.args["batch_size"])

    LOGGER.info("Starting training...")

    model.train()
    for epoch in range(model.args["num_epochs"]):
        with tqdm(
            train_dataloader, ncols=100, desc=f"Epoch {epoch+1}", unit="batch"
        ) as pbar:
            for batch_idx, (inputs, targets) in enumerate(pbar):
                # Move the data to the device
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Forward pass
                output = model(inputs)

                # Calculate the loss
                loss = criterion(output, targets)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # TODO: Add metrics

                # Update the progress bar
                pbar.set_postfix(
                    {"batch": batch_idx, "loss": loss.item()}, refresh=True
                )

                # TODO: Add logging

            # Save the model
            if epoch % 5 == 0:
                torch.save(model.state_dict(), f"checkpoints/model_{epoch}.pt")

            model.eval()  # Set the model to evaluation mode
            valid_loss = 0.0
            with torch.no_grad():
                for batch_idx, (valid_inputs, valid_targets) in enumerate(
                    valid_dataloader
                ):
                    valid_inputs = valid_inputs.to(device)
                    valid_targets = valid_targets.to(device)

                    valid_output = model(valid_inputs)
                    valid_loss += criterion(valid_output, valid_targets).item()

                    # TODO: Calculate and record validation metrics (e.g., accuracy, etc.)

            valid_loss /= len(valid_dataloader)
            LOGGER.info(f"Validation loss after epoch {epoch+1}: {valid_loss}")

            model.train()  # Set the model back to training mode
