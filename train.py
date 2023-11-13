import os

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import PolynomialLR, StepLR
from torchinfo import summary

from tqdm import tqdm
from logger import MetricLogger
from models.base import BaseModel
from data.dataset import create_dataloader

from utils.utils import load_criterion, load_checkpoint, save_checkpoint


def train_one_epoch(
    epoch,
    model,
    train_dataloader,
    optimizer,
    criterion,
    clip_grad,
    device,
):
    train_logger = MetricLogger(delimiter=" ")

    model.train()
    for inputs, labels, img_ids in train_logger.log_every(
        train_dataloader,
        print_freq=1,
        header=f"Epoch: {epoch} Train:",
    ):
        inputs = inputs.to(device)
        labels = labels.to(device)

        output = model(inputs)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()

        if clip_grad is not None:
            clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        train_logger.update(loss=loss.item())

    return {k: meter.global_avg for k, meter in train_logger.meters.items()}


def validate(epoch, model, val_dataloader, criterion, device):
    val_logger = MetricLogger(delimiter=" ")

    model.eval()
    for val_inputs, val_labels, val_img_ids in val_logger.log_every(
        val_dataloader,
        print_freq=1,
        header=f"Epoch: {epoch} Val:",
    ):
        val_inputs = val_inputs.to(device)
        val_labels = val_labels.to(device)

        with torch.no_grad():
            val_output = model(val_inputs)
            val_loss = criterion(val_output, val_labels).item()

        val_logger.update(val_loss=val_loss)

    return {k: meter.global_avg for k, meter in val_logger.meters.items()}


def train(
    model_config: dict,
    train_config: dict,
    val_config: dict,
    data_path: str,
    save_interval: int,
    seed: int,
    checkpoint_name: str = None,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if seed is not None:
        seed = seed
        torch.manual_seed(seed)

        if device == "cuda":
            torch.cuda.manual_seed(seed)

    (
        epochs,
        batch_size,
        num_workers,
        warmup_epochs,
        weight_decay,
        lr,
        lr_gamma,
        lr_step,
        lr_power,
        lr_scheduler,
        clip_grad,
        loss_fn,
        transforms,
    ) = train_config.values()

    val_batch_size, val_num_workers, val_transforms = val_config.values()

    train_dataloader = create_dataloader(
        data_path=os.path.join(data_path, "train"),
        transforms_dict=transforms,
        batch_size=batch_size,
        shuffle=True,
        num_classes=model_config["out_channels"],
        num_workers=num_workers,
    )

    val_dataloader = create_dataloader(
        data_path=os.path.join(data_path, "val"),
        transforms_dict=val_transforms,
        batch_size=val_batch_size,
        shuffle=False,
        num_classes=model_config["out_channels"],
        num_workers=val_num_workers,
    )

    model = BaseModel(**model_config)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    criterion = load_criterion(loss_fn)

    start_epoch = load_checkpoint("train", model, optimizer, checkpoint_name)
    summary(model, input_size=(batch_size, *train_dataloader.dataset[0][0].shape))

    scheduler = (
        PolynomialLR(optimizer, total_iters=epochs, power=lr_power)
        if (lr_scheduler == "poly")
        else StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    )

    model.train()
    with tqdm(range(start_epoch, epochs), desc="Epochs", unit="epoch") as pbar:
        for epoch in pbar:
            train_metrics = train_one_epoch(
                epoch,
                model,
                train_dataloader,
                optimizer,
                criterion,
                clip_grad,
                device,
            )

            val_metrics = validate(epoch, model, val_dataloader, criterion, device)

            pbar.set_postfix(
                **train_metrics,
                **val_metrics,
                lr=optimizer.param_groups[0]["lr"],
            )

            # Gradual warm-up: Adjust learning rate for warm-up epochs
            if epoch < warmup_epochs:
                warmup_factor = epoch / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group["lr"] = param_group["initial_lr"] * warmup_factor
            # Update the learning rate after warm-up
            else:
                scheduler.step()

            save_checkpoint(
                epoch, model.state_dict(), optimizer.state_dict(), save_interval
            )
