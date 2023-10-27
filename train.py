import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR, StepLR

from logger import MetricLogger
from models.base import BaseModel
from data.dataset import create_dataloader

from utils.utils import load_criterion, load_checkpoint, save_checkpoint


def train(
    model_config: dict,
    train_config: dict,
    val_config: dict,
    save_interval: int,
    seed: int,
    checkpoint_name: str = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if seed is not None:
        seed = seed
        torch.manual_seed(seed)

        if device == "cuda":
            torch.cuda.manual_seed(seed)

    (
        train_datapath,
        epochs,
        batch_size,
        warmup_epochs,
        weight_decay,
        lr,
        lr_decay,
        lr_step,
        lr_scheduler,
        clip_grad,
        loss_fn,
        train_transforms,
    ) = train_config.values()

    val_datapath, val_batch_size, val_transforms = val_config.values()

    train_dataloader = create_dataloader(
        train_datapath, train_transforms, batch_size, True
    )

    val_dataloader = create_dataloader(
        val_datapath, val_transforms, val_batch_size, False
    )

    model = BaseModel(**model_config)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    criterion = load_criterion(loss_fn)

    start_epoch = load_checkpoint("train", model, optimizer, checkpoint_name)

    scheduler = (
        LambdaLR(optimizer, lambda e: (1 - e / epochs) ** lr_decay)
        if (lr_scheduler == "poly")
        else StepLR(optimizer, step_size=lr_step, gamma=lr_decay)
    )

    metric_logger = MetricLogger(delimiter=" ")

    total_iterations = epochs * len(train_dataloader)
    start_iteration = start_epoch * len(train_dataloader)

    for batch_idx, (inputs, labels, img_ids) in metric_logger.log_every(
        enumerate(train_dataloader),
        print_freq=1,
        header="Train:",
        n_iterations=total_iterations,
        start_iteration=start_iteration,
    ):
        model.train()
        epoch = batch_idx // len(train_dataloader) + start_epoch

        inputs = inputs.to(device)
        labels = labels.to(device)

        output = model(inputs)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()

        if clip_grad is not None:
            clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        metric_logger.update(loss=loss.item())

        if batch_idx + 1 % len(train_dataloader) == 0:
            model.eval()
            val_loss = 0.0
            for val_batch_idx, (val_inputs, val_labels, val_img_ids) in enumerate(
                val_dataloader
            ):
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)

                with torch.no_grad():
                    val_output = model(val_inputs)
                    val_loss += criterion(val_output, val_labels).item()

            val_loss /= len(val_dataloader)
            metric_logger.update(val_loss=val_loss)

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
