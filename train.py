import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR, StepLR

from models.base import BaseModel
from data.dataset import create_dataloader

import logging
from utils.utils import load_criterion, load_checkpoint, save_checkpoint


def train(
    model_config: dict,
    train_config: dict,
    save_interval: int,
    checkpoint_name: str = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger("StemDetectionLogger")

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

    train_dataloader = create_dataloader(train_datapath, train_transforms, batch_size)

    model = BaseModel(**model_config)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay)
    criterion = load_criterion(loss_fn)

    start_epoch = load_checkpoint("train", model, optimizer, checkpoint_name)

    scheduler = (
        LambdaLR(optimizer, lambda e: (1 - e / epochs) ** lr_decay)
        if (lr_scheduler == "poly")
        else StepLR(optimizer, step_size=lr_step, gamma=lr_decay)
    )

    for epoch in range(start_epoch, epochs):
        model.train()

        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            output = model(inputs)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()

            if clip_grad is not None:
                clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_dataloader)

        # Gradual warm-up: Adjust learning rate for warm-up epochs
        if epoch < warmup_epochs:
            warmup_factor = epoch / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["initial_lr"] * warmup_factor
        # Update the learning rate after warm-up
        else:
            scheduler.step()

        logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}")

        save_checkpoint(
            epoch, model.state_dict(), optimizer.state_dict(), save_interval
        )
