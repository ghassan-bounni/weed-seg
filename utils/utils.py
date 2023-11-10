from typing import Iterator

import os
import json
import logging
import argparse

import cv2
import numpy as np
from importlib import import_module

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger("StemDetectionLogger")


def parse_args():
    parser = argparse.ArgumentParser(description="Model Pipeline")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to the config file."
    )
    parser.add_argument(
        "--mode",
        default="train",
        help="Mode to run the pipeline in.",
        choices=["train", "eval"],
    )
    parser.add_argument(
        "--checkpoint", default=None, help="Path to the checkpoint file."
    )
    parser.add_argument(
        "--data-path", default="data/", help="Path to the data directory."
    )
    parser.add_argument("--output", default=None, help="Path to the output directory.")
    parser.add_argument("--seed", default=42, help="Seed for reproducibility.")
    parser.add_argument(
        "--save-interval", default=1, help="Checkpoint save interval (epochs)."
    )

    args = parser.parse_args()
    return args


def load_optimizer(
    optimizer_name: str, model_params: Iterator, lr: float
) -> optim.Optimizer:
    """
    Loads an optimizer based on the hyperparameters.

    Parameters
    ----------
    optimizer_name : str
        The name of the optimizer.

    model_params : Iterator
        The model parameters.

    lr : float
        The learning rate.

    Returns
    -------
    optim.Optimizer
        The optimizer.
    """

    if hasattr(optim, optimizer_name):
        return getattr(optim, optimizer_name)(model_params, lr=lr)
    else:
        raise ValueError(f"{optimizer_name} is not found in torch.optim")


def load_criterion(loss_fn: str, **kwargs) -> nn.Module:
    """
    Loads a criterion.

    Parameters
    ----------
    loss_fn : str
        The name of the loss function.

    **kwargs:
        Additional keyword arguments to be passed to the criterion.

    Returns
    -------
    nn.Module
        The criterion.
    """

    if hasattr(nn, loss_fn):
        return getattr(nn, loss_fn)()
    else:
        loss_module = import_module("utils.loss")
        if hasattr(loss_module, loss_fn):
            return getattr(loss_module, loss_fn)(**kwargs)
        else:
            raise ValueError(f"{loss_fn} is not found in torch.nn or utils.loss")


def load_checkpoint(
    mode: str,
    model: nn.Module,
    optimizer: optim.Optimizer = None,
    checkpoint_name: str = None,
):
    """
    Loads a selected checkpoint if it exists or checks for the latest checkpoint.\n
    If no checkpoint exists, starts from epoch 0.
    """

    checkpoint_path = (
        "checkpoints/checkpoint_latest.pth"
        if checkpoint_name is None
        else f"checkpoints/{checkpoint_name}"
    )

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state_dict"])

        if mode == "train":
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info(f"Loaded checkpoint: {checkpoint_path} (epoch {epoch})")
        return epoch + 1

    else:
        if checkpoint_name:
            raise ValueError(f"Checkpoint: {checkpoint_name} not found.")
        else:
            logger.info("No checkpoint found. Starting from epoch 0.")
            return 0


def save_checkpoint(epoch, model_state, optimizer_state, save_interval):
    """
    Saves epoch number, model state, and optimizer state to a checkpoint.
    """
    os.makedirs("checkpoints", exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
    }
    torch.save(checkpoint, "checkpoints/checkpoint_latest.pth")

    if epoch % save_interval == 0:
        checkpoint_name = f"checkpoints/checkpoint_{epoch}.pth"
        torch.save(checkpoint, checkpoint_name)


def get_stem_coordinates(
    predictions: torch.Tensor, num_classes: int, img_ids: list
) -> list:
    stem_coordinates = []

    # getting stem coordinates per class for each image in the predictions tensor
    for image_idx, image_argmax in enumerate(predictions):
        image_id = img_ids[image_idx]

        # skipping the background class (class_id = 0)
        for class_id in range(1, num_classes):
            binary_mask = (image_argmax == class_id).cpu().numpy().astype(np.uint8)
            _, _, _, centroids = cv2.connectedComponentsWithStats(
                binary_mask, connectivity=8
            )
            stem_coordinates[class_id] = {
                "file_name": image_id,
                "stems": centroids[1:].tolist(),
            }

    return stem_coordinates


def save_data_to_json(data, file_name: str, output_dir: str):
    """
    Saves data to a json file.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(output_dir, file_name), "w") as f:
        json.dump(data, f)


# TODO: utility functions and classes from ultralytics
# Adapted from https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/tal.py
def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format
    where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray | torch.Tensor):
            The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor):
            The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert (
        x.shape[-1] == 4
    ), f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = (
        torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)
    )  # faster than clone/copy
    dw = x[..., 2] / 2  # half-width
    dh = x[..., 3] / 2  # half-height
    y[..., 0] = x[..., 0] - dw  # top left x
    y[..., 1] = x[..., 1] - dh  # top left y
    y[..., 2] = x[..., 0] + dw  # bottom right x
    y[..., 3] = x[..., 1] + dh  # bottom right y
    return y


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = (
            torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        )  # shift x
        sy = (
            torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        )  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(
        0, reg_max - 0.01
    )  # dist (lt, rb)


def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
    """
    Select the positive anchor center in gt.

    Args:
        xy_centers (Tensor): shape(h*w, 2)
        gt_bboxes (Tensor): shape(b, n_boxes, 4)

    Returns:
        (Tensor): shape(b, n_boxes, h*w)
    """
    n_anchors = xy_centers.shape[0]
    bs, n_boxes, _ = gt_bboxes.shape
    lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
    bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(
        bs, n_boxes, n_anchors, -1
    )
    # return (bbox_deltas.min(3)[0] > eps).to(gt_bboxes.dtype)
    return bbox_deltas.amin(3).gt_(eps)


def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """
    If an anchor box is assigned to multiple gts, the one with the highest IoI will be selected.

    Args:
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
        overlaps (Tensor): shape(b, n_max_boxes, h*w)

    Returns:
        target_gt_idx (Tensor): shape(b, h*w)
        fg_mask (Tensor): shape(b, h*w)
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
    """
    # (b, n_max_boxes, h*w) -> (b, h*w)
    fg_mask = mask_pos.sum(-2)
    if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(
            -1, n_max_boxes, -1
        )  # (b, n_max_boxes, h*w)
        max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)

        is_max_overlaps = torch.zeros(
            mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device
        )
        is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

        mask_pos = torch.where(
            mask_multi_gts, is_max_overlaps, mask_pos
        ).float()  # (b, n_max_boxes, h*w)
        fg_mask = mask_pos.sum(-2)
    # Find each grid serve which gt(index)
    target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
    return target_gt_idx, fg_mask, mask_pos
