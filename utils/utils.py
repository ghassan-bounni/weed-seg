import os.path
from typing import Iterator

import argparse
from importlib import import_module

import torch.nn as nn
import torch.optim as optim

from utils.logger import LOGGER


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
        "--output", default="output", help="Path to the output directory."
    )
    parser.add_argument(
        "--device", default="cpu", help="Device to run the pipeline on."
    )
    parser.add_argument("--seed", default=42, help="Seed for reproducibility.")
    parser.add_argument(
        "--verbose", default=False, help="Whether to print the logs or not."
    )
    args = parser.parse_args()
    return args


def model_info(model: nn.Module, detailed: bool = False) -> tuple:
    """
    Model information.

    Parameters
    ----------
    model : nn.Module
        The model to get the information from.

    detailed : bool, optional
        Whether to print detailed information (defaults to False).

    Returns
    ----------
    tuple
        The number of layers, parameters, and gradients.
    """

    n_p = get_num_params(model)  # number of parameters
    n_g = get_num_gradients(model)  # number of gradients
    n_l = len(list(model.layers))  # number of layers

    if detailed:
        # Get the maximum length of the layer names and the layers for formatting
        max_name_len = max(len(name) for name in model.layer_names)
        max_layer_len = max(len(str(layer)) for layer in model.layers)
        total_len = max_name_len + max_layer_len + 7

        # Log network layers
        LOGGER.info("-" * total_len)
        LOGGER.info(f"| {'name':>{max_name_len}} | {'layer':>{max_layer_len}} |")
        LOGGER.info("-" * total_len)
        for layer, name in zip(model.layers, model.layer_names):
            LOGGER.info(f"| {name:>{max_name_len}} | {str(layer):>{max_layer_len}} |")
        LOGGER.info("-" * total_len)

        # Log detailed information
        LOGGER.info(
            f"{'layer':>5} {'name':>{max_name_len+10}} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'std':>10} {'dtype':>15}"
        )
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace("module_list.", "")
            name = f"{model.layer_names[int(name.split('.')[1])]}.{name.split('.')[-1]}"
            formatting = f"%5s %{max_name_len+10}s %9s %12g %20s %10.3g %10.3g %15s"
            LOGGER.info(
                formatting
                % (
                    i,
                    name,
                    p.requires_grad,
                    p.numel(),
                    list(p.shape),
                    p.mean(),
                    p.std(),
                    p.dtype,
                )
            )

    model_name = getattr(model, "name", "Model")

    # Log the summary
    LOGGER.info(
        f"{model_name} summary: {n_l} layers, {n_p} parameters, {n_g} gradients"
    )
    return n_l, n_p, n_g


def get_num_params(model):
    """Return the total number of parameters in a model."""
    return sum(x.numel() for x in model.parameters())


def get_num_gradients(model):
    """Return the total number of parameters with gradients in a model."""
    return sum(x.numel() for x in model.parameters() if x.requires_grad)


def load_model(
    cfg: dict,
    show_info: bool = True,
) -> nn.Module:
    """
    Loads a model based on the configuration.

    Parameters
    ----------
    cfg : dict
        The configuration of the model.

    show_info : bool, optional
        Whether to show the model info or not (default is True).

    Returns
    -------
    nn.Module
        The model.
    """

    model_name = cfg["name"]

    if os.path.exists(f"models/{model_name}.py"):
        module = import_module(f"models.{model_name}")
        model = getattr(module, model_name)(cfg)

        if show_info:
            model_info(model, detailed=cfg["summary"])

        return model
    else:
        raise ValueError(f"{model_name} is not found in models")


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
