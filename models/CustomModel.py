import torch
import torch.nn as nn


class CustomModel(nn.Module):
    """
    A custom model based on the configuration.

    Attributes
    ----------
    name : str
        The name of the model.

    layers : nn.ModuleList
        The list of layers of the model.

    layer_names : list
        The list of layer names of the model.

    args : dict
        The hyperparameters of the model.

    """

    def __init__(self, model_config: dict) -> None:
        """
        Initializes the model.

        Parameters
        ----------
        model_config : dict, optional
            The configuration of the model.

        Raises
        ----------
        NotImplementedError
            If the layer type is not implemented.
        """
        super(CustomModel, self).__init__()

        # Set the name of the model
        self.name = model_config["name"]

        # Set the model's hyperparameters
        self.args = model_config["hyperparameters"]

        # Create the layers list
        self.layers = nn.ModuleList()

        # Create the layer names list
        self.layer_names = []

        # Looping through the layers in the model_config
        for layer_config in model_config["layers"]:
            layer_type = layer_config["type"]
            layer_config.pop("type")

            if hasattr(nn, layer_type):
                # Create the layer
                layer_name = layer_config["name"]
                layer_config.pop("name")

                self.layer_names.append(layer_name)

                self.layers.append(getattr(nn, layer_type)(**layer_config))
            else:
                raise ValueError(f"Layer type {layer_type} is not in torch.nn module.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        ----------
        x : torch.Tensor
            The output tensor representing the model's prediction.
        """
        for layer in self.layers:
            x = layer(x)
        return x
