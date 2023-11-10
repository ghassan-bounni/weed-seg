import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """
    A Depthwise Separable Convolutional layer.

    Attributes
    ----------
    depthwise : nn.Conv2d
        The depthwise convolutional layer.

    pointwise : nn.Conv2d
        The pointwise convolutional layer.

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ConvNormAct(nn.Module):
    """
    A Convolutional layer with Activation and Batch Normalization.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        conv_mode="full",
        **kwargs,
    ):
        super(ConvNormAct, self).__init__()

        params = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "bias": bias,
        }

        if conv_mode == "full":
            params.update(kwargs)

        self.layers = nn.Sequential(
            nn.Conv2d(**params)
            if conv_mode == "full"
            else DepthwiseSeparableConv(**params),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class DenseLayer(nn.Module):
    def __init__(
        self,
        num_input_features,
        kernel_size,
        growth_rate,
        bn_size,
        drop_rate,
        conv_mode="full",
        **kwargs,
    ) -> None:
        super(DenseLayer, self).__init__()

        self.add_module(
            "bottleneck",
            ConvNormAct(
                num_input_features,
                bn_size * growth_rate,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
        )

        self.add_module(
            "conv",
            ConvNormAct(
                bn_size * growth_rate,
                growth_rate,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=False,
                conv_mode=conv_mode,
                **kwargs,
            ),
        )
        self.drop_rate = float(drop_rate)

    def forward(self, prev_features: list[torch.Tensor]) -> torch.Tensor:
        concated_features = torch.cat(prev_features, 1)
        bottleneck_output = self.bottleneck(concated_features)

        new_features = self.conv(bottleneck_output)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training
            )
        return new_features


class DenseBlock(nn.ModuleDict):
    def __init__(
        self,
        num_layers,
        num_input_features,
        kernel_size,
        bn_size,
        growth_rate,
        drop_rate,
        conv_mode,
        concat_output=True,
        **kwargs,
    ) -> None:
        super(DenseBlock, self).__init__()

        self.concat_output = concat_output
        for i in range(num_layers):
            layer = DenseLayer(
                num_input_features + i * growth_rate,
                kernel_size,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                conv_mode=conv_mode,
                **kwargs,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features) -> torch.Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return (
            torch.cat(features, 1) if self.concat_output else torch.cat(features[1:], 1)
        )


class Transition(nn.Sequential):
    """
    A Transition layer between two dense blocks.
    """

    def __init__(self, num_input_features, num_output_features):
        super(Transition, self).__init__()

        self.add_module(
            "bottleneck",
            ConvNormAct(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
        )

        self.add_module(
            "down_sample",
            ConvNormAct(
                num_output_features,
                num_output_features,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            ),
        )

    def forward(self, x):
        x = self.bottleneck(x)
        x = self.down_sample(x)
        return x


class Encoder(nn.Module):
    """
    The encoder of the model.

    Attributes
    ----------
    in_channels : list[int]
        The number of input channels for each dense block.

    out_channels : list[int]
        The number of output channels for each dense block.

    """

    def __init__(
        self,
        num_layers,
        block_depth,
        num_init_features,
        kernel_size,
        bn_size,
        growth_rate,
        exp_factor,
        drop_rate,
        conv_mode,
        **kwargs,
    ):
        super(Encoder, self).__init__()

        self.in_channels = [num_init_features]
        self.out_channels = []

        self.features = nn.ModuleDict()
        self.skip_connections = []

        num_features = num_init_features
        for i in range(num_layers):
            block = DenseBlock(
                num_layers=block_depth * (exp_factor**i),
                num_input_features=num_features,
                kernel_size=kernel_size,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                conv_mode=conv_mode,
                concat_output=i != num_layers - 1,
                **kwargs,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)

            num_features = (
                num_features + block_depth * (exp_factor**i) * growth_rate
                if i != num_layers - 1
                else block_depth * (exp_factor**i) * growth_rate
            )
            self.out_channels.append(num_features)

            if i != num_layers - 1:
                trans = Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                )
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2
                self.in_channels.append(num_features)

    def forward(self, x):
        self.skip_connections = []
        for i, (name, module) in enumerate(self.features.items()):
            x = module(x)
            if name.startswith("denseblock") and module.concat_output:
                self.skip_connections.append(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        block_depth,
        growth_rate,
        drop_rate,
        conv_mode,
        **kwargs,
    ):
        super(DecoderBlock, self).__init__()

        self.up_sample = nn.ConvTranspose2d(
            in_channels,
            in_channels,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1,
            bias=False,
        )

        self.conv = DenseBlock(
            block_depth,
            in_channels + skip_channels,
            kernel_size=3,
            bn_size=2,
            growth_rate=growth_rate,
            drop_rate=drop_rate,
            conv_mode=conv_mode,
            concat_output=False,
            **kwargs,
        )

    def forward(self, x, skip):
        x = self.up_sample(x)
        if x.size()[2:] != skip.size()[2:]:
            x = torch.nn.functional.interpolate(
                x, size=skip.size()[2:], mode="bilinear", align_corners=True
            )
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        block_depth,
        growth_rate,
        exp_factor,
        drop_rate,
        conv_mode,
        **kwargs,
    ):
        """

        Parameters
        ----------
        in_channels: int
            The number of input channels.

        skip_channels: list[int]
            The number of channels in the skip connections.

        block_depth: int
            The number of layers in each dense block.

        growth_rate: int
            The growth rate of the dense blocks.

        drop_rate: float
            The dropout rate.

        conv_mode: str
            The convolutional mode ("full", "depthwise_separable").
        kwargs
        """
        super(Decoder, self).__init__()

        self.features = nn.ModuleDict()
        for i in range(len(skip_channels)):
            self.features.add_module(
                "decoderblock%d" % (i + 1),
                DecoderBlock(
                    in_channels // (exp_factor**i),
                    skip_channels[i],
                    block_depth * (exp_factor ** (len(skip_channels) - (i + 1))),
                    growth_rate,
                    drop_rate,
                    conv_mode=conv_mode,
                    **kwargs,
                ),
            )

    def forward(self, x, skip_connections):
        for i, (_, decoder_block) in enumerate(self.features.items()):
            x = decoder_block(x, skip_connections[i])
        return x
