import numpy
import torch.nn as nn

from models.submodules import Encoder, Decoder, ConvNormAct


class BaseModel(nn.Module):
    """
    Base class for stem detection model from BONN.
    """

    def __init__(
        self,
        in_channels,
        num_init_features,
        out_channels,
        growth_rate,
        exp_factor,
        drop_rate,
        num_layers,
        block_depth,
        kernel_size,
        conv_mode,
        **kwargs
    ):
        assert conv_mode in ("full", "depthwise_separable"), "Invalid conv mode!"

        super().__init__()
        self.in_channels = in_channels
        self.num_init_features = num_init_features
        self.out_channels = out_channels
        self.growth_rate = growth_rate
        self.exp_factor = exp_factor
        self.drop_rate = drop_rate
        self.num_layers = num_layers
        self.block_depth = block_depth
        self.kernel_size = kernel_size
        self.conv_mode = conv_mode

        # other args
        self.kwargs = kwargs

        self.init_features = self.build_init_features()
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.head = self.build_head()

    def __str__(self):
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([numpy.prod(p.size()) for p in model_params])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

    def build_init_features(self):
        return ConvNormAct(
            in_channels=self.in_channels,
            out_channels=self.num_init_features,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            bias=False,
        )

    def build_encoder(self):
        return Encoder(
            num_layers=self.num_layers + 1,
            block_depth=self.block_depth,
            num_init_features=self.num_init_features,
            kernel_size=self.kernel_size,
            bn_size=2,
            growth_rate=self.growth_rate,
            exp_factor=self.exp_factor,
            drop_rate=self.drop_rate,
            conv_mode=self.conv_mode,
            **self.kwargs
        )

    def build_decoder(self):
        return Decoder(
            in_channels=self.encoder.out_channels[-1],
            skip_channels=self.encoder.out_channels[::-1][1:],
            block_depth=self.block_depth,
            growth_rate=self.growth_rate,
            exp_factor=self.exp_factor,
            drop_rate=self.drop_rate,
            conv_mode=self.conv_mode,
            **self.kwargs
        )

    #
    def build_head(self):
        return nn.Sequential(
            ConvNormAct(
                in_channels=self.encoder.out_channels[-1]
                // (self.exp_factor**self.num_layers),
                out_channels=self.num_init_features,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                bias=False,
                conv_mode=self.conv_mode,
            ),
            nn.Conv2d(
                in_channels=self.num_init_features,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.init_features(x)
        x = self.encoder(x)
        x = self.decoder(x, self.encoder.skip_connections[::-1])
        x = self.head(x)
        return x
