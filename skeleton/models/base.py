import numpy
import torch.nn as nn

from models.submodules import *


class BaseModel(nn.Module):
    '''
    Base class for UNet-like architectures.
    '''

    encoder_type = DownBlock
    residual_type = ResidualBlock
    decoder_type = UpBlock
    predictor_type = PredictBlock

    def __init__(
        self,
        in_channels,
        out_channels,
        exp_factor,
        num_layers,
        num_residuals,
        kernel_size,
        stride,
        embed_dim,
        up_mode,
        down_mode,
        conv_mode,
        skip_type,
        **kwargs
    ):
        assert up_mode in ('bilinear', 'nearest', 'conv'), 'Invalid up mode!'
        assert down_mode in ('pool', 'conv'), 'Invalid down mode!'
        assert conv_mode in ('full', 'depthwise_separable'), 'Invalid conv mode!'
        assert skip_type in ('add', 'cat'), 'Invalid skip mode!'
        
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp_factor = exp_factor
        self.num_layers = num_layers
        self.num_residuals = num_residuals
        self.kernel_size = kernel_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.up_mode = up_mode
        self.down_mode = down_mode
        self.conv_mode = conv_mode
        self.skip_type = skip_type
        
        # other args
        self.kwargs = kwargs
        
        # create input and output channel dimensions
        self.encoder_in_channels = [
            int(self.embed_dim*pow(self.exp_factor, i))
            for i in range(self.num_layers)
        ]
        self.encoder_out_channels = [
            int(self.embed_dim*pow(self.exp_factor, i + 1))
            for i in range(self.num_layers)
        ]
        
        self.max_num_channels = self.encoder_out_channels[-1]
        self.decoder_in_channels = self.encoder_out_channels[::-1]
        self.decoder_out_channels = self.encoder_in_channels[::-1]


    def __str__(self):
        model_params = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([numpy.prod(p.size()) for p in model_params])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
    

    def build_head(self):
        head = ConvActNorm(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.kernel_size,
            stride=1,
            conv_mode=self.conv_mode
        )
        return head


    def build_encoders(self):
        encoders = nn.ModuleList([
            self.encoder_type(
                in_channels,
                out_channels,
                self.kernel_size,
                self.stride,
                self.down_mode,
                self.conv_mode,
                **self.kwargs
            )
            for (in_channels, out_channels) in \
                zip(self.encoder_in_channels, self.encoder_out_channels)
        ])
        return encoders
    

    def build_residuals(self):
        residuals = nn.ModuleList([
            self.residual_type(
                self.max_num_channels,
                self.max_num_channels,
                self.kernel_size,
                self.conv_mode
            )
            for _ in range(self.num_residuals)
        ])
        return residuals
    

    def build_decoders(self):
        decoders = nn.ModuleList()
        for i, (in_channels, out_channels) in \
            enumerate(zip(self.decoder_in_channels, self.decoder_out_channels)):
            pred_channels = 0 if i == 0 else self.out_channels
            if self.skip_type == 'cat':
                decoders.append(
                    self.decoder_type(
                        2*in_channels + pred_channels,
                        out_channels,
                        self.kernel_size,
                        self.stride,
                        self.up_mode,
                        self.conv_mode
                    ))
            else:
                decoders.append(
                    self.decoder_type(
                        in_channels + pred_channels,
                        out_channels,
                        self.kernel_size,
                        self.stride,
                        self.up_mode,
                        self.conv_mode
                    ))
        return decoders
    

    def build_predictors(self):
        predictors = nn.ModuleList([
            self.predictor_type(
                out_channels,
                self.out_channels,
                self.kernel_size,
                self.conv_mode
            )
            for out_channels in self.decoder_out_channels
        ])
        return predictors
    

    def forward(self, x):
        error = 'Forward function not implemented!'
        raise NotImplementedError(error)