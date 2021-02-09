import torch
from torch.nn import (
    BatchNorm1d,
    ConstantPad1d,
    Conv1d,
    Dropout,
    Identity,
    ReLU,
    Sequential,
    Sigmoid,
    Tanh,
)
from torch.nn import functional as F

from pororo.models.tts.tacotron.generated import (
    BatchNorm1dGenerated,
    Conv1dGenerated,
)


def get_activation(name):
    """Get activation function by name."""
    return {
        "relu": ReLU(),
        "sigmoid": Sigmoid(),
        "tanh": Tanh(),
        "identity": Identity(),
    }[name]


class ZoneoutLSTMCell(torch.nn.LSTMCell):
    """Wrapper around LSTM cell providing zoneout regularization."""

    def __init__(
        self,
        input_size,
        hidden_size,
        zoneout_rate_hidden,
        zoneout_rate_cell,
        bias=True,
    ):
        super(ZoneoutLSTMCell, self).__init__(input_size, hidden_size, bias)
        self.zoneout_c = zoneout_rate_cell
        self.zoneout_h = zoneout_rate_hidden

    def forward(self, cell_input, h, c):
        new_h, new_c = super(ZoneoutLSTMCell, self).forward(cell_input, (h, c))
        if self.training:
            new_h = (1 - self.zoneout_h) * F.dropout(
                new_h - h,
                self.zoneout_h,
            ) + h
            new_c = (1 - self.zoneout_c) * F.dropout(
                new_c - c,
                self.zoneout_c,
            ) + c
        else:
            new_h = self.zoneout_h * h + (1 - self.zoneout_h) * new_h
            new_c = self.zoneout_c * c + (1 - self.zoneout_c) * new_c
        return new_h, new_c


class DropoutLSTMCell(torch.nn.LSTMCell):
    """Wrapper around LSTM cell providing hidden state dropout regularization."""

    def __init__(self, input_size, hidden_size, dropout_rate, bias=True):
        super(DropoutLSTMCell, self).__init__(input_size, hidden_size, bias)
        self._dropout = Dropout(dropout_rate)

    def forward(self, cell_input, h, c):
        new_h, new_c = super(DropoutLSTMCell, self).forward(cell_input, (h, c))
        new_h = self._dropout(new_h)
        return new_h, new_c


class ConvBlock(torch.nn.Module):
    """
    One dimensional convolution with batchnorm and dropout, expected channel-first input.

    Arguments:
        input_channels -- number if input channels
        output_channels -- number of output channels
        kernel -- convolution kernel size ('same' padding is used)
    Keyword arguments:
        dropout (default: 0.0) -- dropout rate to be aplied after the block
        activation (default 'identity') -- name of the activation function applied after batchnorm
        dilation (default: 1) -- dilation of the inner convolution
        groups (default: 1) -- number of groups of the inner convolution
        batch_norm (default: True) -- set False to disable batch normalization
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        kernel,
        dropout=0.0,
        activation="identity",
        dilation=1,
        groups=1,
        batch_norm=True,
    ):
        super(ConvBlock, self).__init__()

        self._groups = groups

        p = (kernel - 1) * dilation // 2
        padding = p if kernel % 2 != 0 else (p, p + 1)
        layers = [
            ConstantPad1d(padding, 0.0),
            Conv1d(
                input_channels,
                output_channels,
                kernel,
                padding=0,
                dilation=dilation,
                groups=groups,
                bias=(not batch_norm),
            ),
        ]

        if batch_norm:
            layers += [BatchNorm1d(output_channels)]

        layers += [get_activation(activation)]
        layers += [Dropout(dropout)]

        self._block = Sequential(*layers)

    def forward(self, x):
        return self._block(x)


class ConvBlockGenerated(torch.nn.Module):
    """One dimensional convolution with generated weights and with batchnorm and dropout, expected channel-first input.

    Arguments:
        embedding_dim -- size of the meta embedding
        bottleneck_dim -- size of the generating layer
        input_channels -- number if input channels
        output_channels -- number of output channels
        kernel -- convolution kernel size ('same' padding is used)
    Keyword arguments:
        dropout (default: 0.0) -- dropout rate to be aplied after the block
        activation (default 'identity') -- name of the activation function applied after batchnorm
        dilation (default: 1) -- dilation of the inner convolution
        groups (default: 1) -- number of groups of the inner convolution
        batch_norm (default: True) -- set False to disable batch normalization
    """

    def __init__(
        self,
        embedding_dim,
        bottleneck_dim,
        input_channels,
        output_channels,
        kernel,
        dropout=0.0,
        activation="identity",
        dilation=1,
        groups=1,
        batch_norm=True,
    ):
        super(ConvBlockGenerated, self).__init__()

        self._groups = groups

        p = (kernel - 1) * dilation // 2
        padding = p if kernel % 2 != 0 else (p, p + 1)

        self._padding = ConstantPad1d(padding, 0.0)
        self._convolution = Conv1dGenerated(
            embedding_dim,
            bottleneck_dim,
            input_channels,
            output_channels,
            kernel,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=(not batch_norm),
        )
        self._regularizer = (BatchNorm1dGenerated(
            embedding_dim,
            bottleneck_dim,
            output_channels,
            groups=groups,
        ) if batch_norm else None)
        self._activation = Sequential(
            get_activation(activation),
            Dropout(dropout),
        )

    def forward(self, x):
        e, x = x
        x = self._padding(x)
        x = self._convolution(e, x)
        if self._regularizer is not None:
            x = self._regularizer(e, x)
        x = self._activation(x)
        return e, x


class HighwayConvBlock(ConvBlock):
    """Gated 1D covolution.

    Arguments:
        see ConvBlock
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        kernel,
        dropout=0.0,
        activation="identity",
        dilation=1,
        groups=1,
        batch_norm=True,
    ):
        super(HighwayConvBlock, self).__init__(
            input_channels,
            2 * output_channels,
            kernel,
            dropout,
            activation,
            dilation,
            groups,
            batch_norm,
        )
        self._gate = Sigmoid()

    def forward(self, x):
        h = super(HighwayConvBlock, self).forward(x)
        chunks = torch.chunk(h, 2 * self._groups, 1)
        h1 = torch.cat(chunks[0::2], 1)
        h2 = torch.cat(chunks[1::2], 1)
        p = self._gate(h1)
        return h2 * p + x * (1.0 - p)


class HighwayConvBlockGenerated(ConvBlockGenerated):
    """Gated 1D covolution with generated weights.

    Arguments:
        embedding_dim -- size of the meta embedding
        bottleneck_dim -- size of the generating layer
        see ConvBlockGenerated
    """

    def __init__(
        self,
        embedding_dim,
        bottleneck_dim,
        input_channels,
        output_channels,
        kernel,
        dropout=0.0,
        activation="identity",
        dilation=1,
        groups=1,
        batch_norm=True,
    ):
        super(HighwayConvBlockGenerated, self).__init__(
            embedding_dim,
            bottleneck_dim,
            input_channels,
            2 * output_channels,
            kernel,
            dropout,
            activation,
            dilation,
            groups,
            batch_norm,
        )
        self._gate = Sigmoid()

    def forward(self, x):
        e, x = x
        _, h = super(HighwayConvBlockGenerated, self).forward((e, x))
        chunks = torch.chunk(h, 2 * self._groups, 1)
        h1 = torch.cat(chunks[0::2], 1)
        h2 = torch.cat(chunks[1::2], 1)
        p = self._gate(h1)
        return e, h2 * p + x * (1.0 - p)
