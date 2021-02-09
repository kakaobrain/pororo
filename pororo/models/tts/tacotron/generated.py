import torch
from torch.nn import Linear
from torch.nn import functional as F


class Conv1dGenerated(torch.nn.Module):
    """One dimensional convolution with generated weights (each group has separate weights).

    Arguments:
        embedding_dim -- size of the meta embedding (should be language embedding)
        bottleneck_dim -- size of the generating embedding
        see torch.nn.Conv1d
    """

    def __init__(
        self,
        embedding_dim,
        bottleneck_dim,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(Conv1dGenerated, self).__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        self._groups = groups

        # in_channels and out_channels is divisible by groups
        # tf.nn.functional.conv1d accepts weights of shape [out_channels, in_channels // groups, kernel]

        self._bottleneck = Linear(embedding_dim, bottleneck_dim)
        self._kernel = Linear(
            bottleneck_dim,
            out_channels // groups * in_channels // groups * kernel_size)
        self._bias = Linear(
            bottleneck_dim,
            out_channels // groups,
        ) if bias else None

    def forward(self, generator_embedding, x):

        assert (
            generator_embedding.shape[0] == self._groups
        ), "Number of groups of a convolutional layer must match the number of generators."

        e = self._bottleneck(generator_embedding)
        kernel = self._kernel(e).view(
            self._out_channels,
            self._in_channels // self._groups,
            self._kernel_size,
        )
        bias = self._bias(e).view(self._out_channels) if self._bias else None

        return F.conv1d(
            x,
            kernel,
            bias,
            self._stride,
            self._padding,
            self._dilation,
            self._groups,
        )


class BatchNorm1dGenerated(torch.nn.Module):
    """One dimensional batch normalization with generated weights (each group has separate parameters).

    Arguments:
        embedding_dim -- size of the meta embedding (should be language embedding)
        bottleneck_dim -- size of the generating embedding
        see torch.nn.BatchNorm1d
    Keyword arguments:
        groups -- number of groups with separate weights
    """

    def __init__(
        self,
        embedding_dim,
        bottleneck_dim,
        num_features,
        groups=1,
        eps=1e-8,
        momentum=0.1,
    ):
        super(BatchNorm1dGenerated, self).__init__()

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked",
                             torch.tensor(0, dtype=torch.long))

        self._num_features = num_features // groups
        self._eps = eps
        self._momentum = momentum
        self._groups = groups

        self._bottleneck = Linear(embedding_dim, bottleneck_dim)
        self._affine = Linear(
            bottleneck_dim,
            self._num_features + self._num_features,
        )

    def forward(self, generator_embedding, x):

        assert (
            generator_embedding.shape[0] == self._groups
        ), "Number of groups of a batchnorm layer must match the number of generators."

        if self._momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self._momentum

        e = self._bottleneck(generator_embedding)
        affine = self._affine(e)
        scale = affine[:, :self._num_features].contiguous().view(-1)
        bias = affine[:, self._num_features:].contiguous().view(-1)

        if self.training:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self._momentum is None:
                    exponential_average_factor = 1.0 / float(
                        self.num_batches_tracked)
                else:
                    exponential_average_factor = self._momentum

        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            scale,
            bias,
            self.training,
            exponential_average_factor,
            self._eps,
        )
