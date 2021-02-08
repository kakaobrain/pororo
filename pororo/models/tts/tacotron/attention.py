import torch
from torch.nn import Conv1d, Linear, Parameter
from torch.nn import functional as F


class AttentionBase(torch.nn.Module):
    """Abstract attention class.

    Arguments:
        representation_dim -- size of the hidden representation
        query_dim -- size of the attention query input (probably decoder hidden state)
        memory_dim -- size of the attention memory input (probably encoder outputs)
    """

    def __init__(self, representation_dim, query_dim, memory_dim):
        super(AttentionBase, self).__init__()
        self._bias = Parameter(torch.zeros(1, representation_dim))
        self._energy = Linear(representation_dim, 1, bias=False)
        self._query = Linear(query_dim, representation_dim, bias=False)
        self._memory = Linear(memory_dim, representation_dim, bias=False)
        self._memory_dim = memory_dim

    def reset(self, encoded_input, batch_size, max_len, device):
        """Initialize previous attention weights & prepare attention memory."""
        self._memory_transform = self._memory(encoded_input)
        self._prev_weights = torch.zeros(batch_size, max_len, device=device)
        self._prev_context = torch.zeros(
            batch_size,
            self._memory_dim,
            device=device,
        )
        return self._prev_context

    def _attent(self, query, memory_transform, weights):
        raise NotImplementedError

    def _combine_weights(self, previsous_weights, weights):
        raise NotImplementedError

    def _normalize(self, energies, mask):
        raise NotImplementedError

    def forward(self, query, memory, mask, prev_decoder_output):
        energies = self._attent(
            query,
            self._memory_transform,
            self._prev_weights,
        )
        attention_weights = self._normalize(energies, mask)
        self._prev_weights = self._combine_weights(
            self._prev_weights,
            attention_weights,
        )
        attention_weights = attention_weights.unsqueeze(1)
        self._prev_context = torch.bmm(attention_weights, memory).squeeze(1)
        return self._prev_context, attention_weights.squeeze(1)


class LocationSensitiveAttention(AttentionBase):
    """
    Location Sensitive Attention:
        Location-sensitive attention: https://arxiv.org/abs/1506.07503.
        Extends additive attention (here https://arxiv.org/abs/1409.0473)
        to use cumulative attention weights from previous decoder time steps.

    Arguments:
        kernel_size -- kernel size of the convolution calculating location features
        channels -- number of channels of the convolution calculating location features
        smoothing -- to normalize weights using softmax, use False (default) and True to use sigmoids
    """

    def __init__(
        self,
        kernel_size,
        channels,
        smoothing,
        representation_dim,
        query_dim,
        memory_dim,
    ):
        super(LocationSensitiveAttention,
              self).__init__(representation_dim, query_dim, memory_dim)
        self._location = Linear(channels, representation_dim, bias=False)
        self._loc_features = Conv1d(
            1,
            channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self._smoothing = smoothing

    def _attent(self, query, memory_transform, cum_weights):
        query = self._query(query.unsqueeze(1))
        cum_weights = cum_weights.unsqueeze(-1)
        loc_features = self._loc_features(cum_weights.transpose(1, 2))
        loc_features = self._location(loc_features.transpose(1, 2))
        energy = query + memory_transform + loc_features
        energy = self._energy(torch.tanh(energy + self._bias))
        return energy.squeeze(-1)

    def _normalize(self, energies, mask):
        energies[~mask] = float("-inf")
        if self._smoothing:
            sigmoid = torch.sigmoid(energies)
            total = torch.sum(sigmoid, dim=-1)
            return sigmoid / total
        else:
            return F.softmax(energies, dim=1)

    def _combine_weights(self, previous_weights, weights):
        return previous_weights + weights
