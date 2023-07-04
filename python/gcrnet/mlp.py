import torch.nn as nn
from gcrnet.utils import get_activation

__all__ = [
    'MLPLayer', 'MLPModel'
]

class MLPLayer(nn.Sequential):
    def __init__(self, in_features, out_features, bias=True, activation=None):
        modules = [nn.Linear(in_features, out_features, bias=bias)]
        if activation is not None and activation is not False:
            modules.append(get_activation(activation))
        super().__init__(*modules)


class MLPModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims,  activation='relu', flatten=True, output_bias=False):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = []
        elif type(hidden_dims) is int:
            hidden_dims = [hidden_dims]

        dims = [input_dim]
        dims.extend(hidden_dims)
        dims.append(output_dim)
        modules = []

        nr_hiddens = len(hidden_dims)
        for i in range(nr_hiddens):
            layer = MLPLayer(dims[i], dims[i+1],  activation=activation)
            modules.append(layer)
        layer = nn.Linear(dims[-2], dims[-1], bias=output_bias)
        modules.append(layer)
        self.mlp = nn.Sequential(*modules)
        self.flatten = flatten


    def forward(self, input):
        if self.flatten:
            input = input.view(input.size(0), -1)
        return self.mlp(input)