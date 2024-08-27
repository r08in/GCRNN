import torch.nn as nn
import torch
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

class ParallelMLPLayer(nn.Module):
    def __init__(self, in_features, out_features, n_para, bias=True, activation=None):
        super(ParallelMLPLayer, self).__init__()
        modules = []
        self.n_para = n_para
        self.in_features = in_features
        for _ in range(n_para):
            layer = MLPLayer(in_features, out_features, activation=activation, bias=bias) 
            modules.append(layer)
        self.layers = nn.ModuleList(modules)

    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.layers):
            start_idx = i * self.in_features
            end_idx = (i + 1) * self.in_features
            outputs.append(layer(x[:, start_idx:end_idx]))
        return torch.cat(outputs, dim=1)
    

class MLPModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims,  activation='relu', flatten=True, output_bias=False):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = []
        elif type(hidden_dims) is int:
            hidden_dims = [hidden_dims]
        nr_hiddens = len(hidden_dims)


        n_para=None
        if output_dim>1 and nr_hiddens>0:
            n_para=output_dim
            output_dim=1

        dims = [input_dim]
        dims.extend(hidden_dims)
        dims.append(output_dim)
        modules = []

        
        if n_para is None: 
            for i in range(nr_hiddens):
                layer = MLPLayer(dims[i], dims[i+1],  activation=activation)
                modules.append(layer)
            layer = nn.Linear(dims[-2], dims[-1], bias=output_bias)
        else:
            for i in range(nr_hiddens):
                if i==0:
                    layer = MLPLayer(dims[i], dims[i+1]*n_para,  activation=activation)
                else:
                    layer = ParallelMLPLayer(dims[i], dims[i+1], n_para=n_para, activation=activation)
                modules.append(layer)
            layer =  ParallelMLPLayer(dims[-2], dims[-1], n_para=n_para, bias=output_bias)
        modules.append(layer)
        self.mlp = nn.Sequential(*modules)
        self.flatten = flatten

    def forward(self, input):
        if self.flatten:
            input = input.view(input.size(0), -1)
        return self.mlp(input)

    
    def get_input_layer(self):
        if hasattr(self.mlp[0], "__getitem__"):
            return self.mlp[0][0]
        else:
            return self.mlp[0]
        
    def set_input_layer(self, new_input_layer):
        if hasattr(self.mlp[0], "__getitem__"):
            self.mlp[0][0] = new_input_layer
        else:
            self.mlp[0] = new_input_layer