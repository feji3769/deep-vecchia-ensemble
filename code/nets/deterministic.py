import torch.nn as nn
import torch
import numpy as np
from .spectral_normalization import spectral_norm_fc

class Deterministic(nn.Module):
    def __init__(self,
                input_dim,
                output_dim, 
                hidden_dims = None, 
                sn=False, 
                coeff=2.0,
                seed=42,
                **kwargs
        ):
        super(Deterministic, self).__init__()

        modules = []
        in_dim = input_dim
        i = 0
        for i, h_dim in enumerate(hidden_dims):
            linear = nn.Linear(in_dim, h_dim)
            if sn:
                linear=spectral_norm_fc(linear,coeff,n_power_iterations = 5)
            modules.append(
                nn.Sequential(
                    linear,
                    nn.ReLU(),
                )
            )
            in_dim = h_dim
        self.feature_dim = in_dim
        self.head = nn.Sequential(*modules)
        self.final_layer = nn.Linear(in_dim, output_dim)
        seed=np.random.randint(low=1,high=1000)
        self.initialize_model(seed)

    def initialize_model(self, seed=42):
        g = torch.Generator()
        g.manual_seed(seed)
        self._initialize_weights(self, g)


    def _initialize_weights(self,layers, g):
        for m in layers.modules():
            if isinstance(m, nn.Linear):
                stdv = 1. / np.sqrt(m.weight.size(1))
                nn.init.uniform_(m.weight, a=-stdv, b=stdv, generator=g)
                if m.bias is not None:
                    nn.init.uniform_(m.bias, a=-stdv, b=stdv, generator=g)
                    

            

    def forward(self, x):
        e = self.head(x)
        pred = self.final_layer(e)
        return pred


