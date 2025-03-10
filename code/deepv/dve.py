"""dve.py
Barebones DVE.
"""
import logging
logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
import numpy as np
from .hook import Hook, HookOutput
from collections import defaultdict



cln_key = lambda x: x.replace('.', '_')



class Standardizer(nn.Module):
    def __init__(self,x):
        super(Standardizer,self).__init__()
        d = x.shape[-1]
        self.register_buffer("xmean",torch.zeros(1,d))
        self.register_buffer("xstd",torch.ones(1,d))

        self.mean = x.mean(0,keepdim = True)
        xstd = x.std(0, keepdim = True)
        xstd[xstd == 0.0] = 1.0
        self.xtsd = xstd

    def forward(self,x):
        return (x - self.xmean).div(self.xstd)


class DeepV(nn.Module):
    def __init__(self, 
            net, 
            loader,
            moments,
            layers,
            init_fn,
            train_fn,
            build_args,
            **kwargs
        ):
        super(DeepV,self).__init__()
        self.net = Hook(net, output_layers_specification=layers)
        self.oracle = None
        self.kwargs = kwargs
        self.init_fn = init_fn
        self.train_fn = train_fn
        self.build_args = build_args
        self.moments = moments

        self.vecchia_gps = self.gp_init(loader, build_args)
        

    def gp_init(self, train_loader, build_args):
        e_train, y_train = get_embedded_data(train_loader, self.net)
        gps = {}
        if build_args is None:
            build_args = self.build_args
        for layer_key, e in e_train.items():
            e = self.moments[cln_key(layer_key)](e)
            gps[cln_key(layer_key)],_  = self.init_fn(
                e, y_train.squeeze(), **build_args
            )

        gps = nn.ModuleDict(gps)
        
        return gps


    def fit(self, train_loader, val_loader, build_args=None, **kwargs):
        e_val, y_val = get_embedded_data(val_loader, self.net)

        if build_args is None:
            build_args = self.build_args

        for layer_key, e_val_ in e_val.items():
            gp = self.vecchia_gps[cln_key(layer_key)]
            e_val_ = self.moments[cln_key(layer_key)](e_val_)
            self.vecchia_gps[cln_key(layer_key)] = self.train_fn(
                gp, gp.likelihood, e_val_, y_val, **build_args
            )


    def forward(self, x):
        output = self.net(x)
        preds = {}
        for k,e in output.intermediates.items():
            e = self.moments[cln_key(k)](e)
            self.vecchia_gps[cln_key(k)].eval()
            preds[k] = self.vecchia_gps[cln_key(k)](e)
        return preds

    @property
    def prior_var(self):
        p_var = []
        for k,v in self.vecchia_gps.items():
            p_var.append(v.covar_module.outputscale.detach())
        p_var = torch.Tensor(p_var)

        return p_var

    @property
    def noise(self):
        noise = []
        for k,v in self.vecchia_gps.items():
            noise.append(v.likelihood.noise.detach())

        noise = torch.Tensor(noise)
        return noise

def fit_standardizer(loader, net, layers):
    net = Hook(net, output_layers_specification=layers)
    net.eval()
    moments = {}
    E = defaultdict(list)
    for x, y in loader:
        with torch.no_grad():
            e = net(x)
        for k,v in e.intermediates.items():
            E[k].append(v.cpu())

    for k,v in E.items():
        moments[cln_key(k)] = Standardizer(torch.cat(E[k], axis = 0))

    moments = nn.ModuleDict(moments)
    return moments

def get_embedded_data(loader, net):
    net.eval()
    E = defaultdict(list)
    Y = []
    for x, y in loader:
        Y.append(y)
        with torch.no_grad():
            e = net(x)
        for k,v in e.intermediates.items():
            E[k].append(v.cpu())
    Y = torch.cat(Y, axis = 0).squeeze()
    for k,v in E.items():
        E[k] = torch.cat(E[k], axis = 0)

    return E, Y


