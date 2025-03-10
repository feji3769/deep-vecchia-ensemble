"""hook.py
Hooks for intermediate layers.
"""
import torch.nn as nn
import torch
from dataclasses import dataclass
from collections import OrderedDict

@dataclass
class HookOutput:
    net_pred: torch.Tensor
    intermediates: dict

class Hook(nn.Module):
    def __init__(self, base_model: nn.Module, output_layers_specification):
        # Init and store base model
        super().__init__()
        self.base_model = base_model

        # Output hooks
        self.output_layers = []
        self.fwd_hooks = []
        self._module_to_layer_name = {}  # Mapping of modules to layername

        # Storage for last fwd pass hooks
        self.hook_out = OrderedDict()

        # Register hooks
        self.register_hooks(output_layers_specification)

    def register_hooks(self, output_layers_specification,
                       reset=True):
        """Register the forward hooks for the network.

        :param output_layers_specification:
        :param reset: Whether to remove old hooks or just add new ones. Default: True
        :returns: self.output_layers, module names of all registered hooks.
        """
        if reset:
            self.output_layers = []
            self.fwd_hooks = []
            self._module_to_layer_name = {}  # Mapping of modules to layername

        for module_name, module in filter_all_named_modules(self.base_model, output_layers_specification):
            self._module_to_layer_name[module] = module_name
            self.fwd_hooks.append(
                module.register_forward_hook(self.hook)
            )
            self.output_layers.append(module_name)

        return self.output_layers

    def hook(self, module, inputs, outputs):
        layer_name = self._module_to_layer_name[module]
        assert type(inputs) is tuple and len(inputs) == 1,\
            f"Expected input to be a tuple with length 1, got {inputs}."
        self.hook_out[layer_name] = outputs

    def forward(self, x):
        out = self.base_model(x)
        return HookOutput(out, self.hook_out)


def filter_all_named_modules(model: nn.Module, layer_names,require_leaf: bool = None):
    if not layer_names:
        return
    yield_all = layer_names is True
    if require_leaf is None:
        require_leaf = yield_all

    # TODO(marius): Include check for if layers specifications are never contained in model
    for name, module, in model.named_modules():
        # Don't include if the module has children and we only want leaf modules.
        if require_leaf and len(list(module.children())) != 0:
            continue

        if yield_all:
            yield name, module
        elif sum(map(  # Check if the condition holds for one of the name specifications:
                lambda key: ("^" + name).endswith(str(key)),
                layer_names
        )):
            yield name, module