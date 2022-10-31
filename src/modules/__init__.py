
from typing import Collection
import math
import torch
import torch.nn as nn


class Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.summary = {}

    def add_summary(self, name, val):
        if self.training:
            self.summary[name] = val.clone().detach().cpu().numpy()

    def get_summary(self, base_name=''):
        summary = {}
        if base_name:
            base_name += '/'
        if self.summary:
            summary.update({base_name + name: val for name, val in self.summary.items()})
        for name, child in self.named_children():
            if hasattr(child, 'get_summary'):
                name = base_name + name
                summary.update(child.get_summary(name))
        return summary


class ModuleList(nn.ModuleList):
    def get_summary(self, base_name=''):
        summary = {}
        if base_name:
            base_name += '/'
        for i, module in enumerate(self):
            if hasattr(module, 'get_summary'):
                name = base_name + str(i)
                summary.update(module.get_summary(name))
        return summary


class ModuleDict(nn.ModuleDict):
    def get_summary(self, base_name=''):
        summary = {}
        if base_name:
            base_name += '/'
        for key, module in self.items():
            if hasattr(module, 'get_summary'):
                name = base_name + key
                summary.update(module.get_summary(name))
        return summary


class GeLU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1. + torch.tanh(x * 0.7978845608 * (1. + 0.044715 * x * x)))


class Linear(nn.Module):
    def __init__(self, in_features, out_features, activations=False):
        super().__init__()
        linear = nn.Linear(in_features, out_features)
        nn.init.normal_(linear.weight, std=math.sqrt((2. if activations else 1.) / in_features))
        nn.init.zeros_(linear.bias)
        modules = [nn.utils.weight_norm(linear)]
        if activations:
            modules.append(GeLU())
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


