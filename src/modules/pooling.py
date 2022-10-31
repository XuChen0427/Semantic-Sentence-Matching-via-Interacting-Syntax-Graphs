

import torch.nn as nn


class Pooling(nn.Module):
    def forward(self, x, mask):
        return x.masked_fill_(~mask, -float('inf')).max(dim=1)[0]
