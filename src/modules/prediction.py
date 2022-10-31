
import torch
import torch.nn as nn
from functools import partial
from src.utils.registry import register
from . import Linear

registry = {}
register = partial(register, registry=registry)

class Prediction_Bert(nn.Module):
    def __init__(self, args,hidden_size):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Dropout(args.dropout),
            Linear(hidden_size, hidden_size, activations=True),
            #Linear(args.hidden_size * args.order, args.hidden_size, activations=True),
            #Linear(args.hidden_size * (args.order + 1), args.hidden_size* args.order, activations=True),
            nn.Dropout(args.dropout),
            Linear(hidden_size, args.num_classes),
        )

    def forward(self, x):
        return self.dense(x)

class Prediction_Bert_GAT(nn.Module):
    def __init__(self, args):
        super().__init__()
        print("============")
        print(args.num_classes)
        self.dense = nn.Sequential(
            nn.Dropout(args.dropout),
            Linear(args.hidden_size * 4, args.hidden_size, activations=True),
            nn.Dropout(args.dropout),
            Linear(args.hidden_size, args.num_classes),
        )


    def forward(self, a,b,res):
        #fusion = torch.cat([a, b, (a - b).abs(), a * b], dim=-1) + res
        fusion = torch.cat([a, b, a - b, a * b], dim=-1) + res

        return self.dense(fusion)





