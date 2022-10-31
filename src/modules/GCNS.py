import torch
import torch.nn as nn
import torch_geometric as tg
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import init



class RGAT(torch.nn.Module):
    def __init__(self, input_dim, output_dim,
                 feature_pre=True,  dropout=True, dropout_rate=0.2,**kwargs):
        super(RGAT, self).__init__()
        #self.feature_pre = feature_pre
        #self.middle_dim = hidden_dim*4
        #self.layer_num = layer_num
        self.dropout = dropout
        #self.LayerNorm = nn.LayerNorm(hidden_dim,eps=1e-12)
        self.dropout_rate = dropout_rate

            #self.conv_first = tg.nn.GATConv(hidden_dim, hidden_dim)

        self.conv_hidden = tg.nn.GATConv(input_dim, output_dim,dropout=dropout_rate)
        #self.conv_res = Output(hidden_dim, self.middle_dim,dropout_rate=dropout_rate)
        #self.conv_out = tg.nn.GATConv(hidden_dim, hidden_dim)
        #self.out = nn.Linear(hidden_dim,output_dim)

    def forward(self, x, edge_index):
        #x, edge_index = data.x, data.edge_index
        # if self.feature_pre:
        #     x = self.linear_pre(x)
        #     x = F.relu(x)
        # if self.dropout:
        #     x = F.dropout(x, p=self.dropout_rate,training=self.training)
        #
        # pre_x = x

        x = self.conv_hidden(x, edge_index)
        # if self.dropout:
        #     x = F.dropout(x, p=self.dropout_rate,training=self.training)
        #
        # x = F.gelu(x)
        # x = self.LayerNorm(pre_x+x)
        #
        #
        # x = self.conv_res(x)
        # if self.dropout:
        #     x = F.dropout(x, p=self.dropout_rate,training=self.training)
        #
        # x = self.out(x)
        # if self.dropout:
        #     F.dropout(x, p=self.dropout_rate,training=self.training)
        #x = F.relu(x)

        #x = F.normalize(x, p=2, dim=-1)
        return x

class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, dropout_rate=0.2,**kwargs):
        super(GAT, self).__init__()
        self.feature_pre = feature_pre
        self.middle_dim = hidden_dim*4
        self.layer_num = layer_num
        self.dropout = dropout
        self.LayerNorm = nn.LayerNorm(hidden_dim,eps=1e-12)
        self.dropout_rate = dropout_rate
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, hidden_dim)
            #self.conv_first = tg.nn.GATConv(hidden_dim, hidden_dim)

        self.conv_hidden = nn.ModuleList([tg.nn.GATConv(hidden_dim, hidden_dim,dropout=dropout_rate) for i in range(layer_num)])
        self.conv_res = nn.ModuleList([Output(hidden_dim, self.middle_dim,dropout_rate=dropout_rate) for i in range(layer_num)])
        #self.conv_out = tg.nn.GATConv(hidden_dim, hidden_dim)

        self.out = nn.Linear(hidden_dim,output_dim)



    def forward(self, x, edge_index):
        #x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
            x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, p=self.dropout_rate,training=self.training)

        pre_x = x
        for i in range(self.layer_num):
            x = self.conv_hidden[i](x, edge_index)
            if self.dropout:
                x = F.dropout(x, p=self.dropout_rate,training=self.training)

            x = F.gelu(x)
            x = self.LayerNorm(pre_x+x)


            x = self.conv_res[i](x)
            if self.dropout:
                x = F.dropout(x, p=self.dropout_rate,training=self.training)

        x = self.out(x)
        if self.dropout:
            F.dropout(x, p=self.dropout_rate,training=self.training)
        x = F.relu(x)

        #x = F.normalize(x, p=2, dim=-1)
        return x


class Output(nn.Module):
    def __init__(self, hidden_size,middle_size,dropout_rate=0.2):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.dense_mid = nn.Linear(hidden_size, middle_size)

        self.LayerNorm = nn.LayerNorm(hidden_size,eps=1e-12)
        self.dense_out = nn.Linear(middle_size, hidden_size)

    def forward(self, hidden_states):
        mid_states = self.dense_mid(hidden_states)
        mid_states = F.dropout(mid_states,p=self.dropout_rate, training=self.training)
        mid_states  = F.gelu(mid_states)
        mid_states = self.dense_out(mid_states)
        hidden_states = self.LayerNorm(hidden_states + mid_states)
        return hidden_states