import torch
import torch.nn as nn
from torch.nn import functional as F


class CrossMutilFeatureAttention(nn.Module):
    def __init__(self, input_dim1=12, input_dim2=12, num_class=7):
        super(CrossMutilFeatureAttention, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.num_class = num_class

        self.Wq = nn.Linear(input_dim1, num_class)
        self.Wk = nn.Linear(input_dim2, num_class)
        self.Wv = nn.Linear(input_dim1 + input_dim2, num_class)

    def forward(self, x, y):

        Q1 = self.Wq(x)
        Q2 = self.Wk(y)
        K = self.Wv(torch.cat((x, y), dim=-1))
        out1 = Q1 + (torch.mm(x, torch.transpose(self.Wv.weight[:, :self.input_dim1], 0, 1)) +
                     self.Wv.bias / 2)
        out2 = Q2 + (torch.mm(y, torch.transpose(self.Wv.weight[:, self.input_dim1:], 0, 1)) +
                     self.Wv.bias / 2)
        out = out1 + out2
        return out


class FiLM(nn.Module):

    def __init__(self, input_dim=12, dim=12, output_dim=7):
        super(FiLM, self).__init__()

        self.dim = input_dim
        self.fc_x = nn.Linear(input_dim, 2 * dim)
        self.fc_y = nn.Linear(input_dim, 2 * dim)
        self.fc_out = nn.Linear( 2 *dim, output_dim)

    def forward(self, x, y):

        gamma_x, beta_x = torch.split(self.fc_x(x), self.dim, 1)
        gamma_y, beta_y = torch.split(self.fc_y(y), self.dim, 1)

        x_new = gamma_y * x + beta_y
        y_new = gamma_x * y + beta_x

        output = torch.cat((x_new, y_new), dim=1)
        output = self.fc_out(output)

        return output
