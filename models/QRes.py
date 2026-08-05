'''
Copyright (c) 2026 Beihang University

Developed by:
He Zichuan

Project:
ARKAN: Autoregressive Kolmogorov-Arnold Network for RTM Wavefield Prediction


Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, subject to the following conditions:


The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.


THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
'''
import torch
import torch.nn as nn

from util import get_clones


class QRes_block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(QRes_block, self).__init__()
        self.H1 = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.H2 = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x1 = self.H1(x)
        x2 = self.H2(x)
        return self.act(x1 * x2 + x1)


class Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer):
        super(Model, self).__init__()
        self.N = num_layer - 1
        self.inlayer = QRes_block(in_dim, hidden_dim)
        self.layers = get_clones(QRes_block(hidden_dim, hidden_dim), num_layer - 1)
        self.outlayer = nn.Linear(in_features=hidden_dim, out_features=out_dim)

    def forward(self, x, t):
        src = torch.cat((x, t), dim=-1)
        src = self.inlayer(src)
        for i in range(self.N):
            src = self.layers[i](src)
        src = self.outlayer(src)
        return src
