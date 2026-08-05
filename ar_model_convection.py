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

import os
import argparse
import random

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import LBFGS
from tqdm import tqdm

from util import *
from model_dict import get_model

import time


seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

parser = argparse.ArgumentParser('Training Point Optimization - Auto-Regressive 1D Convection')
parser.add_argument('--model', type=str, default='PINN')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--epochs', type=int, default=1000)
args = parser.parse_args()

model_aliases = {
    'pinn': 'PINN', 'PINN': 'PINN',
    'qres': 'QRes', 'QRes': 'QRes', 'qes': 'QRes', 'QES': 'QRes',
    'fls': 'FLS', 'FLS': 'FLS',
    'kan': 'KAN', 'KAN': 'KAN',
}
args.model = model_aliases.get(args.model, args.model)
supported_models = ['PINN', 'QRes', 'FLS', 'KAN']
if args.model not in supported_models:
    raise ValueError(f'Auto-regressive point optimization supports {supported_models}, got {args.model}.')

device = args.device
if device.startswith('cuda') and not torch.cuda.is_available():
    print(f'CUDA is not available, falling back from {device} to cpu.')
    device = 'cpu'

X_MIN, X_MAX = 0.0, 2.0 * np.pi
T_MIN, T_MAX = 0.0, 1.0
NX, NT = 101, 101
DT = (T_MAX - T_MIN) / (NT - 1)
BETA = 50.0


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)


def build_model():
    # AR input is [x, t_n, u_n].  Repository models do cat([x, t], -1), so
    # t is replaced by [t_n, u_n] here.
    if args.model == 'KAN':
        return get_model(args).Model(width=[3, 5, 5, 1], grid=5, k=3, grid_eps=1.0,
                                     noise_scale_base=0.25, device=device).to(device)
    if args.model == 'QRes':
        net = get_model(args).Model(in_dim=3, hidden_dim=256, out_dim=1, num_layer=4).to(device)
        net.apply(init_weights)
        return net

    net = get_model(args).Model(in_dim=3, hidden_dim=512, out_dim=1, num_layer=4).to(device)
    net.apply(init_weights)
    return net


def initial_condition_torch(x):
    return torch.sin(x)


def u_ana(x, t):
    return np.sin(x - BETA * t)


def rollout(model, x, detach_state=True):
    """Roll out u_{n+1}=N(x,t_n,u_n) from the exact initial state."""
    u = initial_condition_torch(x)
    pred_list = [u]

    for n in range(NT - 1):
        t_n = torch.full_like(x, T_MIN + n * DT)
        state = u.detach() if detach_state else u
        t_aug = torch.cat([t_n, state], dim=-1)
        u_next = model(x, t_aug)
        pred_list.append(u_next)
        u = u_next

    return pred_list


model = build_model()
optim = LBFGS(model.parameters(), line_search_fn='strong_wolfe')

print(model)
print(get_n_params(model))
loss_track = []

x_train = torch.linspace(X_MIN, X_MAX, NX, dtype=torch.float32, device=device).reshape(-1, 1)
x_train.requires_grad_(True)

time_start = time.time()

for i in tqdm(range(args.epochs)):
    def closure():
        loss_res = 0.0
        loss_bc = 0.0

        u = initial_condition_torch(x_train)

        for n in range(NT - 1):
            t_n = torch.full_like(x_train, T_MIN + n * DT)
            u_curr = u.detach()
            t_aug = torch.cat([t_n, u_curr], dim=-1)
            u_next = model(x_train, t_aug)

            u_x_next = torch.autograd.grad(
                u_next, x_train,
                grad_outputs=torch.ones_like(u_next),
                retain_graph=True,
                create_graph=True
            )[0]

            # Semi-implicit AR residual for u_t + 50u_x=0.
            residual = (u_next - u_curr) / DT + BETA * u_x_next
            loss_res = loss_res + torch.mean(residual ** 2)

            # Periodic boundary in x: u(0,t)=u(2*pi,t).
            loss_bc = loss_bc + torch.mean((u_next[0:1] - u_next[-1:]) ** 2)

            u = u_next

        loss_res = loss_res / (NT - 1)
        loss_bc = loss_bc / (NT - 1)
        loss_ic = torch.zeros((), dtype=torch.float32, device=device)

        loss_track.append([loss_res.item(), loss_bc.item(), loss_ic.item()])

        loss = loss_res + loss_bc + loss_ic
        optim.zero_grad()
        loss.backward()
        return loss

    optim.step(closure)

end_time = time.time() - time_start
print('Loss Res: {:4f}, Loss_BC: {:4f}, Loss_IC: {:4f}'.format(loss_track[-1][0], loss_track[-1][1], loss_track[-1][2]))
print('Train Loss: {:4f}'.format(np.sum(loss_track[-1])))
model_name = {args.model}
print(model_name, "time is ", end_time)


if not os.path.exists('./results/'):
    os.makedirs('./results/')

torch.save(model.state_dict(), f'./results/1dconvection_{args.model}_point_ar.pt')

# Visualize by true auto-regressive rollout instead of direct model(x,t) query.
x_test = torch.linspace(X_MIN, X_MAX, NX, dtype=torch.float32, device=device).reshape(-1, 1)

with torch.no_grad():
    pred_steps = rollout(model, x_test, detach_state=True)
    pred = torch.stack(pred_steps, dim=0).squeeze(-1).cpu().numpy()

res_test, _, _, _, _ = get_data([X_MIN, X_MAX], [T_MIN, T_MAX], NX, NT)
u = u_ana(res_test[:, 0], res_test[:, 1]).reshape(NT, NX)

rl1 = np.sum(np.abs(u - pred)) / np.sum(np.abs(u))
rl2 = np.sqrt(np.sum((u - pred) ** 2) / np.sum(u ** 2))

print('relative L1 error: {:4f}'.format(rl1))
print('relative L2 error: {:4f}'.format(rl2))

plt.figure(figsize=(4, 3))
plt.imshow(pred, aspect='equal')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Predicted u(x,t) - AR')
plt.colorbar()
plt.tight_layout()
plt.axis('off')
plt.savefig(f'./results/1dconvection_{args.model}_point_optimization_ar_pred.pdf', bbox_inches='tight')

plt.figure(figsize=(4, 3))
plt.imshow(u, aspect='equal')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Exact u(x,t)')
plt.colorbar()
plt.tight_layout()
plt.axis('off')
plt.savefig('./results/1dconvection_exact.pdf', bbox_inches='tight')

plt.figure(figsize=(4, 3))
plt.imshow(pred - u, aspect='equal', cmap='coolwarm', vmin=-1, vmax=1)
plt.xlabel('x')
plt.ylabel('t')
plt.title('Absolute Error')
plt.colorbar()
plt.tight_layout()
plt.axis('off')
plt.savefig(f'./results/1dconvection_{args.model}_point_optimization_ar_error.pdf', bbox_inches='tight')
