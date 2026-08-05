
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
import time
from model_dict import get_model


seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

parser = argparse.ArgumentParser('Training Point Optimization - Auto-Regressive 1D Wave')
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

X_MIN, X_MAX = 0.0, 1.0
T_MIN, T_MAX = 0.0, 1.0
NX, NT = 101, 101
DT = (T_MAX - T_MIN) / (NT - 1)
C_SQUARED = 4.0


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def build_model():
    # Wave equation is second order in time.  The AR state is [u_n, v_n],
    # where v=u_t.  The model predicts [u_{n+1}, v_{n+1}] from
    # [x, t_n, u_n, v_n].
    if args.model == 'KAN':
        return get_model(args).Model(width=[4, 5, 5, 2], grid=5, k=3, grid_eps=1.0,
                                     noise_scale_base=0.25, device=device).to(device)
    if args.model == 'QRes':
        net = get_model(args).Model(in_dim=4, hidden_dim=256, out_dim=2, num_layer=4).to(device)
        net.apply(init_weights)
        return net

    net = get_model(args).Model(in_dim=4, hidden_dim=512, out_dim=2, num_layer=4).to(device)
    net.apply(init_weights)
    return net


def initial_u_torch(x):
    pi = torch.tensor(np.pi, dtype=torch.float32, device=x.device)
    return torch.sin(pi * x) + 0.5 * torch.sin(3.0 * pi * x)


def initial_v_torch(x):
    return torch.zeros_like(x)


def u_ana(x, t):
    return np.sin(np.pi * x) * np.cos(2.0 * np.pi * t) + 0.5 * np.sin(3.0 * np.pi * x) * np.cos(6.0 * np.pi * t)


def rollout(model, x, detach_state=True):
    """Roll out [u_{n+1},v_{n+1}]=N(x,t_n,u_n,v_n)."""
    u = initial_u_torch(x)
    v = initial_v_torch(x)
    pred_list = [u]

    for n in range(NT - 1):
        t_n = torch.full_like(x, T_MIN + n * DT)
        u_state = u.detach() if detach_state else u
        v_state = v.detach() if detach_state else v
        t_aug = torch.cat([t_n, u_state, v_state], dim=-1)
        out = model(x, t_aug)
        u_next = out[:, 0:1]
        v_next = out[:, 1:2]
        pred_list.append(u_next)
        u = u_next
        v = v_next

    return pred_list


model = build_model()
optim = LBFGS(model.parameters(), line_search_fn='strong_wolfe')

print(model)
print(get_n_params(model))
loss_track = []

x_train = torch.linspace(X_MIN, X_MAX, NX, dtype=torch.float32, device=device).reshape(-1, 1)
x_train.requires_grad_(True)

start = time.time()

for i in tqdm(range(args.epochs)):
    def closure():
        loss_res = 0.0
        loss_bc = 0.0

        u = initial_u_torch(x_train)
        v = initial_v_torch(x_train)

        for n in range(NT - 1):
            t_n = torch.full_like(x_train, T_MIN + n * DT)
            u_curr = u.detach()
            v_curr = v.detach()
            t_aug = torch.cat([t_n, u_curr, v_curr], dim=-1)
            out = model(x_train, t_aug)
            u_next = out[:, 0:1]
            v_next = out[:, 1:2]

            u_x_next = torch.autograd.grad(
                u_next, x_train,
                grad_outputs=torch.ones_like(u_next),
                retain_graph=True,
                create_graph=True
            )[0]
            u_xx_next = torch.autograd.grad(
                u_x_next, x_train,
                grad_outputs=torch.ones_like(u_x_next),
                retain_graph=True,
                create_graph=True
            )[0]

            # First-order AR system:
            #   u_t = v,
            #   v_t = 4u_xx.
            residual_u = (u_next - u_curr) / DT - v_next
            residual_v = (v_next - v_curr) / DT - C_SQUARED * u_xx_next
            loss_res = loss_res + torch.mean(residual_u ** 2) + torch.mean(residual_v ** 2)

            # Dirichlet boundary in x: u(0,t)=u(1,t)=0.
            loss_bc = loss_bc + torch.mean(u_next[0:1] ** 2) + torch.mean(u_next[-1:] ** 2)

            u = u_next
            v = v_next

        loss_res = loss_res / (NT - 1)
        loss_bc = loss_bc / (NT - 1)
        loss_ic = torch.zeros((), dtype=torch.float32, device=device)

        loss_track.append([loss_res.item(), loss_bc.item(), loss_ic.item()])

        loss = loss_res + loss_ic + loss_bc
        optim.zero_grad()
        loss.backward()
        return loss

    optim.step(closure)

end_time = time.time() - start
print('Loss Res: {:4f}, Loss_BC: {:4f}, Loss_IC: {:4f}'.format(loss_track[-1][0], loss_track[-1][1], loss_track[-1][2]))
print('Train Loss: {:4f}'.format(np.sum(loss_track[-1])))

model_name = {args.model}
print(model_name, "time is ", end_time)


if not os.path.exists('./results/'):
    os.makedirs('./results/')

torch.save(model.state_dict(), f'./results/1dwave_{args.model}_point_ar.pt')

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
plt.savefig(f'./results/1dwave_{args.model}_point_optimization_ar_pred.pdf', bbox_inches='tight')

plt.figure(figsize=(4, 3))
plt.imshow(u, aspect='equal')
plt.xlabel('x')
plt.ylabel('t')
plt.title('Exact u(x,t)')
plt.colorbar()
plt.tight_layout()
plt.axis('off')
plt.savefig('./results/1dwave_exact.pdf', bbox_inches='tight')

plt.figure(figsize=(4, 3))
plt.imshow(pred - u, aspect='equal', cmap='coolwarm', vmin=-0.3, vmax=0.3)
plt.xlabel('x')
plt.ylabel('t')
plt.title('Absolute Error')
plt.colorbar()
plt.tight_layout()
plt.axis('off')
plt.savefig(f'./results/1dwave_{args.model}_point_optimization_ar_error.pdf', bbox_inches='tight')
