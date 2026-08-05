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
import os
import matplotlib.pyplot as plt
import random
from torch.optim import LBFGS, Adam
from tqdm import tqdm
import argparse
from util import *
from model_dict import get_model

import sys
import os

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def get_rtm_data(u_abc_cpu, u_abc_npy, tt):
    
    u_abc_t1 = u_abc_npy[:,:,tt-3:tt-1]
    # u_abc_t2 = u_abc_npy[:,:,201:202]
    u_abc_t1_r = u_abc_t1.reshape(100*80, 2)
    # u_abc_t2_r = u_abc_t2.reshape(100*80, 1).unsqueeze(1)
    res = u_abc_t1_r
    u_abc_t3_all =u_abc_npy[:,:,tt-1:tt]
    u_abc_t3 = u_abc_t3_all.reshape(100*80)


    b_left = u_abc_t1[0,:,:]

    b_right = u_abc_t1[-1,:,:]

    b_lower = u_abc_t1[:,-1,:]

    b_upper = u_abc_t1[:,0,:]

    return res, b_left, b_right, b_lower, b_upper, u_abc_t3, u_abc_t3_all


def train_form(u_abc_cpu, u_abc_npy, ttt, model,device, file_name):

    for j in tqdm(range(1000-ttt)):
        t = j + ttt

        res, b_left, b_right, b_lower, b_upper, u_abc_t3, u_abc_t3_all = get_rtm_data(u_abc_cpu, u_abc_npy, t)

        u_left = u_abc_t3_all[0,:]
        u_right = u_abc_t3_all[-1,:]
        u_lower = u_abc_t3_all[:,-1]

        res = torch.tensor(res, dtype=torch.float32, requires_grad=True).to(device)
        b_left = torch.tensor(b_left, dtype=torch.float32, requires_grad=True).to(device)
        b_right = torch.tensor(b_right, dtype=torch.float32, requires_grad=True).to(device)
        # b_upper = torch.tensor(b_upper, dtype=torch.float32, requires_grad=True).to(device)
        b_lower = torch.tensor(b_lower, dtype=torch.float32, requires_grad=True).to(device)

        x_res, t_res = res[:, ..., 0:1], res[:, ..., 1:2]
        x_left, t_left = b_left[:, ..., 0:1], b_left[:, ..., 1:2]
        x_right, t_right = b_right[:, ..., 0:1], b_right[:, ..., 1:2]
        # x_upper, t_upper = b_upper[:, ..., 0:1], b_upper[:, ..., 1:2]
        x_lower, t_lower = b_lower[:, ..., 0:1], b_lower[:, ..., 1:2]

        u_abc_t3 = u_abc_t3.to(device)
        u_left = torch.tensor(u_left, dtype=torch.float32, requires_grad=True).to(device)
        u_right = torch.tensor(u_right, dtype=torch.float32, requires_grad=True).to(device)
        u_lower = torch.tensor(u_lower, dtype=torch.float32, requires_grad=True).to(device)

        optim = LBFGS(model.parameters(), line_search_fn='strong_wolfe')

        n_params = get_n_params(model)

        # print(model)
        # print(get_n_params(model))

        loss_track = []
        pi = torch.tensor(np.pi, dtype=torch.float32, requires_grad=False).to(device)

        for i in range(40):
            def closure():
                pred_res = model(x_res, t_res)
                pred_left = model(x_left, t_left)
                pred_right = model(x_right, t_right)
                # pred_upper = model(x_upper, t_upper)
                
                pred_lower = model(x_lower, t_lower)

                u_x = torch.autograd.grad(pred_res, x_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True,
                                        create_graph=True)[0]
                u_xx = \
                    torch.autograd.grad(u_x, x_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True,
                                        create_graph=True)[0]
                u_t = torch.autograd.grad(pred_res, t_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True,
                                        create_graph=True)[0]
                u_tt = \
                    torch.autograd.grad(u_t, t_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True,
                                        create_graph=True)[0]

                loss_pde = torch.mean((u_abc_t3 - pred_res)**2)

                loss_res = torch.mean((u_tt - 4 * u_xx) ** 2)
                loss_bc1 = torch.mean((u_left - pred_left) ** 2 + (u_right - pred_right) ** 2)
                loss_bc2 = torch.mean((u_lower - pred_lower) ** 2)
                loss_bc = loss_bc1 + loss_bc2


                ui_t = torch.autograd.grad(pred_left, t_left, grad_outputs=torch.ones_like(pred_left), retain_graph=True,
                                        create_graph=True)[0]

                loss_ic_1 = torch.mean(
                    (pred_left[:, 0] - torch.sin(pi * x_left[:, 0]) - 0.5 * torch.sin(3 * pi * x_left[:, 0])) ** 2)
                loss_ic_2 = torch.mean((ui_t) ** 2)

                loss_ic = loss_ic_1 * 0 + loss_ic_2 * 0

                loss_track.append([loss_res.item(), loss_ic.item(), loss_bc.item(), loss_pde.item()])

                loss = loss_res + loss_ic + loss_bc + loss_pde
                optim.zero_grad()
                loss.backward()
                return loss

            optim.step(closure)

    print('Loss Res: {:4f}, Loss_BC: {:4f}, Loss_IC: {:4f}, Loss_PDE'.format(loss_track[-1][0], loss_track[-1][1], loss_track[-1][2]), loss_track[-1][3])
    print('Train Loss: {:4f}'.format(np.sum(loss_track[-1])))

    if not os.path.exists(f'./{file_name}/{args.model}/'):
        os.makedirs(f'./{file_name}/{args.model}/')

    torch.save(model.state_dict(), f'./{file_name}/{args.model}/hzc_rtm_{args.model}_point.pt')


def predict_form(u_abc_cpu, u_abc_npy, ttt, device, model, l1_all, l2_all, l1_line, l2_line, file_name, pic_label):

    t_all = 1000

    for j in range(t_all-ttt):
        print("j is ",j)
        t = j + ttt
        res_test, b_left, b_right, b_lower, b_upper, u_abc_t3, u_abc_t3_all = get_rtm_data(u_abc_cpu, u_abc_npy, t)
        # Visualize
        if args.model == 'PINNsFormer' or args.model == 'PINNsFormer_Enc_Only':
            res_test = make_time_sequence(res_test, num_step=5, step=1e-4)

        res_test = torch.tensor(res_test, dtype=torch.float32, requires_grad=True).to(device)
        x_test, t_test = res_test[:, ..., 0:1], res_test[:, ..., 1:2]

        with torch.no_grad():
            pred = model(x_test, t_test)[:, 0:1]
            pred = pred.cpu().detach().numpy()

        pred_min = pred.min()
        pred_max = pred.max()
        pred_norm = (pred - pred_min) / (pred_max - pred_min)
        # pred = pred.reshape(100,80)
        pred = pred_norm.reshape(100,80)


        def u_ana(x, t):
            return np.sin(np.pi * x) * np.cos(2 * np.pi * t) + 0.5 * np.sin(3 * np.pi * x) * np.cos(6 * np.pi * t)
        u = u_abc_cpu[:,:,t-1:t].reshape(100,80)


        u_min = u.min()
        u_max = u.max()
        u_norm = (u - u_min) / (u_max - u_min)
        # pred = pred.reshape(100,80) * 500*(-1)
        u = u_norm

        rl1 = np.sum(np.abs(u - pred)) / np.sum(np.abs(u))
        rl2 = np.sqrt(np.sum((u - pred) ** 2) / np.sum(u ** 2))
        l1_all += rl1/(t_all-ttt)
        l2_all += rl2/(t_all-ttt)
        l1_line.append(rl1)
        l2_line.append(rl2)

        print('relative L1 error: {:4f}'.format(rl1))
        print('relative L2 error: {:4f}'.format(rl2))


        plt.figure(figsize=(4, 3))
        plt.imshow(pred, aspect='equal')
        plt.xlabel('x')
        plt.ylabel('t')
        # plt.title('Predicted u(x,t)')
        plt.colorbar()
        plt.tight_layout()
        plt.axis('off')
        plt.savefig(f'./{file_name}/{args.model}/hzc_{args.model}_rtm_pred_t{t}.pdf', bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(4, 3))
        plt.imshow(u, aspect='equal')
        plt.xlabel('x')
        plt.ylabel('t')
        # plt.title(pic_label)
        plt.colorbar()
        plt.tight_layout()
        plt.axis('off')
        plt.savefig(f'./{file_name}/{args.model}/hzc_rtm_exact_t{t}.pdf', bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(4, 3))
        plt.imshow(pred - u, aspect='equal', cmap='coolwarm', vmin=-1, vmax=1)
        plt.xlabel('x')
        plt.ylabel('t')
        # plt.title('Absolute Error')
        plt.colorbar()
        plt.tight_layout()
        plt.axis('off')
        plt.savefig(f'./{file_name}/{args.model}/hzc_{args.model}_rtm_error_t{t}.pdf', bbox_inches='tight')
        plt.close()

    print("\n\n\n")
    print('all relative L1 error: ', l1_line)
    print('all relative L2 error: ', l2_line)
    print("\n\n\n")
    print('relative L1 error: {:4f}'.format(l1_all))
    print('relative L2 error: {:4f}'.format(l2_all))


file_name = "hzc_old_pinn_test_v1/0702_all_test_time"
model_file = "hzc_old_pinn_test_v1"

nx = 100
nz = 80
dt = 1e-4
dx = 1.0
dz = 1.0
vp = 2000.0
nt = 1000
vmodel = vp * np.ones((nx, nz), dtype=np.float32)

vmodel[:, 50:] = vp * 1.3
vmodel[70:73, 28:31] = vp * 1.4
vmodel[28:31, 19:22] = vp * 1.6

r_model = vmodel * (dt / dx)

u_abc_cpu = np.load("./rtm_data/u_abc.npy")
u_abc_npy = torch.tensor(np.load("./rtm_data/u_abc.npy"))
u_abc_npy_reshape = u_abc_npy.reshape(100 * 80, 1000)
# u_abc = torch.from_numpy(u_abc_npy_reshape)

wav = np.zeros(nt, dtype=np.float32)
wav1 = np.load("./rtm_data/wave.npy")
wav[:min(len(wav1), nt)] = wav1[:min(len(wav1), nt)]


u_abc_test1 = u_abc_npy[:,:,300:302]
u_abc_test1_r = u_abc_test1.reshape(100*80,2)
res_test = u_abc_test1_r

src_x = 50
src_z = 0


seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

parser = argparse.ArgumentParser('Training Point Optimization')
parser.add_argument('--model', type=str, default='PINN')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--test_only', action='store_true')
args = parser.parse_args()
device = args.device

# 开启日志
log_path = f'./{file_name}/{args.model}/log.txt'
sys.stdout = Logger(log_path)

pic_label = f'{args.model}'

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

if args.model == 'KAN':
    model = get_model(args).Model(width=[2, 5, 5, 1], grid=5, k=3, grid_eps=1.0, \
                                noise_scale_base=0.25, device=device).to(device)
elif args.model == 'QRes':
    model = get_model(args).Model(in_dim=2, hidden_dim=256, out_dim=1, num_layer=4).to(device)
    model.apply(init_weights)
else:
    model = get_model(args).Model(in_dim=2, hidden_dim=512, out_dim=1, num_layer=4).to(device)
    model.apply(init_weights)

if args.test_only:
    model_path = f'./{model_file}/{args.model}/hzc_rtm_{args.model}_point.pt'

    print(f"Loading model from: {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    if args.model != 'KAN':
        model.eval()

else:
    train_form(u_abc_cpu, u_abc_npy, 200, model, device, file_name)


l1_all = 0
l2_all = 0
l1_line = []
l2_line = []

predict_form(
    u_abc_cpu,
    u_abc_npy,
    4,
    device,
    model,
    l1_all,
    l2_all,
    l1_line,
    l2_line,
    file_name,
    pic_label
)



# # train_form(u_abc_cpu, u_abc_npy, 200, model,device, file_name)




# l1_all = 0
# l2_all = 0
# l1_line = []
# l2_line = []

# predict_form(u_abc_cpu, u_abc_npy, 300, device, model, l1_all, l2_all, l1_line, l2_line, file_name, pic_label)


