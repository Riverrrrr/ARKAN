
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
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm

from arkan_1d_common import (
    enable_file_logging,
    latest_checkpoint,
    make_model,
    model_forward,
    plot_result,
    relative_errors,
    safe_scale,
    save_checkpoint,
    set_seed,
)
from util import get_n_params


def exact_wave(x, t):
    return np.sin(np.pi * x) * np.cos(2 * np.pi * t) + 0.5 * np.sin(3 * np.pi * x) * np.cos(6 * np.pi * t)


def build_features(ix, t_idx, nt, x_norm, u_prev, u_curr, u_base, scales):
    t_norm = torch.full_like(ix.float(), 2.0 * float(t_idx) / float(max(nt - 1, 1)) - 1.0)
    return torch.stack(
        [
            x_norm[ix],
            t_norm,
            u_prev[ix] / scales["u"],
            u_curr[ix] / scales["u"],
            u_base[ix] / scales["u"],
        ],
        dim=-1,
    )


def wave_base(u_prev, u_curr, cfl2, baseline):
    if baseline == "persistence":
        return u_curr.clone()
    # Optional central-difference baseline. For the default 101x101 grid and c=2 this is CFL-unstable,
    # so the default keeps a safer persistence baseline.
    u_base = torch.empty_like(u_curr)
    u_base[1:-1] = 2.0 * u_curr[1:-1] - u_prev[1:-1] + cfl2 * (u_curr[2:] - 2.0 * u_curr[1:-1] + u_curr[:-2])
    u_base[0] = 0.0
    u_base[-1] = 0.0
    return u_base


parser = argparse.ArgumentParser("1D Wave ARKAN Point Optimization")
parser.add_argument("--model", type=str, default="ARKAN")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--result_dir", type=str, default="./results/1dwave_arkan")
parser.add_argument("--nx", type=int, default=101)
parser.add_argument("--nt", type=int, default=101)
parser.add_argument("--epochs_per_step", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=101)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--start_t", type=int, default=1)
parser.add_argument("--end_t", type=int, default=-1)
parser.add_argument("--baseline", type=str, default="persistence", choices=["persistence", "fd"])
parser.add_argument("--rollout_loss_weight", type=float, default=1.0)
parser.add_argument("--direct_data_loss_weight", type=float, default=1.0)
parser.add_argument("--direct_l1_loss_weight", type=float, default=0.1)
parser.add_argument("--save_every", type=int, default=10)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--kan_width", type=int, default=32)
parser.add_argument("--kan_depth", type=int, default=2)
parser.add_argument("--grid", type=int, default=5)
parser.add_argument("--k", type=int, default=3)
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--num_layer", type=int, default=4)
args = parser.parse_args()

set_seed(0)
script_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.isabs(args.result_dir):
    args.result_dir = os.path.join(script_dir, args.result_dir)
os.makedirs(args.result_dir, exist_ok=True)
enable_file_logging(args.result_dir, prefix="1dwave")

device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
start_time = time.time()

x_np = np.linspace(0.0, 1.0, args.nx, dtype=np.float32)
t_np = np.linspace(0.0, 1.0, args.nt, dtype=np.float32)
u_np = exact_wave(x_np[:, None], t_np[None, :]).astype(np.float32)

end_t = args.nt - 2 if args.end_t < 0 else min(args.end_t, args.nt - 2)
if args.start_t < 1 or end_t < args.start_t:
    raise ValueError("Expected 1 <= start_t <= end_t <= nt - 2.")

x_norm = torch.linspace(-1.0, 1.0, args.nx, device=device)
u_data = torch.tensor(u_np, dtype=torch.float32, device=device)
scales = {"u": safe_scale(np.max(np.abs(u_np)))}
dx = 1.0 / float(args.nx - 1)
dt = 1.0 / float(args.nt - 1)
c = 2.0
cfl2 = (c * dt / dx) ** 2

in_dim = 5
model = make_model(args, in_dim, device)
optimizer = Adam(model.parameters(), lr=args.lr)

print(model)
print("n_params:", get_n_params(model))
print("data shape:", u_np.shape, "device:", device, "baseline:", args.baseline, "cfl2:", cfl2)
print("scales:", scales)

metadata = {"args": vars(args), "feature_dim": in_dim, "data_shape": list(u_np.shape), "scales": scales, "cfl2": cfl2}
with open(os.path.join(args.result_dir, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

pred_all_path = os.path.join(args.result_dir, "pred_all.npy")
if args.resume and os.path.exists(pred_all_path):
    pred_all = np.load(pred_all_path)
else:
    pred_all = np.zeros_like(u_np, dtype=np.float32)
    pred_all[:, 0] = u_np[:, 0]
    pred_all[:, 1] = u_np[:, 1]

resume_checkpoint = latest_checkpoint(args.result_dir) if args.resume else None
if resume_checkpoint is not None:
    ckpt = torch.load(resume_checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    current_start_t = max(int(ckpt["step_idx"]) + 1, args.start_t)
    pred_prev = ckpt["pred_prev"].to(device)
    pred_curr = ckpt["pred_curr"].to(device)
    elapsed_before = float(ckpt.get("elapsed_seconds", 0.0))
    print(f"Resumed from {resume_checkpoint}, next t_idx={current_start_t}")
else:
    current_start_t = args.start_t
    pred_prev = u_data[:, current_start_t - 1]
    pred_curr = u_data[:, current_start_t]
    elapsed_before = 0.0

loss_history_path = os.path.join(args.result_dir, "loss_history.json")
if args.resume and os.path.exists(loss_history_path):
    with open(loss_history_path, "r", encoding="utf-8") as f:
        loss_history = json.load(f)
else:
    loss_history = []

point_count = args.nx
batch_size = min(args.batch_size, point_count)

for t_idx in range(current_start_t, end_t + 1):
    true_prev = u_data[:, t_idx - 1]
    true_curr = u_data[:, t_idx]
    true_next = u_data[:, t_idx + 1]
    true_base = wave_base(true_prev, true_curr, cfl2, args.baseline).detach()
    rollout_base = wave_base(pred_prev, pred_curr, cfl2, args.baseline).detach()

    torch.nn.Module.train(model, True)
    step_losses = []
    progress = tqdm(range(args.epochs_per_step), desc=f"train t={t_idx}->{t_idx + 1}")
    for _ in progress:
        ix = torch.randint(0, point_count, (batch_size,), device=device)

        features = build_features(ix, t_idx, args.nt, x_norm, true_prev, true_curr, true_base, scales)
        target = (true_next[ix] - true_base[ix]) / scales["u"]
        pred_corr = model_forward(model, features).squeeze(-1)
        teacher_next = true_base[ix] + pred_corr * scales["u"]
        loss_teacher = torch.mean((pred_corr - target) ** 2)
        loss_teacher_data = torch.mean(((teacher_next - true_next[ix]) / scales["u"]) ** 2)

        rollout_features = build_features(ix, t_idx, args.nt, x_norm, pred_prev, pred_curr, rollout_base, scales)
        rollout_target = (true_next[ix] - rollout_base[ix]) / scales["u"]
        rollout_corr = model_forward(model, rollout_features).squeeze(-1)
        rollout_next = rollout_base[ix] + rollout_corr * scales["u"]
        loss_rollout = torch.mean((rollout_corr - rollout_target) ** 2)
        loss_rollout_data = torch.mean(((rollout_next - true_next[ix]) / scales["u"]) ** 2)
        loss_l1 = torch.mean(torch.abs((teacher_next - true_next[ix]) / scales["u"])) + args.rollout_loss_weight * torch.mean(
            torch.abs((rollout_next - true_next[ix]) / scales["u"])
        )
        loss = loss_teacher + args.rollout_loss_weight * loss_rollout + args.direct_data_loss_weight * (
            loss_teacher_data + args.rollout_loss_weight * loss_rollout_data
        ) + args.direct_l1_loss_weight * loss_l1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_losses.append(float(loss.detach().cpu()))
        progress.set_postfix(loss=step_losses[-1])

    pred_next = rollout_base.clone()
    chunks = []
    all_ix = torch.arange(point_count, device=device)
    with torch.no_grad():
        features = build_features(all_ix, t_idx, args.nt, x_norm, pred_prev, pred_curr, rollout_base, scales)
        corr = model_forward(model, features).squeeze(-1) * scales["u"]
        pred_next[all_ix] = rollout_base[all_ix] + corr

    pred_all[:, t_idx + 1] = pred_next.detach().cpu().numpy()
    rel_l1, rel_l2 = relative_errors(pred_all[:, t_idx + 1], u_np[:, t_idx + 1])
    elapsed = elapsed_before + time.time() - start_time
    mean_loss = float(np.mean(step_losses)) if step_losses else 0.0
    loss_history.append({"t_idx": t_idx, "mean_loss": mean_loss, "rel_l1": rel_l1, "rel_l2": rel_l2, "elapsed_seconds": elapsed})
    print(f"t={t_idx}->{t_idx + 1} loss={mean_loss:.6e} rel_l1={rel_l1:.6f} rel_l2={rel_l2:.6f} elapsed={elapsed:.2f}s")

    if (t_idx - args.start_t) % args.save_every == 0 or t_idx == end_t:
        np.save(pred_all_path, pred_all)
        with open(loss_history_path, "w", encoding="utf-8") as f:
            json.dump(loss_history, f, indent=2)
        save_checkpoint(args.result_dir, model, optimizer, t_idx, pred_prev, pred_next, elapsed)

    pred_prev = pred_curr.detach()
    pred_curr = pred_next.detach()

np.save(pred_all_path, pred_all)
torch.save(model.state_dict(), os.path.join(args.result_dir, f"final_{args.model}.pt"))
with open(loss_history_path, "w", encoding="utf-8") as f:
    json.dump(loss_history, f, indent=2)
rl1, rl2 = relative_errors(pred_all[:, args.start_t + 1 : end_t + 2], u_np[:, args.start_t + 1 : end_t + 2])
plot_result(args.result_dir, f"1dwave_{args.model}_arkan", x_np, t_np, pred_all, u_np)
print(f"Finished. total_rel_l1={rl1:.6f} total_rel_l2={rl2:.6f}")
print(f"Results saved to: {args.result_dir}")
