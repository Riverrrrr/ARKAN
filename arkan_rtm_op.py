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
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from model_dict import get_model
from util import get_n_params


seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def laplace_2d_same(u):
    lap = torch.zeros_like(u)
    lap[1:-1, 1:-1] = (
        u[2:, 1:-1]
        + u[:-2, 1:-1]
        + u[1:-1, 2:]
        + u[1:-1, :-2]
        - 4.0 * u[1:-1, 1:-1]
    )
    return lap


def solve_fd2d_abc_torch(u_curr, u_prev, r, wave=None, source_t=None, source_x=50, source_z=0):
    u_out = 2.0 * u_curr - u_prev + (r ** 2) * laplace_2d_same(u_curr)

    u_out[:, -1] = u_curr[:, -2] + (r[:, -1] - 1.0) / (r[:, -1] + 1.0) * (
        u_out[:, -2] - u_curr[:, -1]
    )
    u_out[-1, :] = u_curr[-2, :] + (r[-1, :] - 1.0) / (r[-1, :] + 1.0) * (
        u_out[-2, :] - u_curr[-1, :]
    )
    u_out[0, :] = u_curr[1, :] + (r[0, :] - 1.0) / (r[0, :] + 1.0) * (
        u_out[1, :] - u_curr[0, :]
    )

    if wave is not None and source_t is not None and 0 <= source_t < wave.numel():
        u_out[source_x, source_z] = u_out[source_x, source_z] + wave[source_t]

    return u_out


def make_model(args, in_dim, device):
    if args.model == "ARKAN":
        return get_model(args).Model(
            width=[in_dim, args.kan_width, args.kan_width, 1],
            grid=args.grid,
            k=args.k,
            grid_eps=1.0,
            noise_scale_base=0.25,
            device=device,
        ).to(device)
    if args.model == "QRes":
        model = get_model(args).Model(
            in_dim=in_dim, hidden_dim=args.hidden_dim, out_dim=1, num_layer=args.num_layer
        ).to(device)
    else:
        model = get_model(args).Model(
            in_dim=in_dim, hidden_dim=args.hidden_dim, out_dim=1, num_layer=args.num_layer
        ).to(device)
    model.apply(init_weights)
    return model


def model_forward(model, features):
    empty_tail = features.new_zeros((features.shape[0], 0))
    return model(features, empty_tail)


def safe_scale(value):
    value = float(value)
    return value if value > 1e-12 else 1.0


def correction_scale_from_frame(frame, scales, mode):
    if mode == "global":
        return frame.new_tensor(scales["u"])
    if mode == "frame_rms":
        value = torch.sqrt(torch.mean(frame.detach() ** 2))
    else:
        value = torch.max(torch.abs(frame.detach()))
    return torch.clamp(value, min=1e-6)


def build_features(
    ix,
    iz,
    t_idx,
    nt,
    x_grid,
    z_grid,
    u_prev,
    u_curr,
    u_fd,
    vmodel,
    r,
    wave,
    scales,
    correction_scale,
):
    t_norm = torch.full_like(ix.float(), (2.0 * float(t_idx) / float(max(nt - 1, 1))) - 1.0)
    if wave.numel() == 0:
        wave_t = torch.zeros_like(ix.float())
    else:
        source_t = min(t_idx + 1, wave.shape[0] - 1)
        wave_t = torch.ones_like(ix.float()) * (wave[source_t] / scales["wave"])

    return torch.stack(
        [
            x_grid[ix, iz],
            z_grid[ix, iz],
            t_norm,
            u_curr[ix, iz] / scales["u"],
            u_prev[ix, iz] / scales["u"],
            u_fd[ix, iz] / correction_scale,
            vmodel[ix, iz] / scales["v"],
            r[ix, iz] / scales["r"],
            wave_t,
        ],
        dim=-1,
    )


def predict_next_frame(
    model,
    u_prev,
    u_curr,
    t_idx,
    nt,
    x_grid,
    z_grid,
    vmodel,
    r,
    wave,
    scales,
    correction_scale_mode,
    chunk_size,
    source_x,
    source_z,
    point_scope,
):
    u_fd = solve_fd2d_abc_torch(
        u_curr, u_prev, r, wave=wave, source_t=t_idx + 1, source_x=source_x, source_z=source_z
    )
    u_next = u_fd.clone()
    correction_scale = correction_scale_from_frame(u_fd, scales, correction_scale_mode)

    x_start = 1 if point_scope == "interior" else 0
    x_end = u_curr.shape[0] - 1 if point_scope == "interior" else u_curr.shape[0]
    z_start = 1 if point_scope == "interior" else 0
    z_end = u_curr.shape[1] - 1 if point_scope == "interior" else u_curr.shape[1]
    correction_ix, correction_iz = torch.meshgrid(
        torch.arange(x_start, x_end, device=u_curr.device),
        torch.arange(z_start, z_end, device=u_curr.device),
        indexing="ij",
    )
    ix_flat = correction_ix.reshape(-1)
    iz_flat = correction_iz.reshape(-1)

    corrections = []
    with torch.no_grad():
        for start in range(0, ix_flat.numel(), chunk_size):
            end = start + chunk_size
            features = build_features(
                ix_flat[start:end],
                iz_flat[start:end],
                t_idx,
                nt,
                x_grid,
                z_grid,
                u_prev,
                u_curr,
                u_fd,
                vmodel,
                r,
                wave,
                scales,
                correction_scale,
            )
            corrections.append(model_forward(model, features).squeeze(-1) * correction_scale)

    correction = torch.cat(corrections)
    u_next[ix_flat, iz_flat] = u_fd[ix_flat, iz_flat] + correction
    return u_next, u_fd


def latest_checkpoint(path):
    latest_path = os.path.join(path, "latest.pt")
    return latest_path if os.path.exists(latest_path) else None


def save_checkpoint(path, model, optimizer, step_idx, pred_prev, pred_curr, elapsed):
    checkpoint = {
        "step_idx": step_idx,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "pred_prev": pred_prev.detach().cpu(),
        "pred_curr": pred_curr.detach().cpu(),
        "elapsed_seconds": elapsed,
    }
    torch.save(checkpoint, os.path.join(path, "latest.pt"))
    torch.save(checkpoint, os.path.join(path, f"checkpoint_t{step_idx:04d}.pt"))


def compute_total_metrics(pred_all, u_data, start_frame, end_frame):
    pred_eval = pred_all[:, :, start_frame:end_frame].astype(np.float64)
    true_eval = np.asarray(u_data[:, :, start_frame:end_frame], dtype=np.float64)

    total_rel_l1 = np.sum(np.abs(pred_eval - true_eval)) / (np.sum(np.abs(true_eval)) + 1e-12)
    total_rel_l2 = np.sqrt(
        np.sum((pred_eval - true_eval) ** 2) / (np.sum(true_eval ** 2) + 1e-12)
    )

    norm_l1_list = []
    norm_l2_list = []
    for t in range(start_frame, end_frame):
        pred = pred_all[:, :, t].astype(np.float64)
        true = np.asarray(u_data[:, :, t], dtype=np.float64)
        pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-12)
        true_norm = (true - true.min()) / (true.max() - true.min() + 1e-12)
        norm_l1_list.append(
            np.sum(np.abs(pred_norm - true_norm)) / (np.sum(np.abs(true_norm)) + 1e-12)
        )
        norm_l2_list.append(
            np.sqrt(
                np.sum((pred_norm - true_norm) ** 2) / (np.sum(true_norm ** 2) + 1e-12)
            )
        )

    return {
        "start_frame": int(start_frame),
        "end_frame_exclusive": int(end_frame),
        "total_rel_l1": float(total_rel_l1),
        "total_rel_l2": float(total_rel_l2),
        "mean_norm_l1": float(np.mean(norm_l1_list)),
        "mean_norm_l2": float(np.mean(norm_l2_list)),
    }


parser = argparse.ArgumentParser("RTM AR-KAN Point Optimization")
parser.add_argument("--model", type=str, default="ARKAN")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--data_dir", type=str, default="./rtm_data")
parser.add_argument("--result_dir", type=str, default="./results/rtm_arkan")
parser.add_argument("--epochs_per_step", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--dt", type=float, default=1e-4)
parser.add_argument("--dx", type=float, default=1.0)
parser.add_argument("--dz", type=float, default=1.0)
parser.add_argument("--source_x", type=int, default=50)
parser.add_argument("--source_z", type=int, default=0)
parser.add_argument("--start_t", type=int, default=1)
parser.add_argument("--end_t", type=int, default=-1)
parser.add_argument("--chunk_size", type=int, default=16384)
parser.add_argument("--save_every", type=int, default=1)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--rollout_loss_weight", type=float, default=1.0)
parser.add_argument("--direct_data_loss_weight", type=float, default=1.0)
parser.add_argument("--direct_l1_loss_weight", type=float, default=1.0)
parser.add_argument("--correction_scale", type=str, default="frame_max", choices=["global", "frame_max", "frame_rms"])
parser.add_argument("--point_scope", type=str, default="all", choices=["all", "interior"])
parser.add_argument("--kan_width", type=int, default=32)
parser.add_argument("--grid", type=int, default=5)
parser.add_argument("--k", type=int, default=3)
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--num_layer", type=int, default=4)
args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.isabs(args.data_dir):
    args.data_dir = os.path.join(script_dir, args.data_dir)
if not os.path.isabs(args.result_dir):
    args.result_dir = os.path.join(script_dir, args.result_dir)

device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
os.makedirs(args.result_dir, exist_ok=True)
os.makedirs(os.path.join(args.result_dir, "frames"), exist_ok=True)

start_time = time.time()

u_path = os.path.join(args.data_dir, "u_abc_128_0.npy")
v_path = os.path.join(args.data_dir, "vmodel_256.npy")
wave_path = os.path.join(args.data_dir, "wave.npy")

u_data = np.load(u_path, mmap_mode="r")
vmodel_np = np.load(v_path).astype(np.float32)
wave_np = np.load(wave_path).astype(np.float32).reshape(-1)

nx, nz, nt = u_data.shape
end_t = nt - 2 if args.end_t < 0 else min(args.end_t, nt - 2)
if args.start_t < 1 or end_t < args.start_t:
    raise ValueError("Expected 1 <= start_t <= end_t <= nt - 2.")

vmodel = torch.tensor(vmodel_np, dtype=torch.float32, device=device)
r = vmodel * args.dt / args.dx
wave = torch.tensor(wave_np, dtype=torch.float32, device=device)

x_grid = torch.linspace(-1.0, 1.0, nx, device=device).view(nx, 1).expand(nx, nz)
z_grid = torch.linspace(-1.0, 1.0, nz, device=device).view(1, nz).expand(nx, nz)

scales = {
    "u": safe_scale(np.max(np.abs(u_data))),
    "v": safe_scale(np.max(np.abs(vmodel_np))),
    "r": safe_scale(float(torch.max(torch.abs(r)).detach().cpu())),
    "wave": safe_scale(np.max(np.abs(wave_np))),
}

in_dim = 9
model = make_model(args, in_dim, device)
optimizer = Adam(model.parameters(), lr=args.lr)

print(model)
print("n_params:", get_n_params(model))
print("u shape:", u_data.shape, "v shape:", vmodel_np.shape, "wave shape:", wave_np.shape)
print("scales:", scales)
print("device:", device)

metadata = {
    "args": vars(args),
    "u_shape": list(u_data.shape),
    "v_shape": list(vmodel_np.shape),
    "wave_shape": list(wave_np.shape),
    "feature_dim": in_dim,
    "scales": scales,
}
with open(os.path.join(args.result_dir, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

pred_all_path = os.path.join(args.result_dir, "pred_all.npy")
if args.resume and os.path.exists(pred_all_path):
    pred_all = np.load(pred_all_path)
else:
    pred_all = np.zeros((nx, nz, nt), dtype=np.float32)
    pred_all[:, :, 0] = np.asarray(u_data[:, :, 0], dtype=np.float32)
    pred_all[:, :, 1] = np.asarray(u_data[:, :, 1], dtype=np.float32)

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
    pred_prev = torch.tensor(np.asarray(u_data[:, :, current_start_t - 1], dtype=np.float32), device=device)
    pred_curr = torch.tensor(np.asarray(u_data[:, :, current_start_t], dtype=np.float32), device=device)
    elapsed_before = 0.0

point_count = nx * nz if args.point_scope == "all" else (nx - 2) * (nz - 2)
batch_size = min(args.batch_size, point_count)
loss_history_path = os.path.join(args.result_dir, "loss_history.json")
if args.resume and os.path.exists(loss_history_path):
    with open(loss_history_path, "r", encoding="utf-8") as f:
        loss_history = json.load(f)
else:
    loss_history = []

for t_idx in range(current_start_t, end_t + 1):
    true_prev = torch.tensor(np.asarray(u_data[:, :, t_idx - 1], dtype=np.float32), device=device)
    true_curr = torch.tensor(np.asarray(u_data[:, :, t_idx], dtype=np.float32), device=device)
    true_next = torch.tensor(np.asarray(u_data[:, :, t_idx + 1], dtype=np.float32), device=device)
    true_fd = solve_fd2d_abc_torch(
        true_curr,
        true_prev,
        r,
        wave=wave,
        source_t=t_idx + 1,
        source_x=args.source_x,
        source_z=args.source_z,
    ).detach()
    rollout_fd = solve_fd2d_abc_torch(
        pred_curr,
        pred_prev,
        r,
        wave=wave,
        source_t=t_idx + 1,
        source_x=args.source_x,
        source_z=args.source_z,
    ).detach()
    true_correction_scale = correction_scale_from_frame(true_fd, scales, args.correction_scale)
    rollout_correction_scale = correction_scale_from_frame(rollout_fd, scales, args.correction_scale)

    nn.Module.train(model, True)
    step_losses = []
    progress = tqdm(range(args.epochs_per_step), desc=f"train t={t_idx}->{t_idx + 1}")
    for _ in progress:
        flat_ids = torch.randint(0, point_count, (batch_size,), device=device)
        if args.point_scope == "all":
            ix = flat_ids // nz
            iz = flat_ids % nz
        else:
            ix = flat_ids // (nz - 2) + 1
            iz = flat_ids % (nz - 2) + 1

        features = build_features(
            ix,
            iz,
            t_idx,
            nt,
            x_grid,
            z_grid,
            true_prev,
            true_curr,
            true_fd,
            vmodel,
            r,
            wave,
            scales,
            true_correction_scale,
        )
        target = (true_next[ix, iz] - true_fd[ix, iz]) / true_correction_scale
        pred = model_forward(model, features).squeeze(-1)
        loss_teacher = torch.mean((pred - target) ** 2)
        teacher_next = true_fd[ix, iz] + pred * true_correction_scale
        loss_teacher_data = torch.mean(((teacher_next - true_next[ix, iz]) / true_correction_scale) ** 2)

        rollout_features = build_features(
            ix,
            iz,
            t_idx,
            nt,
            x_grid,
            z_grid,
            pred_prev,
            pred_curr,
            rollout_fd,
            vmodel,
            r,
            wave,
            scales,
            rollout_correction_scale,
        )
        rollout_target = (true_next[ix, iz] - rollout_fd[ix, iz]) / rollout_correction_scale
        rollout_pred = model_forward(model, rollout_features).squeeze(-1)
        loss_rollout = torch.mean((rollout_pred - rollout_target) ** 2)
        rollout_next = rollout_fd[ix, iz] + rollout_pred * rollout_correction_scale
        loss_rollout_data = torch.mean(((rollout_next - true_next[ix, iz]) / rollout_correction_scale) ** 2)

        loss_data = loss_teacher_data + args.rollout_loss_weight * loss_rollout_data
        loss_l1 = torch.mean(torch.abs((teacher_next - true_next[ix, iz]) / true_correction_scale))
        loss_l1 = loss_l1 + args.rollout_loss_weight * torch.mean(
            torch.abs((rollout_next - true_next[ix, iz]) / rollout_correction_scale)
        )
        loss = (
            loss_teacher
            + args.rollout_loss_weight * loss_rollout
            + args.direct_data_loss_weight * loss_data
            + args.direct_l1_loss_weight * loss_l1
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step_losses.append(float(loss.detach().cpu()))
        progress.set_postfix(
            loss=step_losses[-1],
            teacher=float(loss_teacher.detach().cpu()),
            rollout=float(loss_rollout.detach().cpu()),
            data=float(loss_data.detach().cpu()),
            l1=float(loss_l1.detach().cpu()),
        )

    nn.Module.train(model, False)
    pred_next, fd_next = predict_next_frame(
        model,
        pred_prev,
        pred_curr,
        t_idx,
        nt,
        x_grid,
        z_grid,
        vmodel,
        r,
        wave,
        scales,
        args.correction_scale,
        args.chunk_size,
        args.source_x,
        args.source_z,
        args.point_scope,
    )

    pred_np = pred_next.detach().cpu().numpy().astype(np.float32)
    pred_all[:, :, t_idx + 1] = pred_np
    np.save(os.path.join(args.result_dir, "frames", f"pred_t{t_idx + 1:04d}.npy"), pred_np)

    true_np = np.asarray(u_data[:, :, t_idx + 1], dtype=np.float32)
    rel_l1 = np.sum(np.abs(pred_np - true_np)) / (np.sum(np.abs(true_np)) + 1e-12)
    rel_l2 = np.sqrt(np.sum((pred_np - true_np) ** 2) / (np.sum(true_np ** 2) + 1e-12))
    pred_norm = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-12)
    true_norm = (true_np - true_np.min()) / (true_np.max() - true_np.min() + 1e-12)
    norm_rel_l1 = np.sum(np.abs(pred_norm - true_norm)) / (np.sum(np.abs(true_norm)) + 1e-12)
    norm_rel_l2 = np.sqrt(np.sum((pred_norm - true_norm) ** 2) / (np.sum(true_norm ** 2) + 1e-12))
    mean_loss = float(np.mean(step_losses)) if step_losses else 0.0
    elapsed = elapsed_before + time.time() - start_time

    loss_history.append(
        {
            "t_idx": t_idx,
            "mean_loss": mean_loss,
            "last_loss": step_losses[-1] if step_losses else 0.0,
            "rel_l1": float(rel_l1),
            "rel_l2": float(rel_l2),
            "norm_rel_l1": float(norm_rel_l1),
            "norm_rel_l2": float(norm_rel_l2),
            "elapsed_seconds": elapsed,
        }
    )
    print(
        f"t={t_idx}->{t_idx + 1} loss={mean_loss:.6e} "
        f"rel_l1={rel_l1:.6f} rel_l2={rel_l2:.6f} "
        f"norm_l1={norm_rel_l1:.6f} norm_l2={norm_rel_l2:.6f} elapsed={elapsed:.2f}s"
    )

    if (t_idx - args.start_t) % args.save_every == 0 or t_idx == end_t:
        np.save(pred_all_path, pred_all)
        with open(loss_history_path, "w", encoding="utf-8") as f:
            json.dump(loss_history, f, indent=2)
        save_checkpoint(args.result_dir, model, optimizer, t_idx, pred_curr, pred_next, elapsed)

    pred_prev = pred_curr.detach()
    pred_curr = pred_next.detach()

total_elapsed = elapsed_before + time.time() - start_time
np.save(pred_all_path, pred_all)
torch.save(model.state_dict(), os.path.join(args.result_dir, f"final_{args.model}.pt"))

with open(loss_history_path, "w", encoding="utf-8") as f:
    json.dump(loss_history, f, indent=2)

total_metrics = compute_total_metrics(
    pred_all,
    u_data,
    start_frame=args.start_t + 1,
    end_frame=end_t + 2,
)
with open(os.path.join(args.result_dir, "total_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(total_metrics, f, indent=2)

plt.figure(figsize=(5, 4))
plt.imshow(pred_all[:, :, end_t + 1].T, aspect="auto", cmap="seismic")
plt.title(f"Predicted frame t={end_t + 1}")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(args.result_dir, f"pred_t{end_t + 1:04d}.png"), dpi=150)
plt.close()

plt.figure(figsize=(5, 4))
plt.imshow((pred_all[:, :, end_t + 1] - np.asarray(u_data[:, :, end_t + 1], dtype=np.float32)).T,
           aspect="auto", cmap="coolwarm")
plt.title(f"Error frame t={end_t + 1}")
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(args.result_dir, f"error_t{end_t + 1:04d}.png"), dpi=150)
plt.close()

print(f"Finished. Total training time: {total_elapsed:.2f} seconds")
print(
    "Total metrics "
    f"frames=[{total_metrics['start_frame']}, {total_metrics['end_frame_exclusive']}) "
    f"rel_l1={total_metrics['total_rel_l1']:.6f} "
    f"rel_l2={total_metrics['total_rel_l2']:.6f} "
    f"mean_norm_l1={total_metrics['mean_norm_l1']:.6f} "
    f"mean_norm_l2={total_metrics['mean_norm_l2']:.6f}"
)
print(f"Results saved to: {args.result_dir}")
