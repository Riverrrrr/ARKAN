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

import json
import os
import random
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from model_dict import get_model
from util import get_n_params


def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class TeeLogger(object):
    """Print to terminal and a log file at the same time. Works on Windows too."""

    def __init__(self, filename):
        import sys
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def enable_file_logging(result_dir, prefix="train"):
    import sys
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(result_dir, f"{prefix}_{timestamp}.log")
    sys.stdout = TeeLogger(log_path)
    sys.stderr = sys.stdout
    print("=" * 80)
    print(f"Log file: {log_path}")
    print("=" * 80)
    return log_path


def safe_scale(value):
    value = float(value)
    return value if abs(value) > 1e-12 else 1.0


def make_model(args, in_dim, device):
    if args.model == "KAN":
        return get_model(args).Model(
            width=[in_dim] + [args.kan_width] * args.kan_depth + [1],
            grid=args.grid,
            k=args.k,
            grid_eps=1.0,
            noise_scale_base=0.25,
            device=device,
        ).to(device)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

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
    # The modified KAN forward is forward(x_res, t_res) and concatenates them.
    # For ARKAN we put all autoregressive features in x_res and pass an empty tail.
    empty_tail = features.new_zeros((features.shape[0], 0))
    return model(features, empty_tail)


def latest_checkpoint(path):
    latest_path = os.path.join(path, "latest.pt")
    return latest_path if os.path.exists(latest_path) else None


def save_checkpoint(path, model, optimizer, step_idx, pred_prev, pred_curr, elapsed):
    ckpt = {
        "step_idx": step_idx,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "pred_prev": None if pred_prev is None else pred_prev.detach().cpu(),
        "pred_curr": pred_curr.detach().cpu(),
        "elapsed_seconds": elapsed,
    }
    torch.save(ckpt, os.path.join(path, "latest.pt"))
    torch.save(ckpt, os.path.join(path, f"checkpoint_t{step_idx:04d}.pt"))


def relative_errors(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    rel_l1 = np.sum(np.abs(pred - true)) / (np.sum(np.abs(true)) + 1e-12)
    rel_l2 = np.sqrt(np.sum((pred - true) ** 2) / (np.sum(true ** 2) + 1e-12))
    return float(rel_l1), float(rel_l2)


def plot_result(result_dir, name, x, t, pred_all, true_all, cmap_error="coolwarm"):
    X, T = np.meshgrid(x, t, indexing="ij")
    pred = pred_all.T
    true = true_all.T
    error = pred - true

    plt.figure(figsize=(4, 3))
    plt.imshow(pred, aspect="auto", origin="lower", extent=[x[0], x[-1], t[0], t[-1]])
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("Predicted u(x,t)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{name}_pred.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(4, 3))
    plt.imshow(true, aspect="auto", origin="lower", extent=[x[0], x[-1], t[0], t[-1]])
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("Exact u(x,t)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{name}_exact.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(4, 3))
    vmax = max(float(np.max(np.abs(error))), 1e-12)
    plt.imshow(error, aspect="auto", origin="lower", extent=[x[0], x[-1], t[0], t[-1]], cmap=cmap_error, vmin=-vmax, vmax=vmax)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("Error")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{name}_error.png"), dpi=150, bbox_inches="tight")
    plt.close()
