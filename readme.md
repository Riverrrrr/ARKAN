# ARKAN

Official implementation of **ARKAN: Autoregressive Kolmogorov-Arnold Network for RTM Wavefield Prediction**.

This repository provides the implementation of ARKAN and the baseline methods used in the experiments, including RTM wavefield prediction and PDE benchmark tasks.

---

## License

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

---


## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

The RTM dataset can be obtained by running:

```bash
cd ./rtm_data
python ./get_rtm_data/get_data.py
```

---

# Running Experiments

## 1. ARKAN

### RTM Wavefield Prediction

Run ARKAN on the RTM task:

```bash
python ARKAN_RTM_op.py \
--model ARKAN \
--device cuda:0 \
--epochs_per_step 40 \
--batch_size 8000 \
--direct_data_loss_weight 5 \
--direct_l1_loss_weight 2 \
--rollout_loss_weight 2 \
--point_scope all \
--result_dir ./results/test
```

---

### PDE Benchmark Tasks

Run ARKAN on PDE tasks:

#### Wave Equation

```bash
python arkan_wave.py --model ARKAN --epochs 40
```

#### Convection Equation

```bash
python arkan_convection.py --model ARKAN --epochs 40
```

#### Reaction Equation

```bash
python arkan_reaction.py --model ARKAN --epochs 40
```

---

## 2. Autoregressive Baselines on RTM

The following autoregressive baseline models are evaluated on the RTM task:

- AR-PINN
- AR-QRes
- AR-FLS
- AR-KAN

Run the corresponding models:

```bash
python ./ar_model_rtm.py --model PINN --device cuda:0 --test_only

python ./ar_model_rtm.py --model QRes --device cuda:0 --test_only

python ./ar_model_rtm.py --model FLS --device cuda:0 --test_only

python ./ar_model_rtm.py --model KAN --device cuda:0 --test_only
```

---

## 3. Baseline Models on PDE Tasks

The following baseline models are evaluated on PDE benchmark tasks:

- PINN
- QRes
- FLS
- KAN

Available PDE tasks:

```
wave
reaction
convection
```

Run the corresponding experiment:

```bash
python ./original_{task_name}.py --model PINN --device cuda:0

python ./original_{task_name}.py --model QRes --device cuda:0

python ./original_{task_name}.py --model FLS --device cuda:0

python ./original_{task_name}.py --model KAN --device cuda:0
```

Replace `{task_name}` with one of:

```
wave
reaction
convection
```

