# ARKAN

Official implementation of **ARKAN: Autoregressive Kolmogorov-Arnold Network for RTM Wavefield Prediction**.

This repository provides the implementation of ARKAN for autoregressive RTM wavefield prediction.

---

## Environment Setup

Install the required Python dependencies using:

```bash
pip install -r requirements.txt
```

A CUDA-enabled GPU is recommended for running the experiments.

---

## Test Example

The following command can be used to evaluate a trained ARKAN model on the RTM wavefield prediction task.

### Windows PowerShell

```powershell
python arkan_rtm_op.py `
  --model ARKAN `
  --device cuda:0 `
  --batch_size 8000 `
  --direct_data_loss_weight 5 `
  --direct_l1_loss_weight 2 `
  --rollout_loss_weight 2 `
  --point_scope all `
  --test_only `
  --test_checkpoint_dir ".\results\rtm_arkan_total_l1" `
  --test_checkpoint_pattern "checkpoint_t{t:04d}.pt" `
  --test_output_dir ".\results\rtm_arkan_checkpoint_test" `
  --start_t 1 `
  --end_t 998
```

The main testing arguments are:

- `--model ARKAN`: specifies the ARKAN model.
- `--device cuda:0`: uses the first CUDA-enabled GPU.
- `--test_only`: runs the model in testing mode without training.
- `--test_checkpoint_dir`: specifies the directory containing the trained checkpoints.
- `--test_checkpoint_pattern`: specifies the naming pattern of checkpoints for different time steps.
- `--test_output_dir`: specifies the directory used to save the testing results.
- `--start_t 1`: specifies the first time step to be evaluated.
- `--end_t 998`: specifies the last time step to be evaluated.

Please make sure that the trained checkpoint files are available in:

```text
.\results\rtm_arkan_total_l1
```

before running the test.

---

## License

MIT License

Copyright (c) 2026 Beihang University

Developed by:  
**He Zichuan**

Project:  
**ARKAN: Autoregressive Kolmogorov-Arnold Network for RTM Wavefield Prediction**

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
