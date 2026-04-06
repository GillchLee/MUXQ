MUXQ: Mixed-to-Uniform precision matriX Quantization via Low-Rank Outlier Decomposition

python eval_gpt2_util2.py

# MUXQ Quantization Experiment Suite

Automated quantization experiments for GPT2 models.

## Quick Start

### Install
```bash
pip install -r requirements.txt
```

### Auto Run (Fixed Configuration)

**3 models, per-tensor & per-vector, act_bits 4-8, weight_bits 8 (fixed)**

```bash
python auto_run.py
```

This runs all combinations automatically:
- Models: gpt2-small, gpt2-medium, gpt2-large
- Activation bits: 4, 5, 6, 7, 8
- Weight bits: 8 (fixed)
- Modes: per-tensor, per-vector
- Total: 30 experiments


<img width="649" height="210" alt="muxq_performance" src="https://github.com/user-attachments/assets/8fd02ef2-cc66-4c87-a6ef-9a93c89b0c1d" />



<img width="1230" height="808" alt="muxq_structure" src="https://github.com/user-attachments/assets/23226190-1995-4a7f-84c4-3f458fe36478" />
