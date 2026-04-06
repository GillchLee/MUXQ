MUXQ: Mixed-to-Uniform precision matriX Quantization via Low-Rank Outlier Decomposition

python eval_gpt2_util2.py

# MUXQ Quantization Experiment Suite

Automated quantization experiments for GPT2 models.

## Quick Start

### Install
```bash
pip install -r requirements.txt
```

```bash
python eval_gpt2_muxq.py \
  --model_path "openai-community/gpt2-large" \
  --zscore 3.0 \
  --out_mag 5.0 \
  --split_exponent 2 \
  --quant_method naive \
  --quant_mode per-tensor \
  --act_bits 8 \
  --weight_bits 8 \
  --device cuda
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

<p align="center">
  <img src="https://github.com/user-attachments/assets/8fd02ef2-cc66-4c87-a6ef-9a93c89b0c1d"
       alt="muxq_performance"
       width="650">
</p>


<p align="center">
  <img src="./MUXQ/muxq%20white%20wall.png" alt="MUXQ Structure" width="900">
</p>
