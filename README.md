# BigData2026 Assignment 1 — Analysis of SVD for Fine-Tuning LLMs

This repository contains the code and report for Assignment 1 of the *Big Data Analytics Techniques and Applications* course (Fall 2026). The assignment studies how different parts of the singular-value spectrum affect low-rank fine-tuning of a large language model.

## What this project does

`meta-llama/Llama-3.2-1B` is fine-tuned on 25,000 samples of **MetaMathQA** using **PiSSA** (Principal Singular values and Singular vectors Adaptation) and evaluated on **GSM8K** and **MATH** (accuracy on 1,000 questions each).

PiSSA is a low-rank adaptation method. For a pretrained weight matrix `W`, it computes the SVD `W = U S V^T` and initialises the trainable adapter from the selected singular components:

```text
A = U_r sqrt(S_r),   B = sqrt(S_r) V_r^T,   W_residual = W - AB
```

Only `A` and `B` are trained, while `W_residual` stays frozen. After training, `W_residual + AB` is merged back into the model. The adapters are applied to `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` in every Transformer layer.

## Experiments

| Question | Topic |
|---|---|
| Q1 | Accuracy of the original model vs. fine-tuned models with target rank 16 / 64 / 256 |
| Q2 | Training time for each target rank |
| Q3 | Rank fixed at 16; fine-tune the top 16, the 16 values starting at the 25% position, the 16 values starting at the 50% position, and the bottom 16 singular values |
| Q4 | SVD of `q_proj` in layers 0, 7 and 15 of the original model: minimum rank for 50% cumulative energy, cumulative-energy curves and singular-value magnitudes (no fine-tuning) |

## Files

- `bigdata-hw1-q1.ipynb`, `bigdata-hw1-q3.ipynb` — fine-tuning and evaluation (Q1–Q3), with outputs.
- `bigdata-hw1-q4.ipynb` — SVD analysis of the original model (Q4), with outputs.
- `extra_q4_all_layers.ipynb` — executed notebook (with outputs) of an optional extra analysis that repeats the Q4 `q_proj` analysis for all 16 layers.
- `report/` — the report (`Asg1_315581033.pdf`) and its figures.

The notebooks are written to run on Kaggle (GPU) and require access to the gated `meta-llama/Llama-3.2-1B` model via a Hugging Face token stored as the Kaggle secret `HF_TOKEN`.

## Acknowledgement and original source

The code is adapted from the course example code, which is itself modified from the official PiSSA implementation.

- Meng, F., Wang, Z., & Zhang, M. (2024). **PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models.** *Advances in Neural Information Processing Systems (NeurIPS 2024)*. https://arxiv.org/abs/2404.02948
- Official code: https://github.com/MuLabPKU/PiSSA
