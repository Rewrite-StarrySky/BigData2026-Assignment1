# Extra analysis (NOT required by the assignment):
# Q4 only looks at layers 0/7/15. Here we repeat the same q_proj SVD for all 16 layers
# to check whether "layer 7 is the most spread-out" is a depth trend or a one-off.
# Run it as a new cell in finetune_q4.ipynb after `model` is loaded (cell 1),
# or as a script: it loads the model itself if `model` is not defined.
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path

if 'model' not in globals():
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B')

THRESHOLD = 50.0  # same 50% cumulative-energy criterion as Q4b
OUT_DIR = Path('/kaggle/working/Assignment1_q4') if Path('/kaggle').exists() else Path('Assignment1_q4')
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []
for k, layer in enumerate(model.model.layers):
    W = layer.self_attn.q_proj.weight.detach().float()
    S = torch.linalg.svdvals(W)                       # singular values only, sorted descending
    E = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2) * 100
    rows.append({
        'layer': k,
        'min_rank_for_50pct_energy': int((E >= THRESHOLD).nonzero()[0].item()) + 1,
        'energy_at_rank_16 (%)': float(E[15]),
        'energy_at_rank_64 (%)': float(E[63]),
        'energy_at_rank_256 (%)': float(E[255]),
        'sigma_1_energy_share (%)': float(S[0] ** 2 / torch.sum(S ** 2) * 100),
    })

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / 'extra_all_layers_summary.csv', index=False)
print(df.to_string(index=False))

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(df['layer'], df['min_rank_for_50pct_energy'], marker='o')
ax.set_xticks(df['layer'])
ax.set_xlabel('Layer'); ax.set_ylabel('Min rank for >= 50% energy')
ax.set_title('q_proj: minimum rank for 50% energy, all 16 layers'); ax.grid(True)
fig.tight_layout(); fig.savefig(OUT_DIR / 'extra_all_layers_min_rank.png', dpi=200); plt.show()
