# LLM Fine-Tuning: SFT, RLHF, DPO, and ORPO with PEFT + LoRA

A comprehensive, hands-on collection of Jupyter notebooks covering every major approach
to fine-tuning and aligning Large Language Models — from basic Supervised Fine-Tuning
all the way to preference optimization with LoRA adapters for low-cost training.

All notebooks use **Mistral-7B** as the base model and are designed to run on a single
GPU using 4-bit quantization (QLoRA).

---

## What's Inside

| Notebook | Method | Description |
|---|---|---|
| `sft_with_trl.ipynb` | SFT | Full fine-tune using TRL's SFTTrainer |
| `sft_with_peft_lora.ipynb` | SFT + LoRA | Parameter-efficient SFT with QLoRA |
| `sft_with_rlhf_PPO.ipynb` | SFT + RLHF + PPO | Full RLHF pipeline with reward model and PPO |
| `dpo_with_trl.ipynb` | DPO | Direct Preference Optimization (full fine-tune) |
| `dpo_with_peft_lora_trl.ipynb` | DPO + LoRA | DPO with parameter-efficient LoRA adapters |
| `orpo_with_trl.ipynb` | ORPO | Odds Ratio Preference Optimization (full fine-tune) |
| `orpo_with_peft_lora_trl.ipynb` | ORPO + LoRA | ORPO with parameter-efficient LoRA adapters |

---

## Method Overview

Understanding which method to use and when is the most important decision in LLM alignment.
Here is how each method in this repo fits into the bigger picture.

### Supervised Fine-Tuning (SFT)

SFT is the foundation of everything. It takes a raw pretrained model that only knows how
to complete text and teaches it to follow instructions by training on `(prompt, response)`
pairs with standard cross-entropy loss.
```
Pretrained base model  →  SFT  →  Instruction-following model
```

**When to use:** Domain adaptation, teaching a new format or style, or as the mandatory
first step before DPO.

**Dataset format:**
```json
{"messages": [
    {"role": "user", "content": "Explain black holes."},
    {"role": "assistant", "content": "A black hole is..."}
]}
```

---

### RLHF + PPO (Reinforcement Learning from Human Feedback)

The original alignment method. Trains a separate reward model on human preference data,
then uses PPO (Proximal Policy Optimization) to update the SFT model toward
higher-reward responses. Requires four model roles simultaneously:

- **Policy** — the SFT model being trained
- **Reference model** — frozen SFT copy providing the KL penalty
- **Reward model** — frozen, scores each generated response
- **Value head** — predicts expected reward for advantage estimation
```
Pretrained  →  SFT  →  Train Reward Model  →  PPO loop  →  Aligned model
```

**When to use:** When your reward signal is complex (code that must pass tests, multi-step
tasks, live human feedback in the loop). For most static alignment tasks, DPO achieves
comparable results with far less complexity.

---

### DPO (Direct Preference Optimization)

Eliminates the reward model and PPO entirely. Reparameterizes the reward directly in
terms of the policy, turning alignment into a single supervised loss. Requires a frozen
copy of the SFT model as `π_ref` — this is why **SFT must come first**.
```
Loss = -log σ( β·log(π_θ(chosen)/π_ref(chosen)) - β·log(π_θ(rejected)/π_ref(rejected)) )
```
```
Pretrained  →  SFT (save checkpoint as π_ref)  →  DPO  →  Aligned model
```

**When to use:** You already have an SFT checkpoint and preference pair data. Stable,
efficient, and achieves alignment quality close to RLHF.

**Dataset format:**
```json
{"prompt": "...", "chosen": "good response", "rejected": "bad response"}
```

> **Note:** If using an `-Instruct` model (e.g. `Mistral-7B-Instruct-v0.2`), it is
> already SFT'd — you can go straight to DPO without the SFT step.

---

### ORPO (Odds Ratio Preference Optimization)

The most streamlined approach. Combines SFT and preference alignment into a **single
training step** with a single loss function — no reference model needed at all.
```
L_ORPO = L_SFT + λ · L_OR

L_SFT = standard cross-entropy on chosen responses
L_OR  = -log σ( log[odds(chosen)] - log[odds(rejected)] )
```
```
Pretrained  →  ORPO  →  Aligned model   (one step, one model copy)
```

**When to use:** Starting from a raw base model with no SFT checkpoint, VRAM-constrained
environments, or when you want the simplest possible pipeline.

**Dataset format:** Same as DPO — `(prompt, chosen, rejected)` pairs.

---

### PEFT + LoRA

Applied across all methods above. Instead of updating all model weights, LoRA injects
small trainable rank-decomposition matrices into the attention layers. Only ~0.3% of
parameters are trained, reducing VRAM requirements dramatically.

| Approach | VRAM (7B model) | Trainable params |
|---|---|---|
| Full SFT / DPO / ORPO | ~40–80 GB | 7B (100%) |
| LoRA SFT / DPO / ORPO | ~8–16 GB | ~20M (0.3%) |
| QLoRA (4-bit + LoRA) | ~5–10 GB | ~20M (0.3%) |

---

## Method Decision Guide
```
Do you have preference pairs (chosen + rejected)?
│
├── No  →  Use SFT only
│
└── Yes
    │
    ├── Do you have an SFT checkpoint already?
    │   │
    │   ├── Yes  →  Use SFT → DPO
    │   │           (best control, fine-tune β for drift)
    │   │
    │   └── No
    │       │
    │       ├── VRAM constrained?  →  Use ORPO
    │       │                          (1 step, 1× model in memory)
    │       │
    │       └── Not constrained   →  Use SFT → DPO
    │
    └── Using an -Instruct model?  →  Skip SFT, go straight to DPO
```

---

## Requirements
```bash
pip install transformers trl peft bitsandbytes datasets accelerate torch
```

| Library | Purpose |
|---|---|
| `transformers` | Model loading, tokenization, generation |
| `trl` | SFTTrainer, DPOTrainer, ORPOTrainer, PPOTrainer |
| `peft` | LoRA, QLoRA adapter management |
| `bitsandbytes` | 4-bit quantization (QLoRA) |
| `datasets` | Dataset loading and preprocessing |
| `accelerate` | Multi-GPU and device management |

---

## Hardware

All notebooks are designed to run on a **single GPU** using 4-bit QLoRA quantization.

| Method | Minimum VRAM |
|---|---|
| SFT + QLoRA | 8 GB |
| DPO + QLoRA | 10 GB (2 model copies during training) |
| ORPO + QLoRA | 6 GB (1 model copy) |
| RLHF + PPO | 24 GB+ (4 model roles) |

Tested on Google Colab (A100) and Kaggle (P100/T4).

---

## Base Model

All notebooks use **`mistralai/Mistral-7B-v0.1`** as the base model.

- SFT notebooks train from the raw base — a chat template is set manually before training
- DPO notebooks can optionally start from `mistralai/Mistral-7B-Instruct-v0.2` directly,
  skipping the SFT step since the Instruct variant is already fine-tuned
- The Mistral `[INST]...[/INST]` chat format is used consistently across all notebooks

---

## Key Concepts

**LoRA rank (`r`)** — controls how much capacity the adapter has. Higher rank = more
expressive but more parameters. Use `r=64` for SFT (learning new behavior), `r=16` for
alignment (steering existing behavior).

**β (beta)** — in DPO, controls how far the policy can drift from the reference model.
In ORPO, controls the weight of the odds ratio penalty. Start at `0.1` for both.

**QLoRA** — combines 4-bit quantization with LoRA. The base model is loaded in 4-bit
(frozen), while LoRA adapters are trained in bf16. This is the primary technique enabling
7B model training on consumer GPUs.

**Chat template** — the `[INST]...[/INST]` format that tells the model where user turns
end and assistant turns begin. Base models have no template — you set one during SFT.
Instruct models have it built in.

---

## Repo Structure
```
peft-lora-llm-finetuning/
│
├── sft_with_trl.ipynb              # SFT — full fine-tune
├── sft_with_peft_lora.ipynb        # SFT — QLoRA
├── sft_with_rlhf_PPO.ipynb         # SFT + RLHF + PPO full pipeline
│
├── dpo_with_trl.ipynb              # DPO — full fine-tune
├── dpo_with_peft_lora_trl.ipynb    # DPO — QLoRA
│
├── orpo_with_trl.ipynb             # ORPO — full fine-tune
└── orpo_with_peft_lora_trl.ipynb   # ORPO — QLoRA
```

---

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — Rafailov et al., 2023
- [ORPO: Monolithic Preference Optimization](https://arxiv.org/abs/2403.07691) — Hong et al., 2024
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — Ouyang et al., 2022
- [TRL — Transformer Reinforcement Learning](https://github.com/huggingface/trl)
- [PEFT — Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)

---

## License

MIT License — free to use, modify, and distribute.
