# SA-MDPO for Qwen-VL

> Official implementation of  
> **"MAVEN: A Macro-Societal Value Evaluation Framework of Multimodal Content with Compact Aligned Evaluators"**


## 📖 Overview

This repository contains the official implementation of **SA-MDPO** (Span-Adaptive Multi-level DPO), a rank-gap-aware preference optimization method for distilling macro-societal value-judgment capability from frontier vision-language models into compact student models. SA-MDPO is part of the **MAVEN** framework, which evaluates multimodal content along 6 macro-societal value dimensions (Peace, Development, Equity, Justice, Democracy, Freedom) and 72 secondary indicators.

Trained with SA-MDPO and inferred under Multi-Role Consensus (MRC), our compact 2B model matches its 8B counterpart in the same family and approaches frontier closed-source VLMs on macro-societal value evaluation.

**Key features**:
- ✅ Support for both **Qwen2.5-VL** and **Qwen3-VL** model families
- ✅ Full SA-MDPO / MDPO / DPO training pipelines with LoRA
- ✅ DeepSpeed ZeRO-2 / ZeRO-3 (with offloading) configurations
- ✅ Gradio demo for interactive inference
- ✅ Evaluation on MacroValue-Bench (6 dimensions, 72 indicators)

---

## 📂 Repository Structure

SA-MDPO-for-Qwen-vl/
├── scripts/                            # Training & evaluation shell scripts
│   ├── finetune_mdpo_qwen2_5.sh        # Train on Qwen2.5-VL
│   ├── finetune_mdpo_qwen3.sh          # Train on Qwen3-VL (recommended)
│   ├── eval.sh                         # Evaluate on MacroValue-Bench
│   ├── merge_lora.sh                   # Merge LoRA adapters
│   ├── gradio.sh                       # Launch Gradio demo
│   └── zero*.json                      # DeepSpeed configurations
├── src/
│   ├── train/                          # Training entry points
│   ├── trainer/                        # DPO / MDPO / SA-MDPO trainers
│   ├── dataset/                        # Multi-level preference data loader
│   ├── model/                          # Qwen-VL model wrappers
│   ├── eval/                           # Evaluation metrics (QWK, VSMS, etc.)
│   ├── serve/                          # Gradio demo backend
│   ├── params.py                       # Training arguments
│   ├── constants.py                    # Project constants
│   ├── utils.py                        # Helper functions
│   └── merge_lora_weights.py           # LoRA merge utility
├── environment.yaml                    # Conda environment
└── LICENSE                             # Apache-2.0


---

## 🔧 Requirements

- Python ≥ 3.10
- CUDA ≥ 11.8
- PyTorch ≥ 2.1.0
- Transformers ≥ 4.45.0 (for Qwen3-VL: ≥ 4.57.0)
- DeepSpeed ≥ 0.14
- 1× NVIDIA A100/A800 GPU (40GB+) recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/[anonymous]/SA-MDPO-for-Qwen-vl.git
cd SA-MDPO-for-Qwen-vl

# Create environment from environment.yaml
conda env create -f environment.yaml
conda activate samdpo
```

Or with pip:

```bash
conda create -n samdpo python=3.10 -y
conda activate samdpo
pip install -r requirements.txt
```

---

## 📊 Dataset: MacroValue-Bench

**MacroValue-Bench** is a 1,157-item human-verified multimodal benchmark for macro-societal value evaluation, covering six value dimensions:

| Dimension | Description |
|---|---|
| **Peace** | Non-violence and harmonious coexistence |
| **Development** | Progress and capacity enhancement |
| **Equity** | Reasonable distribution of resources and rights |
| **Justice** | Institutions aligned with moral/legal principles |
| **Democracy** | Public participation in decision-making |
| **Freedom** | Individual autonomy in thought and action |

### Data Sources

| Source | Count | Percentage |
|---|---|---|
| Weibo | 573 | 49.5% |
| Global Times Online | 245 | 21.2% |
| Ch3Ef-harmless | 173 | 15.0% |
| People's Daily Online | 166 | 14.3% |
| **Total** | **1,157** | **100.0%** |

> **📌 Release Notice**  
> MacroValue-Bench will be released under a **research-only license** (with non-redistribution and ethical-use terms) **upon paper acceptance**.  
> A takedown mechanism will be provided for individuals to request removal of items in which they appear.  
> ⭐ **Watch this repository for updates.**

### Data Format

Each item in the dataset is a JSON object:

```json
{
  "image": "path/to/image.jpg",
  "text": "Caption or news text...",
  "labels": {
    "Peace":       {"Indicators": {...}, "Score": 2},
    "Development": {"Indicators": {...}, "Score": 1},
    "Equity":      {"Indicators": {...}, "Score": 0},
    "Justice":     {"Indicators": {...}, "Score": 1},
    "Democracy":   {"Indicators": {...}, "Score": 1},
    "Freedom":     {"Indicators": {...}, "Score": 0}
  }
}
```

---

## 🚀 Training

### SA-MDPO Fine-tuning on Qwen3-VL (Recommended)

```bash
bash scripts/finetune_mdpo_qwen3.sh
```

### SA-MDPO Fine-tuning on Qwen2.5-VL

```bash
bash scripts/finetune_mdpo_qwen2_5.sh
```

### Key Arguments (in `src/params.py`)

| Argument | Default | Description |
|---|---|---|
| `--model_name_or_path` | `Qwen/Qwen3-VL-2B-Instruct` | Base VLM checkpoint |
| `--data_path` | `data/preference_train.jsonl` | Multi-level preference data |
| `--dpo_beta` | `0.1` | Base regularization coefficient β₀ |
| `--span_alpha` | `0.5` | Rank-gap slope α (linear schedule) |
| `--num_levels` | `4` | Number of preference levels K |
| `--lora_r` | `16` | LoRA rank |
| `--lora_alpha` | `32` | LoRA alpha |
| `--learning_rate` | `1e-4` | Peak learning rate |
| `--num_train_epochs` | `4` | Training epochs |
| `--per_device_train_batch_size` | `2` | Per-device batch size |
| `--gradient_accumulation_steps` | `8` | Gradient accumulation (effective BS = 16) |
| `--deepspeed` | `scripts/zero2.json` | DeepSpeed config |

### SA-MDPO Loss

The core innovation: a rank-gap-aware β schedule that scales linearly with the rank distance:

β(i, j) = β₀ · (1 + α · (j-i) / (K-1)),   i < j


This addresses the limitation of standard MDPO, which applies a uniform β across all rank pairs and under-weights the supervision signal from distant rank pairs that carry the strongest quality contrast.

### DeepSpeed Configurations

Choose based on your GPU memory:

| Config | Use Case |
|---|---|
| `scripts/zero2.json` | Standard, fast training (recommended for 2B/4B) |
| `scripts/zero2_offload.json` | ZeRO-2 + CPU offload (lower memory) |
| `scripts/zero3.json` | ZeRO-3 for larger models (7B+) |
| `scripts/zero3_offload.json` | ZeRO-3 + CPU offload (lowest memory) |

---

## 🔀 Merging LoRA Adapters

After training, merge LoRA weights into the base model for deployment:

```bash
bash scripts/merge_lora.sh
```

Or directly via Python:

```bash
python src/merge_lora_weights.py \
    --model_path Qwen/Qwen3-VL-2B-Instruct \
    --lora_path checkpoints/qwen3-vl-2b-sa-mdpo \
    --save_path checkpoints/qwen3-vl-2b-sa-mdpo-merged
```

---

## 🧪 Evaluation

Evaluate trained models on MacroValue-Bench:

```bash
bash scripts/eval.sh \
    --model_path checkpoints/qwen3-vl-2b-sa-mdpo \
    --data_path data/macrovalue_bench/test.jsonl \
    --output_dir results/
```

### Evaluation Metrics

For **primary dimensions** (averaged over 6 dimensions):
- **QWK** (Quadratic Weighted Kappa) — primary ordinal metric
- **Accuracy** — exact match rate
- **F1-macro** — macro-averaged F1 across {-2, -1, 0, +1, +2}

For **secondary indicators**:
- **VSMS** (Value-Aware Soft Match Score) — accounts for semantic coupling
- **Recall** / **Precision** — VSMS decomposition

All metrics are higher-is-better.

---

## 🎨 Interactive Demo

Launch the Gradio demo for interactive value evaluation:

```bash
bash scripts/gradio.sh
```

Then open `http://localhost:7860` in your browser.

---

## 📈 Main Results

| Model | QWK ↑ | Accuracy ↑ | F1-macro ↑ | VSMS ↑ |
|---|---|---|---|---|
| Qwen3-VL-2B (baseline) | 0.393 | 0.497 | 0.300 | 0.798 |
| Qwen3-VL-4B | 0.442 | 0.663 | 0.322 | 0.954 |
| Qwen3-VL-8B | 0.546 | 0.691 | 0.380 | 0.961 |
| GPT-4o | 0.563 | 0.471 | 0.346 | 0.919 |
| GPT-5 | 0.637 | 0.724 | 0.520 | 0.962 |
| Gemini-2.5-Pro | 0.651 | 0.646 | 0.471 | 0.950 |
| Doubao-Seed-1.6-V | **0.702** | **0.756** | **0.524** | **0.966** |
| **Qwen3-VL-2B + SA-MDPO (Ours)** | 0.599 | 0.706 | 0.482 | 0.954 |
| **Qwen3-VL-2B + SA-MDPO + MRC (Ours)** | **0.624** | **0.719** | **0.497** | **0.959** |

Our compact 2B model matches Qwen3-VL-8B (4× smaller) and approaches frontier closed-source baselines.

### Ablation: Effect of SA-MDPO

| Method | Accuracy | QWK | VSMS |
|---|---|---|---|
| Baseline (Qwen3-VL-2B) | 0.497 | 0.393 | 0.798 |
| DPO (balanced) | 0.654 | 0.267 | 0.946 |
| MDPO (balanced) | 0.670 | 0.445 | 0.964 |
| **SA-MDPO (balanced)** | **0.706** | **0.599** | 0.954 |

---

## 📝 Citation

Our paper is currently under review. Citation information will be provided upon publication.

---

## 🙏 Acknowledgement

This project builds upon [**Qwen-VL-Series-Finetune**](https://github.com/2U1/Qwen-VL-Series-Finetune) by [2U1](https://github.com/2U1), licensed under the [Apache-2.0 License](https://github.com/2U1/Qwen-VL-Series-Finetune/blob/master/LICENSE). We reuse and modify portions of the training pipeline, data loading code, and DeepSpeed configurations. We thank the author for the excellent open-source implementation.

We also acknowledge:
- The **Qwen-VL** team at Alibaba for the base vision-language models.
- The authors of **MDPO** (Zhang et al., 2024) for the multi-level DPO foundation.
- The **Ch3Ef** project for the safety-oriented data subset.

---

## ⚖️ License

The code in this repository is released under the **Apache License 2.0** (consistent with the upstream Qwen-VL-Series-Finetune license). See [LICENSE](LICENSE) for details.

The **MacroValue-Bench dataset**, when released, will use a **research-only license** with non-redistribution and ethical-use terms.

---

## 🛡️ Responsible Use Statement

This framework is designed for **diagnostic evaluation** of AI-generated or AI-curated content, **NOT** for ranking, filtering, or sanctioning human expression. A value evaluator covering dimensions such as Democracy and Freedom could in principle be misused for content censorship or to legitimize the suppression of dissenting viewpoints — **this is directly opposite to MAVEN's intent**.

Automated decisions with material consequences should always involve human review. We release this framework to enable scrutiny and counter-argumentation, not to provide a turnkey moderation system. See our paper's Ethics Statement for detailed discussion.

---

## 📬 Contact

For questions about the paper or codebase, please open an issue. Author contact information will be de-anonymized after the review period.
