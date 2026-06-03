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


### Installation

```bash
# Clone the repository
git clone https://github.com/[anonymous]/SA-MDPO-for-Qwen-vl.git
cd SA-MDPO-for-Qwen-vl

# Create environment from environment.yaml
conda env create -f environment.yaml
conda activate samdpo
```





## 🚀 Training

### SA-MDPO Fine-tuning on Qwen3-VL (Recommended)

```bash
bash scripts/finetune_mdpo_qwen3.sh
```

### SA-MDPO Fine-tuning on Qwen2.5-VL

```bash
bash scripts/finetune_mdpo_qwen2_5.sh
```



## 🧪 Evaluation

Evaluate trained models on MacroValue-Bench:

```bash
bash scripts/eval.sh 
```


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

## 🙏 Acknowledgement

This project builds upon [**Qwen-VL-Series-Finetune**](https://github.com/2U1/Qwen-VL-Series-Finetune) by [2U1](https://github.com/2U1), licensed under the [Apache-2.0 License](https://github.com/2U1/Qwen-VL-Series-Finetune/blob/master/LICENSE). We reuse and modify portions of the training pipeline, data loading code, and DeepSpeed configurations. We thank the author for the excellent open-source implementation.



---

## ⚖️ License

The code in this repository is released under the **Apache License 2.0** (consistent with the upstream Qwen-VL-Series-Finetune license). See [LICENSE](LICENSE) for details.



## 📬 Contact

For questions about the paper or codebase, please open an issue. Author contact information will be de-anonymized after the review period.
