# [Workshop on Asian Translation 2025](https://lotus.kuee.kyoto-u.ac.jp/WAT/WAT2025/index.html)

This reponsitory contains the data source, fine-tuning scripts and eval results for the Workshop on Asian Translation (WAT) 2025.

## System Description
- We present OdiaGenAI's participation in the WAT 2025 for Indic text-to-text machine translation task.
- We fine-tuned the base model for 4 times, once for each English-Indic language pair (English-Hindi, English-Bengali, English-Malayalam, English-Odia) using the LoRA method on top of the NLLB-200-3.3B model.
- Base Model: [NLLB-200-3.3B](https://huggingface.co/facebook/nllb-200-3.3B)
- Fine-tuning Framework: LoRA method
- Infrastructure used: 8 × AMD Instinct MI250X/MI250 GPUs

## Data
- We used the following datasets for fine-tuning the NLLB-200-3.3B model:

- Training Data:  
    - WAT 2025 Indic text-to-text machine translation task training data
        - 28930 sentence pairs for English-Indic language pairs (Train)
        - 998 sentence pairs for English-Indic language pairs (Dev)
        - 1595 sentence pairs for Indic-Indic language pairs (Evaluation)
        - 1400 sentence pairs for Indic-Indic language pairs (Challenge)
    - Samanantar corpus
        - 100K sentence pairs for English-Indic language pairs (Train)

## Metric used for model's evaluation
- Sacrebleu score (During fine-tuning)
- Sacrebleu and RIBES score (Final evaluation)

## Training logs
- English-Bengali text-to-text translation task: [Training Logs](https://wandb.ai/debasishdhal/wat2025-facebook-nllb-200-3.3B-bengali-finetune?nw=nwuserdebasishdhal)
- English-Hindi text-to-text translation task: [Training Logs](https://wandb.ai/debasishdhal/wat2025-facebook-nllb-200-3.3B-hindi-finetune?nw=nwuserdebasishdhal)
- Hindi-Malayalam text-to-text translation task: [Training Logs](https://wandb.ai/debasishdhal/wat2025-facebook-nllb-200-3.3B-malayalam-finetune?nw=nwuserdebasishdhal)
- English-Odia text-to-text translation task: [Training Logs](https://wandb.ai/debasishdhal/wat2025-facebook-nllb-200-3.3B-odia-finetune?nw=nwuserdebasishdhal)

## Fine-tuned Models
- [HuggingFace Link](https://huggingface.co/collections/OdiaGenAI/wat-2025-finetunedmodels)
- Here you can find the fine-tuned models for the following language pairs:
    - English to Hindi
    - English to Bengali
    - English to Malayalam
    - English to Odia

# Evaluation Results
### Table: WAT2025 Automatic and Manual Evaluation Results

| **System and WAT Task Label** | **OdiaGen AI (BLEU)** | **Best Comp (BLEU)** | **OdiaGen AI (RIBES)** | **Best Comp (RIBES)** |
|-------------------------------|-----------------------:|----------------------:|------------------------:|-----------------------:|
| **English → Hindi** |||||
| MMEVTEXT21en-hi | 45.10 | **45.40** | 0.831 | **0.834** |
| MMCHTEXT22en-hi | **56.90** | 56.10 | 0.870 | **0.870** |
| **English → Bengali** |||||
| MMEVTEXT22en-bn | **49.50** | **49.50** | **0.804** | 0.801 |
| MMCHTEXT22en-bn | **50.10** | 47.50 | **0.830** | 0.819 |
| **English → Malayalam** |||||
| MMEVTEXT21en-ml | 43.20 | **51.20** | 0.708 | **0.760** |
| MMCHTEXT22en-ml | **44.20** | 40.30 | **0.775** | 0.757 |
| **English → Odia** |||||
| MMEVTEXT21en-od | 62.90 | **64.30** | 0.903 | **0.906** |
| MMCHTEXT21en-od | **56.40** | 55.40 | 0.916 | **0.916** |

## Contributors
- Debasish Dhal
- Sambit Sekhar
- Revathy V. R.
- Akash Kumar Dhaka
- Shantipriya Parida