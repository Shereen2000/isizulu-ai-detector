# isiZulu AI-Detection Classifier

A binary text classifier that distinguishes between **machine-generated** (label `1`) and **human-written** (label `0`) isiZulu text, fine-tuned on top of AfroXLMR-Large.

---

## Results

| Metric    | Score   |
|-----------|---------|
| Accuracy  | 98.70%  |
| F1        | 98.72%  |
| Precision | 99.71%  |
| Recall    | 97.75%  |
| Loss      | 0.0805  |

Training ran for **4 epochs** (~22 minutes on an RTX 5090) before early stopping triggered. The best checkpoint was saved at **epoch 2**.

### Per-Epoch Evaluation

| Epoch | F1     | Accuracy | Precision | Recall | Eval Loss |
|-------|--------|----------|-----------|--------|-----------|
| 1     | 96.69% | 96.69%   | 99.85%    | 93.72% | 0.1495    |
| **2** | **98.72%** | **98.70%** | **99.71%** | **97.75%** | **0.0805** ← best |
| 3     | 97.82% | 97.80%   | 99.78%    | 95.93% | 0.1129    |
| 4     | 98.42% | 98.39%   | 99.75%    | 97.12% | 0.1079    |

Early stopping (patience=2) halted training after epoch 4 since no epoch exceeded the epoch 2 F1 score.

---

## Project Structure

```
project/
├── base model/
│   └── model/                    # AfroXLMR-Large-29L base model weights
│       ├── config.json
│       ├── model.safetensors     # 2.24 GB
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── special_tokens_map.json
├── datasets/
│   ├── train.jsonl               # 16,377 samples
│   ├── eval.jsonl                # 1,991 samples
│   └── test.jsonl                # 2,022 samples
├── scripts/
│   └── finetune.py               # Main training script
├── finetuned_classifier/
│   ├── final_model/              # Best checkpoint (epoch 2), saved here
│   ├── checkpoint-1556/          # End of epoch 1
│   ├── checkpoint-3112/          # End of epoch 2 (best)
│   ├── checkpoint-4668/          # End of epoch 3
│   ├── checkpoint-6224/          # End of epoch 4
│   └── metrics.json              # Final evaluation metrics
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Dataset

All splits use the same format — one JSON object per line:

```json
{"text": "Umculo udlala indima ebalulekile...", "label": 0}
{"text": "Awu mngani, waze wangitshela...", "label": 1}
```

| Split | Total  | Label 0 (human) | Label 1 (machine) |
|-------|--------|-----------------|-------------------|
| train | 16,377 | 8,207 (50.1%)   | 8,170 (49.9%)     |
| eval  | 1,991  | 994 (49.9%)     | 997 (50.1%)       |
| test  | 2,022  | 1,013 (50.1%)   | 1,009 (49.9%)     |

The dataset is well-balanced across all splits. `train.jsonl` and `eval.jsonl` are used during fine-tuning; `test.jsonl` is reserved for held-out evaluation.

---

## Base Model

**AfroXLMR-Large-29L** (`bert_models/afro-xlmr-large-29L`)

An XLM-RoBERTa Large variant pre-trained with a focus on African languages, making it well-suited for isiZulu. It was downloaded from Google Drive and stored locally.

| Property            | Value         |
|---------------------|---------------|
| Architecture        | XLM-RoBERTa   |
| Hidden size         | 1024          |
| Attention heads     | 16            |
| Transformer layers  | 24            |
| Parameters          | 559.9M        |
| Vocabulary          | 250,002 tokens|
| Max position embeds | 514           |

The classifier head (`classifier.dense` + `classifier.out_proj`) was randomly initialised and trained from scratch on top of the frozen-then-updated encoder.

---

## Training Configuration

| Hyperparameter           | Value                  |
|--------------------------|------------------------|
| Max sequence length      | 512 tokens             |
| Batch size (per device)  | 8                      |
| Gradient accumulation    | 4 steps (effective 32) |
| Learning rate            | 2e-5                   |
| LR scheduler             | Cosine                 |
| Warmup ratio             | 0.1                    |
| Weight decay             | 0.01                   |
| Max epochs               | 5                      |
| Early stopping patience  | 2 epochs               |
| Best model metric        | F1 (binary)            |
| Mixed precision          | fp16 (auto, if CUDA)   |
| Eval strategy            | Every epoch            |
| Seed                     | 42                     |

**Hardware used:** NVIDIA GeForce RTX 5090 (33.7 GB VRAM)
**Training time:** ~22 minutes (1,334 seconds)
**Throughput:** 186.6 samples/sec · 5.83 steps/sec

---

## Reproducing with Docker

Docker mounts the large model and dataset files at runtime rather than baking them into the image, so the image stays small and portable.

### Prerequisites

- Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA driver ≥ 550
- The `base model/` and `datasets/` directories present in the project root

### Build and run

```bash
docker compose up --build
```

This will:
1. Build from `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04`
2. Install PyTorch 2.3.1 (cu121)
3. Install pinned Python dependencies from `requirements.txt`
4. Mount `base model/`, `datasets/`, and `finetuned_classifier/` as volumes
5. Run `scripts/finetune.py`

The fine-tuned model is written to `./finetuned_classifier/final_model/` on the host.

### Manual (no Docker)

```bash
# 1. Install PyTorch
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# 2. Install remaining dependencies
pip install -r requirements.txt

# 3. Run training
python3 scripts/finetune.py
```

---

## Inference

Load the saved model from `finetuned_classifier/final_model/` for inference:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "finetuned_classifier/final_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    label_id = logits.argmax().item()
    labels = {0: "human-written", 1: "machine-written"}
    return labels[label_id]

print(predict("Umuntu wakhe uhambile"))
```

---

## Dependencies

| Package        | Version  |
|----------------|----------|
| transformers   | 4.47.1   |
| datasets       | 4.8.5    |
| accelerate     | 1.13.0   |
| scikit-learn   | 1.7.2    |
| numpy          | >=1.24   |
