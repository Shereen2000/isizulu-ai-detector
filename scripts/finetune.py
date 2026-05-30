import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
import json
from datetime import datetime

import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

MODEL_PATH   = os.path.join(PROJECT_DIR, "base_model")
TRAIN_PATH   = os.path.join(PROJECT_DIR, "datasets", "train.jsonl")
EVAL_PATH    = os.path.join(PROJECT_DIR, "datasets", "eval.jsonl")
OUTPUT_DIR   = os.path.join(PROJECT_DIR, "finetuned_model")

MAX_LENGTH    = 512
BATCH_SIZE    = 8
GRAD_ACCUM    = 4      
LEARNING_RATE = 2e-5
EPOCHS        = 5
SEED          = 42

# ============================================================================
# LOAD DATASET
# ============================================================================

print("ISIZULU AI-DETECTION CLASSIFIER  —  FINE-TUNING")
print("  Label 1 = machine-written  |  Label 0 = human-written")

def load_jsonl(path):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            texts.append(sample["text"])
            labels.append(int(sample["label"]))
    return texts, labels

print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading datasets")

from datasets import ClassLabel
label_feature = ClassLabel(names=["human", "machine"])

train_texts, train_labels = load_jsonl(TRAIN_PATH)
eval_texts,  eval_labels  = load_jsonl(EVAL_PATH)

train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels}).cast_column("label", label_feature)
eval_dataset  = Dataset.from_dict({"text": eval_texts,  "label": eval_labels}).cast_column("label", label_feature)

print(f"   Training samples  : {len(train_dataset)}  (0: {train_labels.count(0)}, 1: {train_labels.count(1)})")
print(f"   Evaluation samples: {len(eval_dataset)}   (0: {eval_labels.count(0)}, 1: {eval_labels.count(1)})")

# ============================================================================
# LOAD TOKENIZER & MODEL
# ============================================================================

print(f"\nLoading model from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print("   Tokenizer loaded")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=2,
    id2label={0: "human", 1: "machine"},
    label2id={"human": 0, "machine": 1},
    ignore_mismatched_sizes=True,
)
print(f"   Model loaded  ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters)")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Device: {device}")

# ============================================================================
# TOKENIZE
# ============================================================================

print(f"\nTokenizing (max_length={MAX_LENGTH})...")

def preprocess(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )

train_dataset = train_dataset.map(preprocess, batched=True)
eval_dataset  = eval_dataset.map(preprocess, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
eval_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

rng = np.random.default_rng(SEED)
subset_idx = rng.choice(len(train_dataset), size=min(2000, len(train_dataset)), replace=False).tolist()
train_eval_dataset = train_dataset.select(subset_idx)

print(f"   Tokenization complete")

# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs_machine = (exp / exp.sum(axis=-1, keepdims=True))[:, 1]

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    return {
        # Performance
        "accuracy"       : accuracy_score(labels, preds),
        "f1"             : f1_score(labels, preds, average="binary"),
        "precision"      : precision_score(labels, preds, average="binary"),
        "recall"         : recall_score(labels, preds, average="binary"),
        # Per-class
        "f1_human"       : f1_score(labels, preds, pos_label=0, average="binary"),
        "f1_machine"     : f1_score(labels, preds, pos_label=1, average="binary"),
        "precision_human": precision_score(labels, preds, pos_label=0, average="binary"),
        "recall_human"   : recall_score(labels, preds, pos_label=0, average="binary"),
        # Calibration / ranking
        "roc_auc"        : roc_auc_score(labels, probs_machine),
        "mcc"            : matthews_corrcoef(labels, preds),
        # Confusion matrix counts 
        "tp"             : int(tp),   # machine predicted machine  
        "tn"             : int(tn),   # human   predicted human    
        "fp"             : int(fp),   # human   predicted machine  
        "fn"             : int(fn),   # machine predicted human    
    }

# ============================================================================
# TRAINING ARGUMENTS
# ============================================================================

print("\nConfiguring training arguments...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    warmup_ratio=0.1,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_eval_f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    gradient_checkpointing=True,
    seed=SEED,
    push_to_hub=False,
    report_to="none",
)

# ============================================================================
# TRAINER
# ============================================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset={"eval": eval_dataset, "train_subset": train_eval_dataset},
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print(f"\nStarting training ({EPOCHS} epochs, early stopping patience=2)\n")
trainer.train()

# ============================================================================
# SAVE & EVALUATE
# ============================================================================

print("TRAINING COMPLETE")

final_path = os.path.join(OUTPUT_DIR, "final_model")
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)
print(f"Final model saved to: {final_path}")

print("\nRunning final evaluation on eval set...")
eval_results        = trainer.evaluate(eval_dataset=eval_dataset,        metric_key_prefix="eval")
print("\nRunning final evaluation on train subset...")
train_eval_results  = trainer.evaluate(eval_dataset=train_eval_dataset,  metric_key_prefix="train")

print("\n── Eval set ──────────────────────────────────────")
print(f"   Loss      : {eval_results['eval_loss']:.4f}")
print(f"   Accuracy  : {eval_results['eval_accuracy']:.4f}")
print(f"   F1        : {eval_results['eval_f1']:.4f}")
print(f"   Precision : {eval_results['eval_precision']:.4f}")
print(f"   Recall    : {eval_results['eval_recall']:.4f}")
print(f"   ROC-AUC   : {eval_results['eval_roc_auc']:.4f}")
print(f"   MCC       : {eval_results['eval_mcc']:.4f}")
print(f"   TP/TN/FP/FN: {eval_results['eval_tp']} / {eval_results['eval_tn']} / {eval_results['eval_fp']} / {eval_results['eval_fn']}")

print("\n── Train subset (overfitting check) ──────────────")
print(f"   Loss      : {train_eval_results['train_loss']:.4f}")
print(f"   Accuracy  : {train_eval_results['train_accuracy']:.4f}")
print(f"   F1        : {train_eval_results['train_f1']:.4f}")
print(f"   ROC-AUC   : {train_eval_results['train_roc_auc']:.4f}")
print(f"   MCC       : {train_eval_results['train_mcc']:.4f}")

gap = train_eval_results["train_f1"] - eval_results["eval_f1"]
print(f"\n   Train-Eval F1 gap : {gap:+.4f}  {'(possible overfitting)' if gap > 0.05 else '(looks healthy)'}")

all_metrics = {**eval_results, **train_eval_results}
metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(all_metrics, f, indent=2)
print(f"\nMetrics saved to: {metrics_path}")

print(f"\nDone at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Output directory: {OUTPUT_DIR}")
