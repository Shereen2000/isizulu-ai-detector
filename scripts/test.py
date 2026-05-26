import os
import json
from datetime import datetime

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix,
)

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_PATH  = os.path.join(PROJECT_DIR, "finetuned_model", "final_model")
TEST_PATH   = os.path.join(PROJECT_DIR, "datasets", "test3.jsonl")
OUTPUT_PATH = os.path.join(PROJECT_DIR, "finetuned_model", "test3_metrics.json")

BATCH_SIZE = 16
MAX_LENGTH = 512

print("=" * 70)
print("ISIZULU AI-DETECTION CLASSIFIER  —  TEST SET EVALUATION")
print("  Label 1 = machine-written  |  Label 0 = human-written")
print("=" * 70)

# ── Load test data ────────────────────────────────────────────────────────────
print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading test set from {TEST_PATH}")
texts, labels = [], []
with open(TEST_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        sample = json.loads(line)
        texts.append(sample["text"])
        labels.append(int(sample["label"]))

labels = np.array(labels)
print(f"   Samples: {len(texts)}  (0: {(labels==0).sum()}, 1: {(labels==1).sum()})")

# ── Load model & tokenizer ────────────────────────────────────────────────────
print(f"\nLoading model from {MODEL_PATH}")
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()
print(f"   Device: {device}")

# ── Run inference in batches ──────────────────────────────────────────────────
print(f"\nRunning inference (batch_size={BATCH_SIZE})...")
all_logits = []
with torch.no_grad():
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)
        logits = model(**inputs).logits
        all_logits.append(logits.cpu().float().numpy())
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"   {min(i + BATCH_SIZE, len(texts))}/{len(texts)} samples processed")

all_logits = np.concatenate(all_logits, axis=0)

# ── Compute metrics ───────────────────────────────────────────────────────────
preds = np.argmax(all_logits, axis=-1)
exp   = np.exp(all_logits - all_logits.max(axis=-1, keepdims=True))
probs_machine = (exp / exp.sum(axis=-1, keepdims=True))[:, 1]

tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

metrics = {
    "accuracy"        : round(accuracy_score(labels, preds),                              4),
    "f1_human"        : round(f1_score(labels, preds, pos_label=0, average="binary"),     4),
    "f1_machine"      : round(f1_score(labels, preds, pos_label=1, average="binary"),     4),
    "precision_human" : round(precision_score(labels, preds, pos_label=0, average="binary"), 4),
    "precision_machine": round(precision_score(labels, preds, pos_label=1, average="binary"), 4),
    "recall_human"    : round(recall_score(labels, preds, pos_label=0, average="binary"), 4),
    "recall_machine"  : round(recall_score(labels, preds, pos_label=1, average="binary"), 4),
    "roc_auc"         : round(roc_auc_score(labels, probs_machine),                       4),
    "mcc"             : round(matthews_corrcoef(labels, preds),                           4),
    "tp"              : int(tp),
    "tn"              : int(tn),
    "fp"              : int(fp),
    "fn"              : int(fn),
    "total_samples"   : len(texts),
}

# ── Print results ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TEST SET RESULTS")
print("=" * 70)
print(f"   Accuracy          : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
print(f"   ROC-AUC           : {metrics['roc_auc']:.4f}")
print(f"   MCC               : {metrics['mcc']:.4f}")
print()
print(f"   {'':20s}  {'Human':>8}  {'Machine':>8}")
print(f"   {'F1':20s}  {metrics['f1_human']:>8.4f}  {metrics['f1_machine']:>8.4f}")
print(f"   {'Precision':20s}  {metrics['precision_human']:>8.4f}  {metrics['precision_machine']:>8.4f}")
print(f"   {'Recall':20s}  {metrics['recall_human']:>8.4f}  {metrics['recall_machine']:>8.4f}")
print()
print(f"   Confusion matrix:")
print(f"                    Predicted")
print(f"                  Human   Machine")
print(f"   Actual Human    {tn:4d}     {fp:4d}    (TN / FP)")
print(f"   Actual Machine  {fn:4d}     {tp:4d}    (FN / TP)")

# ── Save ──────────────────────────────────────────────────────────────────────
with open(OUTPUT_PATH, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved to: {OUTPUT_PATH}")
print(f"Done at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
