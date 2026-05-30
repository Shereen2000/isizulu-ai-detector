import os
import json
from datetime import datetime

import gdown
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

FINETUNED_SAFETENSORS_FILE_ID = "1fBFNnUb3xrfPpOJ_p6CDDQE4zQZq2FEM"

def download_finetuned_weights():
    dest = os.path.join(MODEL_PATH, "model.safetensors")
    if os.path.exists(dest):
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] model.safetensors already exists, skipping download")
        return
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Downloading model.safetensors into {MODEL_PATH}")
    gdown.download(id=FINETUNED_SAFETENSORS_FILE_ID, output=dest, quiet=False)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Download complete")

download_finetuned_weights()

BATCH_SIZE = 16
MAX_LENGTH = 512

TEST_FILES = [
    ("test.jsonl",  "test_metrics.json"),
    ("test2.jsonl", "test2_metrics.json"),
    ("test3.jsonl", "test3_metrics.json"),
]

print("ISIZULU AI-DETECTION CLASSIFIER  —  TEST SET EVALUATION")
print("  Label 1 = machine-written  |  Label 0 = human-written")

# Load model & tokenizer once
print(f"\nLoading model from {MODEL_PATH}")
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()
print(f"   Device: {device}")

for test_file, metrics_file in TEST_FILES:
    test_path   = os.path.join(PROJECT_DIR, "datasets", test_file)
    output_path = os.path.join(PROJECT_DIR, "finetuned_model", metrics_file)

    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Evaluating {test_file}")

    # Load test data
    texts, labels = [], []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            texts.append(sample["text"])
            labels.append(int(sample["label"]))

    labels = np.array(labels)
    print(f"   Samples: {len(texts)}  (0: {(labels==0).sum()}, 1: {(labels==1).sum()})")

    # Run inference in batches
    print(f"   Running inference (batch_size={BATCH_SIZE})...")
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

    # Compute metrics
    preds = np.argmax(all_logits, axis=-1)
    exp   = np.exp(all_logits - all_logits.max(axis=-1, keepdims=True))
    probs_machine = (exp / exp.sum(axis=-1, keepdims=True))[:, 1]

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    metrics = {
        "accuracy"         : round(accuracy_score(labels, preds),                               4),
        "f1_human"         : round(f1_score(labels, preds, pos_label=0, average="binary"),      4),
        "f1_machine"       : round(f1_score(labels, preds, pos_label=1, average="binary"),      4),
        "precision_human"  : round(precision_score(labels, preds, pos_label=0, average="binary"), 4),
        "precision_machine": round(precision_score(labels, preds, pos_label=1, average="binary"), 4),
        "recall_human"     : round(recall_score(labels, preds, pos_label=0, average="binary"),  4),
        "recall_machine"   : round(recall_score(labels, preds, pos_label=1, average="binary"),  4),
        "roc_auc"          : round(roc_auc_score(labels, probs_machine),                        4),
        "mcc"              : round(matthews_corrcoef(labels, preds),                            4),
        "tp"               : int(tp),
        "tn"               : int(tn),
        "fp"               : int(fp),
        "fn"               : int(fn),
        "total_samples"    : len(texts),
    }

    # Print results
    print(f"\n   Results for {test_file}:")
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

    # Save
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n   Metrics saved to: {output_path}")

print(f"\n{'='*60}")
print(f"All evaluations complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
