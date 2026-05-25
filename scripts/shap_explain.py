import os
import json
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_PATH  = os.path.join(PROJECT_DIR, "finetuned_classifier", "final_model")
TEST_PATH   = os.path.join(PROJECT_DIR, "datasets", "test.jsonl")
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "shap_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_LENGTH  = 512
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── How many samples per category to explain ─────────────────────────────────
N_CORRECT   = 10   # true positives + true negatives
N_ERRORS    = 10   # all false positives + false negatives (may be fewer than 10)

print("=" * 70)
print("SHAP EXPLAINABILITY  —  ISIZULU AI-DETECTION CLASSIFIER")
print("=" * 70)

# ── Load model ────────────────────────────────────────────────────────────────
print(f"\nLoading model from {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()
print(f"   Device: {DEVICE}")

# ── Load test data & run inference to find FP/FN ─────────────────────────────
print("\nLoading test set and identifying error cases...")
texts, labels = [], []
with open(TEST_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        s = json.loads(line)
        texts.append(s["text"])
        labels.append(int(s["label"]))

labels = np.array(labels)

# Run full inference to get predictions
all_preds = []
with torch.no_grad():
    for i in range(0, len(texts), 32):
        batch = tokenizer(
            texts[i:i+32], truncation=True, padding=True,
            max_length=MAX_LENGTH, return_tensors="pt"
        ).to(DEVICE)
        logits = model(**batch).logits
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())

all_preds = np.array(all_preds)

# Categorise samples
tp_idx = np.where((labels == 1) & (all_preds == 1))[0]
tn_idx = np.where((labels == 0) & (all_preds == 0))[0]
fp_idx = np.where((labels == 0) & (all_preds == 1))[0]
fn_idx = np.where((labels == 1) & (all_preds == 0))[0]

print(f"   TP={len(tp_idx)}  TN={len(tn_idx)}  FP={len(fp_idx)}  FN={len(fn_idx)}")

# Build explain set: all errors + balanced sample of correct ones
rng = np.random.default_rng(42)
n_each = N_CORRECT // 2
selected_tp = rng.choice(tp_idx, size=min(n_each, len(tp_idx)), replace=False)
selected_tn = rng.choice(tn_idx, size=min(n_each, len(tn_idx)), replace=False)
selected_fp = fp_idx                  # explain all false positives
selected_fn = fn_idx                  # explain all false negatives

explain_idx = np.concatenate([selected_tp, selected_tn, selected_fp, selected_fn])
explain_texts  = [texts[i] for i in explain_idx]
explain_labels = [labels[i] for i in explain_idx]
explain_preds  = [all_preds[i] for i in explain_idx]

label_map = {0: "human", 1: "machine"}
print(f"\nExplaining {len(explain_texts)} samples:")
print(f"   {len(selected_tp)} TP  |  {len(selected_tn)} TN  |  {len(selected_fp)} FP  |  {len(selected_fn)} FN")

# ── Define prediction function for SHAP ──────────────────────────────────────
def predict(texts_list):
    if isinstance(texts_list, np.ndarray):
        texts_list = texts_list.tolist()
    texts_list = [str(t) for t in texts_list]
    inputs = tokenizer(
        texts_list, truncation=True, padding=True,
        max_length=MAX_LENGTH, return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs   # shape (n, 2) — col 0 = human prob, col 1 = machine prob

# ── Run SHAP ──────────────────────────────────────────────────────────────────
print("\nInitialising SHAP explainer (Partition/TextMasker)...")
masker   = shap.maskers.Text(tokenizer)
explainer = shap.Explainer(predict, masker, output_names=["human", "machine"])

print(f"Computing SHAP values for {len(explain_texts)} samples — this may take a while...\n")
shap_values = explainer(explain_texts)

# ── Save outputs ──────────────────────────────────────────────────────────────

# 1. Global bar plot — mean absolute SHAP for machine class across all samples
print("Saving global bar plot...")
plt.figure()
shap.plots.bar(shap_values[:, :, 1], max_display=20, show=False)
plt.title("Global Feature Importance — Machine Class")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "global_bar.png"), dpi=150, bbox_inches="tight")
plt.close()

# 2. Per-sample waterfall plots
print("Saving per-sample waterfall plots...")
categories = (
    [(i, "TP") for i in range(len(selected_tp))] +
    [(i + len(selected_tp), "TN") for i in range(len(selected_tn))] +
    [(i + len(selected_tp) + len(selected_tn), "FP") for i in range(len(selected_fp))] +
    [(i + len(selected_tp) + len(selected_tn) + len(selected_fp), "FN") for i in range(len(selected_fn))]
)

for idx, category in categories:
    true_label = label_map[explain_labels[idx]]
    pred_label = label_map[explain_preds[idx]]
    plt.figure()
    shap.plots.waterfall(shap_values[idx, :, 1], max_display=15, show=False)
    plt.title(f"[{category}] True={true_label}  Pred={pred_label}")
    plt.tight_layout()
    fname = f"waterfall_{category}_{idx:03d}_true{true_label}_pred{pred_label}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close()

# 3. Text plots for error cases (FP + FN) — highlights tokens inline
print("Saving text plots for error cases...")
error_start = len(selected_tp) + len(selected_tn)
for i in range(len(selected_fp) + len(selected_fn)):
    idx       = error_start + i
    category  = "FP" if i < len(selected_fp) else "FN"
    true_label = label_map[explain_labels[idx]]
    pred_label = label_map[explain_preds[idx]]
    html = shap.plots.text(shap_values[idx, :, 1], display=False)
    fname = f"text_{category}_{idx:03d}_true{true_label}_pred{pred_label}.html"
    with open(os.path.join(OUTPUT_DIR, fname), "w", encoding="utf-8") as f:
        f.write(html)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
print(f"Outputs saved to: {OUTPUT_DIR}")
print(f"  global_bar.png          — top tokens driving machine classification")
print(f"  waterfall_*.png         — per-sample token contributions")
print(f"  text_FP/FN_*.html       — highlighted token view for error cases")
