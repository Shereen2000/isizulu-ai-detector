import os
import json
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores
import pandas as pd
from datetime import datetime

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
MODEL_PATH  = os.path.join(PROJECT_DIR, "finetuned_model", "final_model")
TEST_PATH   = os.path.join(PROJECT_DIR, "datasets", "test.jsonl")

SPLITS = {
    "train": os.path.join(PROJECT_DIR, "datasets", "train.jsonl"),
    "eval" : os.path.join(PROJECT_DIR, "datasets", "eval.jsonl"),
    "test" : TEST_PATH,
}

OUTPUT_DIR       = os.path.join(PROJECT_DIR, "model_audit")
SHAP_DIR         = os.path.join(OUTPUT_DIR, "shap")
CLEANLAB_DIR     = os.path.join(OUTPUT_DIR, "cleanlab")
os.makedirs(SHAP_DIR,     exist_ok=True)
os.makedirs(CLEANLAB_DIR, exist_ok=True)

MAX_LENGTH = 512
BATCH_SIZE = 16
N_CORRECT  = 200
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("MODEL AUDIT — SHAP EXPLAINABILITY + CLEANLAB LABEL QUALITY")
print(f"  Device : {DEVICE}")
print("=" * 70)

# ── Load model (shared) ───────────────────────────────────────────────────────
print(f"\nLoading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()
print("  Model loaded.")

# ── Shared helpers ────────────────────────────────────────────────────────────
def load_jsonl(path):
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            texts.append(s["text"])
            labels.append(int(s["label"]))
    return texts, np.array(labels)

def get_probs(texts):
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = tokenizer(
                texts[i:i+BATCH_SIZE],
                truncation=True, padding=True,
                max_length=MAX_LENGTH, return_tensors="pt"
            ).to(DEVICE)
            logits = model(**batch).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            if (i // BATCH_SIZE) % 20 == 0:
                print(f"      {min(i+BATCH_SIZE, len(texts))}/{len(texts)} done")
    return np.concatenate(all_probs, axis=0)   # (n, 2)

label_map = {0: "human", 1: "machine"}

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("PART 1 — SHAP EXPLAINABILITY")
print(f"{'='*70}")

print("\nLoading test set and running inference...")
test_texts, test_labels = load_jsonl(TEST_PATH)
test_probs = get_probs(test_texts)
test_preds = np.argmax(test_probs, axis=1)

tp_idx = np.where((test_labels == 1) & (test_preds == 1))[0]
tn_idx = np.where((test_labels == 0) & (test_preds == 0))[0]
fp_idx = np.where((test_labels == 0) & (test_preds == 1))[0]
fn_idx = np.where((test_labels == 1) & (test_preds == 0))[0]
print(f"  TP={len(tp_idx)}  TN={len(tn_idx)}  FP={len(fp_idx)}  FN={len(fn_idx)}")
print("\n  Per-class classification report (test set):")
print(classification_report(test_labels, test_preds,
                             target_names=["human", "machine"], digits=4))

rng      = np.random.default_rng(42)
n_each   = N_CORRECT // 2
sel_tp   = rng.choice(tp_idx, size=min(n_each, len(tp_idx)), replace=False)
sel_tn   = rng.choice(tn_idx, size=min(n_each, len(tn_idx)), replace=False)
sel_fp   = fp_idx
sel_fn   = fn_idx

explain_idx    = np.concatenate([sel_tp, sel_tn, sel_fp, sel_fn])
explain_texts  = [test_texts[i] for i in explain_idx]
explain_labels = [test_labels[i] for i in explain_idx]
explain_preds  = [test_preds[i]  for i in explain_idx]

print(f"\nExplaining {len(explain_texts)} samples: "
      f"{len(sel_tp)} TP | {len(sel_tn)} TN | {len(sel_fp)} FP | {len(sel_fn)} FN")

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
    return torch.softmax(logits, dim=-1).cpu().numpy()

print("\nInitialising SHAP explainer...")
masker    = shap.maskers.Text(tokenizer)
explainer = shap.Explainer(predict, masker, output_names=["human", "machine"])

print(f"Computing SHAP values — this may take a while...")
shap_values = explainer(explain_texts)

print("Saving global bar plot...")
plt.figure()
shap.plots.bar(shap_values[:, :, 1], max_display=20, show=False)
plt.title("Global Feature Importance — Machine Class")
plt.tight_layout()
plt.savefig(os.path.join(SHAP_DIR, "global_bar.png"), dpi=150, bbox_inches="tight")
plt.close()

print("Saving beeswarm plot...")
try:
    plt.figure()
    shap.plots.beeswarm(shap_values[:, :, 1], max_display=20, show=False)
    plt.title("SHAP Beeswarm — Machine Class (all explained samples)")
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_DIR, "beeswarm.png"), dpi=150, bbox_inches="tight")
    plt.close()
except ValueError as e:
    plt.close()
    print(f"  Beeswarm skipped (ragged token lengths across samples): {e}")

print("Saving per-sample waterfall plots...")
categories = (
    [(i, "TP") for i in range(len(sel_tp))] +
    [(i + len(sel_tp), "TN") for i in range(len(sel_tn))] +
    [(i + len(sel_tp) + len(sel_tn), "FP") for i in range(len(sel_fp))] +
    [(i + len(sel_tp) + len(sel_tn) + len(sel_fp), "FN") for i in range(len(sel_fn))]
)

for idx, category in categories:
    true_l = label_map[explain_labels[idx]]
    pred_l = label_map[explain_preds[idx]]
    plt.figure()
    shap.plots.waterfall(shap_values[idx, :, 1], max_display=15, show=False)
    plt.title(f"[{category}] True={true_l}  Pred={pred_l}")
    plt.tight_layout()
    fname = f"waterfall_{category}_{idx:03d}_true{true_l}_pred{pred_l}.png"
    plt.savefig(os.path.join(SHAP_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close()

print("Saving text plots for error cases...")
error_start = len(sel_tp) + len(sel_tn)
for i in range(len(sel_fp) + len(sel_fn)):
    idx      = error_start + i
    category = "FP" if i < len(sel_fp) else "FN"
    true_l   = label_map[explain_labels[idx]]
    pred_l   = label_map[explain_preds[idx]]
    html     = shap.plots.text(shap_values[idx, :, 1], display=False)
    fname    = f"text_{category}_{idx:03d}_true{true_l}_pred{pred_l}.html"
    with open(os.path.join(SHAP_DIR, fname), "w", encoding="utf-8") as f:
        f.write(html)

print(f"\nSHAP outputs saved to: {SHAP_DIR}")

# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — CLEANLAB LABEL QUALITY AUDIT
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("PART 2 — CLEANLAB LABEL QUALITY AUDIT")
print(f"{'='*70}")

summary_rows = []

for split_name, split_path in SPLITS.items():
    print(f"\n{'─'*70}")
    print(f"Auditing: {split_name}.jsonl")
    print(f"{'─'*70}")

    if split_name == "test":
        texts  = test_texts
        labels = test_labels
        probs  = test_probs
        print(f"  Samples : {len(texts)}  (reusing test inference from SHAP step)")
    else:
        texts, labels = load_jsonl(split_path)
        print(f"  Samples : {len(texts)}  (label 0: {(labels==0).sum()}, label 1: {(labels==1).sum()})")
        print(f"  Running inference...")
        probs = get_probs(texts)

    label_issues_idx = find_label_issues(
        labels=labels,
        pred_probs=probs,
        return_indices_ranked_by="self_confidence",
    )
    quality_scores = get_label_quality_scores(labels=labels, pred_probs=probs)

    n_issues      = len(label_issues_idx)
    issues_set    = set(label_issues_idx)
    pct           = 100 * n_issues / len(texts)
    print(f"\n  Label issues found : {n_issues} / {len(texts)}  ({pct:.2f}%)")
    print(f"  Mean quality score : {quality_scores.mean():.4f}")
    print(f"  Min  quality score : {quality_scores.min():.4f}")

    preds_local  = np.argmax(probs, axis=1)
    conf_scores  = probs[np.arange(len(probs)), preds_local]
    print(f"\n  Prediction confidence (in predicted class):")
    print(f"    mean={conf_scores.mean():.4f}  median={np.median(conf_scores):.4f}  "
          f"std={conf_scores.std():.4f}  min={conf_scores.min():.4f}")
    print(f"    >0.99: {(conf_scores > 0.99).sum()}  "
          f">0.95: {(conf_scores > 0.95).sum()}  "
          f"0.90–0.95: {((conf_scores >= 0.90) & (conf_scores <= 0.95)).sum()}  "
          f"<0.90: {(conf_scores < 0.90).sum()}")

    df = pd.DataFrame({
        "index"            : range(len(texts)),
        "given_label"      : labels,
        "given_label_name" : ["human" if l == 0 else "machine" for l in labels],
        "predicted_label"  : preds_local,
        "predicted_name"   : ["human" if p == 0 else "machine" for p in preds_local],
        "prob_human"       : probs[:, 0].round(4),
        "prob_machine"     : probs[:, 1].round(4),
        "confidence"       : conf_scores.round(4),
        "quality_score"    : quality_scores.round(4),
        "flagged"          : [i in issues_set for i in range(len(texts))],
        "text_snippet"     : [t[:120] for t in texts],
    })

    df.to_csv(os.path.join(CLEANLAB_DIR, f"{split_name}_quality.csv"), index=False)
    df[df["flagged"]].to_csv(
        os.path.join(CLEANLAB_DIR, f"{split_name}_flagged.csv"), index=False)

    if n_issues > 0:
        n_human_flagged   = int((df[df["flagged"]]["given_label"] == 0).sum())
        n_machine_flagged = int((df[df["flagged"]]["given_label"] == 1).sum())
        print(f"\n  Flagged by class: human={n_human_flagged}  machine={n_machine_flagged}")
        print(f"\n  Top 10 most suspicious:")
        top10 = df[df["flagged"]].nsmallest(10, "quality_score")
        for _, row in top10.iterrows():
            print(f"    idx={row['index']:5d}  given={row['given_label_name']:7s}  "
                  f"pred={row['predicted_name']:7s}  "
                  f"p_human={row['prob_human']:.3f}  p_machine={row['prob_machine']:.3f}  "
                  f"quality={row['quality_score']:.3f}")
            print(f"      text: {row['text_snippet'][:80]}...")

    summary_rows.append({
        "split"                : split_name,
        "total_samples"        : len(texts),
        "label_issues"         : n_issues,
        "issue_pct"            : round(pct, 2),
        "human_flagged"        : int((df[df["flagged"]]["given_label"] == 0).sum()),
        "machine_flagged"      : int((df[df["flagged"]]["given_label"] == 1).sum()),
        "mean_quality_score"   : round(float(quality_scores.mean()), 4),
        "min_quality_score"    : round(float(quality_scores.min()),  4),
        "mean_confidence"      : round(float(conf_scores.mean()),    4),
        "pct_conf_above_99"    : round(100 * float((conf_scores > 0.99).mean()), 2),
        "pct_conf_below_90"    : round(100 * float((conf_scores < 0.90).mean()), 2),
    })

print(f"\n{'='*70}")
print("CLEANLAB SUMMARY")
print(f"{'='*70}")
print(pd.DataFrame(summary_rows).to_string(index=False))

with open(os.path.join(CLEANLAB_DIR, "summary.json"), "w") as f:
    json.dump(summary_rows, f, indent=2)

print(f"\nCleanlab outputs saved to: {CLEANLAB_DIR}")

# ── Final ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("ALL DONE")
print(f"{'='*70}")
print(f"\nOutputs saved to: {OUTPUT_DIR}")
print(f"  shap/")
print(f"    global_bar.png          — top tokens driving machine classification")
print(f"    beeswarm.png            — distribution of token impacts across all samples")
print(f"    waterfall_*.png         — per-sample token contributions")
print(f"    text_FP/FN_*.html       — highlighted token view for error cases")
print(f"  cleanlab/")
print(f"    <split>_quality.csv     — quality + confidence score for every sample")
print(f"    <split>_flagged.csv     — flagged/suspicious samples with predicted labels")
print(f"    summary.json            — overall stats per split incl. confidence distribution")
print(f"\nDone at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
