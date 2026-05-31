import os
import json
import random

_dir     = os.path.dirname(os.path.abspath(__file__))
SRC      = os.path.join(_dir, "../data")
OUT      = os.path.join(_dir, "../data")
SEED     = 42

SPLITS = {
    "test":  ("isizulu_AI.txt",  "x.txt",  "test.jsonl"),
}

def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def interleave_and_shuffle(ai_lines, human_lines, seed):
    records = []

    # alternate line by line
    for a, h in zip(ai_lines, human_lines):
        records.append({"text": a, "label": 1})   # 1 = AI
        records.append({"text": h, "label": 0})   # 0 = Human

    # append leftover from the longer file at the bottom
    shorter = min(len(ai_lines), len(human_lines))
    for line in ai_lines[shorter:]:
        records.append({"text": line, "label": 1})
    for line in human_lines[shorter:]:
        records.append({"text": line, "label": 0})

    random.seed(seed)
    random.shuffle(records)
    return records

for split, (ai_file, human_file, out_file) in SPLITS.items():
    ai_lines    = read_lines(os.path.join(SRC, ai_file))
    human_lines = read_lines(os.path.join(SRC, human_file))

    records = interleave_and_shuffle(ai_lines, human_lines, SEED)

    out_path = os.path.join(OUT, out_file)
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    ai_count    = sum(1 for r in records if r["label"] == 1)
    human_count = sum(1 for r in records if r["label"] == 0)
    print(f"{split:>5} → {len(records):>5} samples  (AI: {ai_count}, Human: {human_count})  →  {out_file}")
