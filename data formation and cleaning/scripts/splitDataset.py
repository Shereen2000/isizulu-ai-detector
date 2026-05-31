import os
import random

_dir = os.path.dirname(os.path.abspath(__file__))
DATASETS   = os.path.join(_dir, "../data")
INPUT_FILE = os.path.join(DATASETS, "final AI dataset pre chunks combined.txt")

TRAIN_RATIO = 0.80
EVAL_RATIO  = 0.10
TEST_RATIO  = 0.10
SEED        = 42

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    samples = [l.strip() for l in f if l.strip()]

random.seed(SEED)
random.shuffle(samples)

total      = len(samples)
train_end  = int(total * TRAIN_RATIO)
eval_end   = train_end + int(total * EVAL_RATIO)

train = samples[:train_end]
eval_ = samples[train_end:eval_end]
test  = samples[eval_end:]

splits = {
    "AI_train.txt": train,
    "AI_eval.txt":  eval_,
    "AI_test.txt":  test,
}

for filename, data in splits.items():
    path = os.path.join(DATASETS, filename)
    with open(path, "w", encoding="utf-8") as f:
        for line in data:
            f.write(line + "\n")

print(f"Total samples : {total}")
print(f"Train         : {len(train)}  ({len(train)/total*100:.1f}%)")
print(f"Eval          : {len(eval_)}   ({len(eval_)/total*100:.1f}%)")
print(f"Test          : {len(test)}   ({len(test)/total*100:.1f}%)")
print(f"\nFiles written to: {DATASETS}")
