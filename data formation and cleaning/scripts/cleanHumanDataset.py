import os
import re

_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_dir, "../raw data/CORP.NCHLT.zu.CLEAN.2.0.txt"), "r", encoding="utf-8-sig") as f:
    lines = f.read().splitlines()

result = " ".join(line for line in lines if line)

def clean_text(text):
    text = text.replace("Ã¯ÂÅ¸Ã¯â‚¬Â", "")
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"\*", "", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

result = clean_text(result)

chunks = re.split(r"<fn>[^<]*</fn>", result)

print(len(chunks))

output_path = os.path.join(_dir, "../data_/zuluhumantext_cleaned.txt")
with open(output_path, "w", encoding="utf-8") as f:
    for chunk in chunks:
        stripped = chunk.strip()
        if stripped:
            f.write(stripped + "\n")