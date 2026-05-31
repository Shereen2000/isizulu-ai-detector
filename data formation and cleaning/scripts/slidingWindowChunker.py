import os
from transformers import AutoTokenizer
from nltk.tokenize import sent_tokenize

_dir = os.path.dirname(os.path.abspath(__file__))
DATASETS    = os.path.join(_dir, "../data")
INPUT_FILE  = os.path.join(DATASETS, "isizulu_AI.txt")
OUTPUT_FILE = os.path.join(DATASETS, "isizulu_AI chunked.txt")

MAX_TOKENS        = 512
MODEL_NAME        = "Davlan/afro-xlmr-base"
OVERLAP_SENTENCES = 3
MAX_CHUNKS        = 36 # max chunks to keep per sample; set to None for no limit

print(f"Loading tokenizer: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def safe_truncate(sentence):
    tokens = tokenizer.encode(sentence, add_special_tokens=True)
    text   = tokenizer.decode(tokens[:MAX_TOKENS], skip_special_tokens=True).strip()

    # backtrack to last sentence-ending punctuation
    for punct in [".", "!", "?"]:
        pos = text.rfind(punct)
        if pos != -1:
            candidate = text[:pos + 1].strip()
            # verify the decoded result truly fits
            if len(tokenizer.encode(candidate, add_special_tokens=True)) <= MAX_TOKENS:
                return candidate

    # no punctuation found, return raw truncation only if it fits
    if len(tokenizer.encode(text, add_special_tokens=True)) <= MAX_TOKENS and len(text) > 2:
        return text
    return None


def sliding_window_chunk(text, overlap=OVERLAP_SENTENCES, max_chunks=MAX_CHUNKS):
    sentences = sent_tokenize(text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 2]

    chunks  = []
    current = []

    for sentence in sentences:
        if max_chunks and len(chunks) >= max_chunks:
            break

        candidate   = " ".join(current + [sentence])
        token_count = len(tokenizer.encode(candidate, add_special_tokens=True))

        if token_count <= MAX_TOKENS:
            current.append(sentence)
        else:
            if current:
                chunks.append(" ".join(current))
                current = current[-overlap:]

            if max_chunks and len(chunks) >= max_chunks:
                break

            solo_count = len(tokenizer.encode(sentence, add_special_tokens=True))
            if solo_count <= MAX_TOKENS:
                current.append(sentence)
            else:
                truncated = safe_truncate(sentence)
                if truncated:
                    chunks.append(truncated)
                current = []

    if current and (not max_chunks or len(chunks) < max_chunks):
        chunks.append(" ".join(current))

    return chunks


lines_read    = 0
chunks_written = 0

with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

    for line in infile:
        line = line.strip()
        if not line:
            continue

        lines_read += 1
        chunks = sliding_window_chunk(line)

        for chunk in chunks:
            outfile.write(chunk + "\n")
            chunks_written += 1

        print(f"Line {lines_read:>5} → {len(chunks)} chunk(s)")

print()
print(f"Lines read    : {lines_read}")
print(f"Chunks written: {chunks_written}")
print(f"Output        : {OUTPUT_FILE}")
