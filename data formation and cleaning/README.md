# Data Formation and Cleaning

This folder contains the raw data sources, intermediate files, and scripts used to build the final training/eval/test datasets for the isiZulu AI-detection classifier.

---

## Folder Structure

```
data formation and cleaning/
├── raw data/                         # Original unprocessed source corpora
│   ├── CORP.NCHLT.zu.CLEAN.2.0.txt   # NCHLT isiZulu human corpus
│   └── isolezwe-corpus-clean.csv     # Isolezwe newspaper articles
├── data/                             # Intermediate and final processed files
│   ├── final Human dataset pre chunks.txt
│   ├── final AI dataset pre chunks v1.txt
│   ├── final AI dataset pre chunks v2.txt
│   ├── final AI dataset pre chunks combined.txt
│   ├── AI_train / eval / test chunked.txt
│   ├── Human_train / eval / test chunked.txt
│   ├── train.jsonl / eval.jsonl / test.jsonl
│   └── isizulu_dataset.jsonl
└── scripts/                          # Pipeline scripts (run in order below)
```

---

## Data Sources

**Human text (label 0)**
- **NCHLT isiZulu Corpus** — a large cleaned isiZulu text corpus from the South African Centre for Digital Language Resources
- **Isolezwe** — isiZulu newspaper articles scraped and cleaned from the Isolezwe corpus CSV

**AI-generated text (label 1)**
- Generated using **GPT-4o-mini** prompted to produce isiZulu text across varied topics and styles, with randomised temperatures (0.2, 0.5, 0.8, 1.0) and a 30% chance of paraphrasing to increase diversity

---

## Common Text Cleaner

All three data sources (NCHLT, Isolezwe, GPT-4o-mini) pass through the same `clean_text` function before anything else happens. The steps are identical across all scripts:

| Step | What it does |
|------|-------------|
| Strip `**` and `*` | Removes markdown bold/italic/list symbols that appear in LLM output and raw corpora |
| Strip `Isihloko: ` | Removes the Zulu heading label that prefixes Isolezwe articles and sometimes appears in GPT output |
| Collapse `\n+` → space | Flattens multi-line blocks into a single line |
| Collapse `\s+` → single space | Normalises all whitespace |
| `.strip()` | Removes leading/trailing whitespace |

The NCHLT script has two extra steps on top of these:
- Removes a specific UTF-8 mojibake artefact (`Ã¯ÂÅ¸Ã¯â‚¬Â`) that appears in the raw corpus file due to encoding issues
- Splits on `<fn>...</fn>` tags to strip footnote markers embedded in the corpus

This means every sample in the final dataset — regardless of whether it's human or AI — went through the same normalisation before chunking and labelling.

---

## Pipeline

The scripts are meant to be run in this order:

### 1. `cleanHumanDataset.py`
Reads `CORP.NCHLT.zu.CLEAN.2.0.txt`, applies the common cleaner plus NCHLT-specific artefact removal, and outputs a single clean block of human isiZulu text.

### 2. `isolezweCorpusToText.py`
Reads `isolezwe-corpus-clean.csv`, applies the common cleaner to each article, and writes plain text to `data/`.

### 3. `extractor.py`
Calls the OpenAI API (GPT-4o-mini) to generate AI isiZulu text samples, applies the common cleaner to each response, and writes output to `isizulu_dataset.jsonl` in `data/`.

### 4. `translatorToEnlish.py` *(optional — data augmentation)*
Translates chunked human isiZulu text to English via GPT-4o-mini. Used as an intermediate step for back-translation augmentation. Saves a checkpoint every 50 samples so it can resume if interrupted.

### 5. `translatorToZulu.py` *(optional — data augmentation)*
Takes the English translations from step 4 and translates them back to isiZulu. The round-trip introduces natural variation, producing additional training samples that are still human-origin but paraphrased.

### 6. `slidingWindowChunker.py`
Chunks long text files into segments of up to 512 tokens using the AfroXLMR tokenizer with a 3-sentence overlap between chunks. Caps at 36 chunks per source sample to avoid any single source dominating.

### 7. `splitDataset.py`
Shuffles and splits the combined AI dataset into train (80%) / eval (10%) / test (10%) text files. Seed is fixed at 42 for reproducibility.

### 8. `buildJsonl.py`
Interleaves AI and human chunked text files, assigns labels (`0` = human, `1` = AI), shuffles, and writes the final `.jsonl` files used for training.

---

## Final Output

The `train.jsonl`, `eval.jsonl`, and `test.jsonl` in `data/` are copied into `datasets/` at the project root for use by `scripts/finetune.py` and `scripts/test.py`.

Each line is a JSON object:

```json
{"text": "Umculo udlala indima ebalulekile", "label": 0}
{"text": "Ingxenye yesibili yezinhlobo zezilimi", "label": 1}
```
