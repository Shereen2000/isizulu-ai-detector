import re
import os
import csv

INPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw data", "isolezwe-corpus-clean.csv")
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "isolezwe-corpus-clean.txt")


def clean_text(text):
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"Isihloko: ", "", text)
    text = re.sub(r"\*", "", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    with open(INPUT_FILE, encoding="utf-8") as f_in, open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        count = 0
        for row in reader:
            cleaned = clean_text(row["text"])
            if cleaned:
                f_out.write(cleaned + "\n")
                count += 1
    print(f"Done. {count} lines written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
