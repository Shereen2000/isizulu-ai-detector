import json
import os
import time
from datetime import datetime
from openai import OpenAI

client = OpenAI(api_key="api key")

# CONFIG
MODEL = "gpt-4o-mini"
DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
INPUT_FILE = os.path.join(DATASETS_DIR, "zuluhumantext_chunks.txt")
OUTPUT_FILE = os.path.join(DATASETS_DIR, "zuluhumantext_chunks_english.txt")
CHECKPOINT_FILE = os.path.join(DATASETS_DIR, "translation_checkpoint.json")
SAVE_EVERY = 50


# CHECKPOINT
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"translated_count": 0, "output_file": OUTPUT_FILE, "started_at": datetime.now().isoformat()}


def save_checkpoint(checkpoint):
    checkpoint["last_updated"] = datetime.now().isoformat()
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)


# TRANSLATION
def translate_line(text):
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Translate the following Zulu text to English. Output only the translation, nothing else."},
                {"role": "user", "content": text}
            ],
            temperature=0.2,
            max_tokens=3000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error translating: {e}")
        return None


# FLUSH BUFFER
def flush_buffer(buffer, output_file):
    with open(output_file, "a", encoding="utf-8") as f:
        for line in buffer:
            f.write(line + "\n")
    print(f"[Saved] {len(buffer)} translations flushed to {output_file}")


# MAIN
def main():
    checkpoint = load_checkpoint()
    translated_count = checkpoint["translated_count"]
    output_file = checkpoint.get("output_file", OUTPUT_FILE)

    print(f"Resuming from line {translated_count}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    buffer = []

    for i, line in enumerate(lines):
        if i < translated_count:
            continue

        line = line.strip()
        if not line:
            translated_count += 1
            continue

        translation = translate_line(line)
        if translation is None:
            print(f"[Warning] Line {i} failed to translate, skipping.")
            translated_count += 1
            continue

        buffer.append(translation)
        translated_count += 1

        print(f"[{translated_count}/{total}] Done")

        if len(buffer) >= SAVE_EVERY:
            flush_buffer(buffer, output_file)
            checkpoint["translated_count"] = translated_count
            save_checkpoint(checkpoint)
            buffer = []

        time.sleep(0.5)

    if buffer:
        flush_buffer(buffer, output_file)
        checkpoint["translated_count"] = translated_count
        save_checkpoint(checkpoint)

    print(f"Done. All {translated_count} translations saved to {output_file}")


if __name__ == "__main__":
    main()
