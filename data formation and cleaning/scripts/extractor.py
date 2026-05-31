import json
import os
import random
import time
from datetime import datetime
from openai import OpenAI

import re

client = OpenAI(api_key="api key")

# CONFIG
MODEL = "gpt-4o-mini"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OUTPUT_BASE = "isizulu_dataset"

TOTAL_SAMPLES = 250
PARAPHRASE_PROB = 0.3

TEMPERATURES = [0.2, 0.5, 0.8, 1.0]

TOPICS = [
    "Ukuhlaziywa komthelela wemidlalo ye-eSports entsheni yaseNingizimu Afrika",
    "Ukusetshenziswa kwezimali zedijithali (cryptocurrency) emabhizinisini amancane",
    "Izinguquko emikhubeni yokuthenga ngenxa yokukhula kwe-e-commerce",
    "Indima yezokuvakasha zasemakhaya ekuthuthukiseni umnotho wendawo",
    "Ukuhlaziywa kokusetshenziswa kwezilimi zomdabu ezinkundleni zedijithali",

    "Ukukhula komkhakha wezokudlala imidlalo yamavidiyo eNingizimu Afrika",
    "Izinselelo zokuvikelwa kwedatha yomuntu siqu ezinkundleni zedijithali",
    "Ukusetshenziswa kwama-drone kwezolimo zanamuhla",
    "Indima yabasunguli bamabhizinisi obuchwepheshe (tech startups) emnothweni",
    "Ukusetshenziswa kwezinkundla zokufunda online kubantu abadala",

    "Umthelela wokukhula kwemakethe yezimoto zikagesi",
    "Izinguquko endleleni abantu abasebenzisa ngayo amabhange edijithali",
    "Ukusetshenziswa kwe-blockchain ngaphandle kwezimali zedijithali",
    "Ukuhlaziywa komkhakha wezokudlala amabhodi nemidlalo yokucabanga",
    "Indima yeminyuziyamu ekugcineni umlando namasiko",

    "Ukusetshenziswa kobuchwepheshe bokuphrinta be-3D embonini",
    "Umthelela wokwanda kwezindawo zokusebenza ezihlanganyelwayo (co-working spaces)",
    "Ukukhula komkhakha wokuthengisa online kubathengisi abazimele",
    "Ucwaningo ngezindlela abantu abasebenzisa ngazo ama-smartwatch",
    "Izindlela zokuvikela izingane emhlabeni wedijithali",

    "Ukuhlaziywa komkhakha wezincwadi ezilalelwayo (audiobooks)",
    "Ukukhula kwezinhlelo zokusebenza zokufunda izilimi",
    "Umthelela wezobuciko bedijithali emisebenzini yesimanje",
    "Ukuhlaziywa kokwanda kokusebenza kwama-robot embonini",
    "Ukusetshenziswa kobuhlakani bokwenziwa embonini yezokuthutha",

    "Indima yemidlalo yendabuko ekwakheni ubumbano emphakathini",
    "Ukuhlaziywa komkhakha wokukhiqiza okuqukethwe ku-YouTube",
    "Izinguquko emikhubeni yokulalela umsakazo ngenxa yezinkundla zokusakaza",
    "Ucwaningo ngokusetshenziswa kwezimoto ezizishayelayo emhlabeni",
    "Ukusetshenziswa kobuchwepheshe be-virtual reality emfundweni"
]

STYLES = [
"Idokhumenti yenqubomgomo (policy document)",
"Idokhumenti yenqubo (Procedural document)",
"Iphepha lenqubomgomo (white paper)",
"isitayela esisemthethweni sezemfundo (formal academic)",
"isitayela sengxoxo esingakahleleki (informal conversational)",
"isitayela sombiko wezindaba (news report style)",
"isitayela sencazelo yokufundisa (educational explanation)",
"isitayela sombono (opinion piece)",
"isitayela semiyalelo noma izinqubo (procedural/instructional)",
"isitayela sokuxoxa indaba (narrative storytelling)",
"isitayela semibuzo nezimpendulo (Q&A format)",
"indaba yezindaba(news article)",    
"umbono womuntu(opinion piece)",        
"indaba emfushane(short story)",        
"ingxoxo(interview)",                 
"ukubuyekezwa(review)",           
"indaba yomlando(historical account)",         
"incazelo yomcimbi(event description)",       
"iphrofayela yomuntu(profile/biography)",    
"ingxoxo-mpikiswano(argumentative discussion)"       
]

TASKS = [
    """Bhala umbhalo ophelele ngesiZulu usebenzisa isihloko nesitayela esinikeziwe.

Isihloko: {topic}
Isitayela: {style}

Umbhalo mawuzwakale ngokwemvelo, ugeleze kahle, futhi ubhalwe ngendlela evumelana nesitayela esikhethiwe.

Ubude: amagama ayi-200–300.
"""
]

# Clean
def clean_text(text):
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"Isihloko: ", "", text)
    text = re.sub(r"\*", "", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# GENERATION
def generate_text(prompt, temperature, topic=None, style=None):
    system_msg = "Bhala kuphela ngesiZulu esicacile."
    if topic and style:
        system_msg += f" Isihloko: {topic}. Isitayela: {style}."
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=3000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error: {e}")
        return None

# PARAPHRASING
def paraphrase_text(text):
    prompt = f"Phinda ubhale lokhu ngenye indlela kodwa ugcine incazelo efanayo:\n\n{text}"
    return generate_text(prompt, temperature=0.7)

# QUALITY FILTER
def is_valid(text):
    if not text:
        return False
    if len(text.split()) < 20:
        return False
    if " the " in text.lower():  # crude non-Zulu check
        return False
    return True

CHECKPOINT_EVERY = 50

# CHECKPOINT
def save_checkpoint(data, output_file, output_csv):
    with open(output_file, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(output_csv, "w", encoding="utf-8") as f:
        for row in data:
            f.write(row["text"] + "\n")
    print(f"[Checkpoint] {len(data)} samples saved.")

# MAIN PIPELINE
def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{OUTPUT_DIR}/{OUTPUT_BASE}_{ts}.jsonl"
    output_csv = f"{OUTPUT_DIR}/{OUTPUT_BASE}_{ts}.csv"

    combos = [(topic, style, task) for topic in TOPICS for style in STYLES for task in TASKS]
    random.shuffle(combos)
    combos = combos[:TOTAL_SAMPLES]

    data = []

    for topic, style, task_template in combos:
        prompt = task_template.format(topic=topic, style=style)
        temp = random.choice(TEMPERATURES)

        text = clean_text(generate_text(prompt, temp, topic=topic, style=style))

        if not is_valid(text):
            continue

        if random.random() < PARAPHRASE_PROB:
            para = clean_text(paraphrase_text(text))
            if is_valid(para):
                text = para

        data.append({
            "text": text,
            "label": 1,
            "topic": topic,
            "style": style,
            "temperature": temp,
            "source": "gpt-4o-mini"
        })

        print(f"{len(data)}/{TOTAL_SAMPLES}")
        if len(data) % CHECKPOINT_EVERY == 0:
            save_checkpoint(data, output_file, output_csv)
        time.sleep(1)

    save_checkpoint(data, output_file, output_csv)
    print(f"Dataset saved: {output_file} and {output_csv}.")

if __name__ == "__main__":
    main()