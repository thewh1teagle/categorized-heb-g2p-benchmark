"""
Download models before running:
    wget https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx -O renikud.onnx
    wget https://huggingface.co/thewh1teagle/phonikud-onnx/resolve/main/phonikud-1.0.int8.onnx -O phonikud.onnx
"""

import re
import csv
import pandas as pd
from tqdm import tqdm
from renikud_onnx import G2P
from phonikud_onnx import Phonikud
from phonikud import phonemize

HEBREW_RE = re.compile(r"[\u05d0-\u05ff]+")
PHONEME_RE = re.compile(r"[abdefhijklmnopstuvwzɡʁʃʒʔˈχ]+")


def extract_hebrew_words(sentence: str) -> list[str]:
    return HEBREW_RE.findall(sentence)


def extract_target_phonemes(phonemized_sentence: str, word_index: int) -> str:
    """Extract phonemes for the word at 1-based word_index from a phonemized sentence."""
    tokens = PHONEME_RE.findall(phonemized_sentence)
    if word_index < 0 or word_index >= len(tokens):
        return ""
    return tokens[word_index]


def main():
    renikud = G2P("renikud.onnx")
    phonikud_model = Phonikud("phonikud.onnx")

    df = pd.read_csv("data/gt.csv", header=None, names=["category", "sentence", "gt_raw"])

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        category = row["category"]
        sentence = row["sentence"]
        gt_raw = str(row["gt_raw"])

        # GT may have multiple targets: "2=leχˈa" or "1=ʔelˈajiχ 4=kibˈalt"
        targets = {}
        for part in gt_raw.strip().split():
            if "=" in part:
                idx_str, phonemes = part.split("=", 1)
                targets[int(idx_str)] = phonemes

        renikud_full = renikud.phonemize(sentence)
        vocalized = phonikud_model.add_diacritics(sentence)
        phonikud_full = phonemize(vocalized)

        word_indices = " ".join(str(i) for i in targets)
        gt_phonemes = " ".join(targets.values())
        renikud_pred = " ".join(extract_target_phonemes(renikud_full, i) for i in targets)
        phonikud_pred = " ".join(extract_target_phonemes(phonikud_full, i) for i in targets)
        rows.append({
            "category": category,
            "sentence": sentence,
            "word_indices": word_indices,
            "gt": gt_phonemes,
            "renikud": renikud_pred,
            "phonikud": phonikud_pred,
        })

    out = pd.DataFrame(rows)
    out.to_csv("data/predictions.csv", index=False, quoting=csv.QUOTE_ALL)
    print(f"Saved {len(out)} rows to data/predictions.csv")


if __name__ == "__main__":
    main()
