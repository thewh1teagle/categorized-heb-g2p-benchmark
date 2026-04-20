# Categorized Hebrew G2P Benchmark

Benchmark for Hebrew grapheme-to-phoneme (G2P) models, evaluated per linguistic category.

## Categories

Acronyms, Foreign, Gender, Homographed Stress, Names, Rare Phonemes, Secondary Stress, Slang

## Models

- **renikud** — [`thewh1teagle/renikud`](https://huggingface.co/thewh1teagle/renikud)
- **phonikud** — [`thewh1teagle/phonikud-onnx`](https://huggingface.co/thewh1teagle/phonikud-onnx)

## Setup

```bash
uv sync
wget https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx -O renikud.onnx
wget https://huggingface.co/thewh1teagle/phonikud-onnx/resolve/main/phonikud-1.0.int8.onnx -O phonikud.onnx
```

## Usage

```bash
uv run src/predict.py    # generates data/predictions.csv
uv run src/benchmark.py  # generates data/benchmark_results.csv + charts
```

## Metrics

WER and CER (lower is better) computed with [jiwer](https://github.com/jitsi/jiwer) on target words specified in `data/gt.csv`.
