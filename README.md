# Neural networks 1.0

Learning repo with two PyTorch models built from scratch: a 3-layer feed-forward classifier and a tiny decoder-only character transformer. The transformer uses causal self-attention, residuals, layer norms, AdamW + cross-entropy, has ~1.81M params (0.001807B), trains on a built-in text corpus, and generates short samples.

## What this repo contains
- `simple_ffn.py`: three-layer feed-forward classifier with ReLU and softmax-style training (logits + cross-entropy).
- `tiny_char_llm.py`: a minimal decoder-only transformer that learns next-character prediction and can generate text.

## Tiny transformer details
- Parameters: 1,807,104 (0.001807104B)
- Model: 4 layers, 4 heads, 192 embedding size, block size 64
- Training: AdamW, cross-entropy, next-token prediction
- Sampling: temperature + top-k

## How to run

Create and activate a Python environment, then install dependencies:

```bash
pip install torch
```

Run the feed-forward network demo:

```bash
python3 simple_ffn.py
```

Run the tiny transformer:

```bash
python3 tiny_char_llm.py
```

## Notes
This is a learning-focused project. The tiny transformer is intentionally small so it can run on a laptop.
