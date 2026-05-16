import math
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device() -> torch.device:
    """Pick the best available device (MPS on Apple, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_context(device: torch.device, enabled: bool):
    """Return an autocast context for faster mixed-precision on supported devices."""
    if enabled and device.type in {"mps", "cuda"}:
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return nullcontext()


def make_text() -> str:
    """Create a small, built-in training corpus (ASCII only)."""
    base = (
        "Once upon a time there was a small village by the sea.\n"
        "The people worked, learned, and told stories at night.\n"
        "A curious student kept asking questions about the stars.\n"
        "The teacher smiled and said, learn a little every day.\n"
        "Practice makes progress, and progress brings confidence.\n"
        "\n"
        "A neural net is a set of simple rules stacked together.\n"
        "It learns patterns by adjusting numbers called weights.\n"
        "With time, it gets better at predicting the next symbol.\n"
        "Small steps, repeated often, can teach a big idea.\n"
        "\n"
        "The student wrote code, tested it, then tried again.\n"
        "The model slowly learned how to spell familiar words.\n"
        "The class kept notes about experiments and small changes.\n"
        "They compared losses, saved samples, and shared results.\n"
        "Some runs were noisy, others calm, but both taught lessons.\n"
        "A clear plan and steady practice helped the model improve.\n"
        "They read data, built batches, and trained with care.\n"
        "\n"
        "Attention lets each token look back at earlier context.\n"
        "Layer norms keep the scale of activations stable.\n"
        "Residual links help gradients flow through depth.\n"
        "Dropout adds small noise so the model learns robust patterns.\n"
        "\n"
        "When text makes sense, it feels like magic, but it is math.\n"
        "The model does not understand, it only predicts.\n"
        "Yet the predictions can feel smooth when trained well.\n"
        "\n"
        "A short poem can teach rhythm and simple rhyme.\n"
        "A list can teach structure and careful repetition.\n"
        "A dialog can teach how voices trade lines.\n"
        "\n"
        "Speaker A: What are you building today?\n"
        "Speaker B: A tiny model that learns from text.\n"
        "Speaker A: How does it learn?\n"
        "Speaker B: By predicting the next character.\n"
        "\n"
        "Consider a recipe for learning: gather data, write code, train, test.\n"
        "Consider a habit for learning: read, reflect, retry.\n"
        "Consider a rule for learning: small steps add up.\n"
        "\n"
        "In the morning the sea was calm, in the evening the wind grew loud.\n"
        "In the notebook there were sketches, in the terminal there were logs.\n"
        "The student kept both, and learned from each run.\n"
        "\n"
        "A model can mimic style, but style is not understanding.\n"
        "Still, it is useful for practice and for learning the tools.\n"
    )
    return base * 200


def build_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    """Create char-to-index and index-to-char maps."""
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> torch.Tensor:
    """Turn a string into a tensor of token ids."""
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def decode(ids: torch.Tensor, itos: dict[int, str]) -> str:
    """Turn a tensor of token ids back into a string."""
    return "".join([itos[int(i)] for i in ids])


class MultiHeadSelfAttention(nn.Module):
    """Causal self-attention (no looking ahead)."""

    def __init__(self, n_embd: int, n_heads: int, block_size: int, dropout: float) -> None:
        super().__init__()
        assert n_embd % n_heads == 0
        self.n_heads = n_heads
        self.head_size = n_embd // n_heads

        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Precompute a causal mask for max block size
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q = self.query(x).view(b, t, self.n_heads, self.head_size).transpose(1, 2)
        k = self.key(x).view(b, t, self.n_heads, self.head_size).transpose(1, 2)
        v = self.value(x).view(b, t, self.n_heads, self.head_size).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)
        att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        y = self.resid_drop(self.proj(y))
        return y


class FeedForward(nn.Module):
    """A simple MLP for each token position."""

    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """One transformer decoder block (LN -> Attn -> LN -> FF)."""

    def __init__(self, n_embd: int, n_heads: int, block_size: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadSelfAttention(n_embd, n_heads, block_size, dropout)
        self.ff = FeedForward(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyCharLLM(nn.Module):
    """A small decoder-only transformer for character-level modeling."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[TransformerBlock(n_embd, n_heads, block_size, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        pos = torch.arange(0, t, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = 0) -> torch.Tensor:
        """Autoregressive sampling from the model."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k > 0:
                values, _ = torch.topk(logits, k=top_k)
                logits[logits < values[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random chunks of text for training."""
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def estimate_loss(
    model: nn.Module,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: torch.device,
    eval_iters: int,
    use_mixed_precision: bool,
) -> dict[str, float]:
    """Estimate train/val loss without updating weights."""
    model.eval()
    out = {}
    with torch.no_grad():
        for split, data in [("train", train_data), ("val", val_data)]:
            losses = torch.zeros(eval_iters)
            for i in range(eval_iters):
                x, y = get_batch(data, batch_size, block_size, device)
                with autocast_context(device, use_mixed_precision):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                losses[i] = loss.item()
            out[split] = losses.mean().item()
    model.train()
    return out


def print_param_breakdown(model: nn.Module) -> None:
    """Print parameter counts by major component."""
    def count_params(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters())

    total = count_params(model)
    parts = {
        "token_emb": count_params(model.token_emb),
        "pos_emb": count_params(model.pos_emb),
        "blocks": count_params(model.blocks),
        "ln_f": count_params(model.ln_f),
        "head": count_params(model.head),
    }

    print("\n--- Parameter Breakdown ---")
    for name, count in parts.items():
        pct = 100.0 * count / total
        print(f"{name:9s}: {count:>10d} ({pct:5.1f}%)")
    print(f"total     : {total:>10d} (100.0%)")


def main() -> None:
    torch.manual_seed(1337)
    device = get_device()

    # Data and vocab
    text = make_text()
    stoi, itos = build_vocab(text)
    data = encode(text, stoi)

    # Train/validation split
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # Hyperparameters
    batch_size = 32
    block_size = 64
    n_embd = 192
    n_heads = 4
    n_layers = 4
    dropout = 0.1
    learning_rate = 3e-4
    max_iters = 2000
    eval_interval = 400
    eval_iters = 5
    use_mixed_precision = True
    use_compile = True

    # Model
    model = TinyCharLLM(
        vocab_size=len(stoi),
        block_size=block_size,
        n_embd=n_embd,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)

    if use_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception:
            pass

    print_param_breakdown(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for step in range(1, max_iters + 1):
        x, y = get_batch(train_data, batch_size, block_size, device)
        with autocast_context(device, use_mixed_precision):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0:
            losses = estimate_loss(
                model,
                train_data,
                val_data,
                batch_size,
                block_size,
                device,
                eval_iters,
                use_mixed_precision,
            )
            print(
                f"Step {step:04d} | Train loss: {losses['train']:.4f} | Val loss: {losses['val']:.4f}"
            )

    # Generate text from a single zero token
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample_ids = model.generate(start, max_new_tokens=300, temperature=0.7, top_k=20)
    print("\n--- Sample ---")
    print(decode(sample_ids[0], itos))


if __name__ == "__main__":
    main()
