from transformers import AutoTokenizer


def load_tokenizer(model_name: str = "gpt2"):
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_text(tokenizer, text: str, max_length: int = 512):
    return tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def encode_corpus(tokenizer, texts: list[str]) -> list[int]:
    """Encode a list of texts into a single flat token ID list."""
    ids = []
    for text in texts:
        ids.extend(tokenizer.encode(text))
    return ids
