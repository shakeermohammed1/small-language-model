import torch
from src.training.model import SLM
from src.data.tokenizer import load_tokenizer


def load_model(checkpoint_path: str, config: dict, device: str = "cpu") -> SLM:
    model = SLM(**config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model.to(device)


def generate_text(
    model: SLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    device: str = "cpu",
) -> str:
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    out = model.generate(tokens, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
