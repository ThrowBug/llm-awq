import json
from pathlib import Path

import torch


@torch.no_grad()
def save_fake_quant_checkpoint(
    model,
    tokenizer,
    output_path,
    metadata=None,
    save_dtype="bfloat16",
):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if save_dtype not in ("bfloat16", "float16"):
        raise ValueError(f"Unsupported save dtype: {save_dtype}")
    dtype = torch.bfloat16 if save_dtype == "bfloat16" else torch.float16
    original_use_cache = getattr(model, "_awq_original_use_cache", None)
    if original_use_cache is not None:
        model.config.use_cache = original_use_cache

    model.to(device="cpu", dtype=dtype)
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(
        output_path, safe_serialization=True, max_shard_size="5GB"
    )
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.save_pretrained(output_path)

    if metadata is not None:
        metadata_path = output_path / "awq_fake_quant_config.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)

    required_files = ("config.json", "tokenizer_config.json")
    missing = [name for name in required_files if not (output_path / name).is_file()]
    weight_files = list(output_path.glob("*.safetensors")) + list(output_path.glob("*.bin"))
    if missing or not weight_files:
        details = []
        if missing:
            details.append(f"missing files: {missing}")
        if not weight_files:
            details.append("no .safetensors or .bin weight files")
        raise RuntimeError(
            f"Incomplete fake-quant Hugging Face checkpoint at {output_path}: "
            + "; ".join(details)
        )

    return output_path
