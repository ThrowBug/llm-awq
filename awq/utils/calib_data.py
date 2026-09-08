import itertools
import os

import torch
from datasets import load_dataset


def _get_c4_calib_dataset(data_path, tokenizer, n_samples, block_size, seed):
    if not data_path:
        raise ValueError("--calib_data_path is required when --calib_dataset=c4.")
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"C4 calibration file not found: {data_path}")

    dataset = load_dataset(
        "json", data_files={"train": data_path}, split="train"
    )
    dataset = dataset.shuffle(seed=seed).select(
        range(min(n_samples * 16, len(dataset)))
    )
    text_column = "text" if "text" in dataset.features else list(dataset.features)[0]

    tokenized = dataset.map(
        lambda examples: tokenizer(examples[text_column]),
        batched=True,
        remove_columns=list(dataset.features),
    )

    def group_texts(examples):
        concatenated = {
            key: list(itertools.chain(*examples[key])) for key in examples
        }
        first_key = next(iter(concatenated))
        total_length = (len(concatenated[first_key]) // block_size) * block_size
        return {
            key: [
                tokens[i : i + block_size]
                for i in range(0, total_length, block_size)
            ]
            for key, tokens in concatenated.items()
        }

    blocks = tokenized.map(group_texts, batched=True)
    if len(blocks) < n_samples:
        raise ValueError(
            f"C4 calibration file produced only {len(blocks)} blocks of "
            f"length {block_size}; {n_samples} are required."
        )
    blocks = blocks.select(range(n_samples))
    return torch.tensor(blocks["input_ids"], dtype=torch.long)


def get_calib_dataset(
    data="pileval",
    tokenizer=None,
    n_samples=512,
    block_size=512,
    data_path=None,
    seed=42,
):
    if data == "c4":
        return _get_c4_calib_dataset(
            data_path=data_path,
            tokenizer=tokenizer,
            n_samples=n_samples,
            block_size=block_size,
            seed=seed,
        )
    if data == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    else:
        raise NotImplementedError
    dataset = dataset.shuffle(seed=seed)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]
