import functools
import gc
import hashlib
from collections import defaultdict

import torch
import torch.nn as nn
import tqdm

from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

try:
    from tinychat.models import LlavaLlamaForCausalLM
except ImportError:
    LlavaLlamaForCausalLM = ()

from .auto_clip import apply_clip, auto_clip_block
from .auto_scale import apply_scale, auto_scale_block
from .qwen3_moe import is_qwen3_moe_model, validate_qwen3_moe_layer

__all__ = ["run_awq", "apply_awq"]


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_blocks(model):
    if model.__class__.__name__ in (
        "LlamaForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3MoeForCausalLM",
    ):
        layers = model.model.layers
    elif model.__class__.__name__ == "InternVL3":
        layers = model.language_model.model.layers
    elif model.__class__.__name__ == "LlavaLlamaForCausalLM":
        layers = model.model.layers
    elif isinstance(model, OPTForCausalLM):
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):
        layers = model.transformer.h
    elif "mpt" in str(model.__class__).lower():
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "bigcode" in str(model.__class__).lower():
        layers = model.transformer.h
    elif "neox" in str(model.__class__).lower():
        layers = model.gpt_neox.layers
    elif model.__class__.__name__ == "LlavaLlamaModel":
        layers = model.llm.model.layers
    else:
        raise NotImplementedError(type(model))
    return layers


def move_embed(model, device):
    if model.__class__.__name__ in (
        "LlamaForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3MoeForCausalLM",
    ):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        if hasattr(model.model, "rotary_emb"):
            model.model.rotary_emb = model.model.rotary_emb.to(device)
    elif model.__class__.__name__ == "InternVL3":
        model.language_model.model.embed_tokens = (
            model.language_model.model.embed_tokens.to(device)
        )
        model.language_model.model.rotary_emb = (
            model.language_model.model.rotary_emb.to(device)
        )
        model.vision_model.embeddings.to(device)
    elif isinstance(model, LlavaLlamaForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.vision_tower.vision_tower.vision_model.embeddings.to(device)
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(
            device
        )
    elif isinstance(model, BloomForCausalLM):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
        model.transformer.word_embeddings_layernorm = (
            model.transformer.word_embeddings_layernorm.to(device)
        )
    elif "mpt" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.emb_drop = model.transformer.emb_drop.to(device)
    elif "falcon" in str(model.__class__).lower():
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device)
    elif "bigcode" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device)
        model.transformer.wpe = model.transformer.wpe.to(device)
        model.transformer.drop = model.transformer.drop.to(device)
    elif "neox" in str(model.__class__).lower():
        model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(device)
        model.gpt_neox.emb_dropout = model.gpt_neox.emb_dropout.to(device)
        model.embed_out = model.embed_out.to(device)
    elif "llavallamamodel" in str(model.__class__).lower():
        model.llm.model.embed_tokens = model.llm.model.embed_tokens.to(device)
    else:
        raise NotImplementedError(type(model))


def _normalize_calib_samples(samples):
    if isinstance(samples, torch.Tensor):
        if samples.dim() != 2:
            raise ValueError(
                f"Calibration input_ids must be rank 2, got {tuple(samples.shape)}."
            )
        return samples.contiguous()
    return torch.cat(samples, dim=0).contiguous()


def _qwen3_hook_targets(named_linears):
    names = ["self_attn.q_proj", "self_attn.o_proj"]
    for name in named_linears:
        if name.startswith("mlp.experts.") and name.endswith(
            (".gate_proj", ".down_proj")
        ):
            names.append(name)
    return names


def _alias_qwen3_inputs(input_feat, named_linears):
    input_feat["self_attn.v_proj"] = input_feat["self_attn.q_proj"]
    for name in named_linears:
        if name.startswith("mlp.experts.") and name.endswith(".up_proj"):
            gate_name = name[: -len("up_proj")] + "gate_proj"
            if gate_name not in input_feat:
                raise RuntimeError(
                    f"Calibration did not route any token through {gate_name}; "
                    "increase the calibration sample count."
                )
            input_feat[name] = input_feat[gate_name]


@torch.no_grad()
def run_awq(
    model,
    enc,
    w_bit,
    q_config,
    n_samples=512,
    seqlen=512,
    auto_scale=True,
    mse_range=True,
    calib_data="pileval",
    calib_data_path=None,
    calib_batch_size=1,
    seed=42,
    quant_policy=None,
):
    from ..utils.calib_data import get_calib_dataset
    from ..utils.module import append_str_prefix, get_op_name

    if calib_batch_size < 1:
        raise ValueError("calib_batch_size must be at least 1.")
    if "bigcode" in str(model.__class__).lower():
        model.transformer.bias = model.transformer.bias.to("cuda")

    qwen3_moe = is_qwen3_moe_model(model)
    if qwen3_moe and quant_policy is None:
        raise ValueError("Qwen3-MoE AWQ requires an explicit heterogeneous policy.")

    layers = get_blocks(model)
    samples = _normalize_calib_samples(
        get_calib_dataset(
            data=calib_data,
            tokenizer=enc,
            n_samples=n_samples,
            block_size=seqlen,
            data_path=calib_data_path,
            seed=seed,
        )
    )
    if calib_data == "c4" and samples.shape != (n_samples, seqlen):
        raise ValueError(
            f"Expected calibration input_ids shape {(n_samples, seqlen)}, "
            f"got {tuple(samples.shape)}."
        )
    num_calib_samples = samples.shape[0]
    calib_seqlen = samples.shape[1]
    sample_hash = hashlib.sha256(samples.numpy().tobytes()).hexdigest()
    print(
        f" * Calibration input_ids: shape={tuple(samples.shape)}, "
        f"sha256={sample_hash}"
    )

    inps = []
    layer_kwargs = {}
    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")

    class CatcherExit(Exception):
        pass

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp.detach().cpu())
            layer_kwargs.update(kwargs)
            raise CatcherExit

    layers[0] = Catcher(layers[0])
    for start in range(0, num_calib_samples, calib_batch_size):
        batch = samples[start : start + calib_batch_size]
        try:
            if model.__class__.__name__ == "LlavaLlamaModel":
                model.llm(batch.to(next(model.parameters()).device))
            elif model.__class__.__name__ == "InternVL3":
                model.language_model(batch.to(next(model.parameters()).device))
            else:
                model(batch.to(next(model.parameters()).device))
        except CatcherExit:
            pass
    layers[0] = layers[0].module
    inps = torch.cat(inps, dim=0)
    del samples

    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")
    gc.collect()
    torch.cuda.empty_cache()

    awq_results = {
        "scale": [],
        "clip": [],
        "metadata": {
            "format_version": 2,
            "model_type": getattr(model.config, "model_type", None),
            "model_shape": {
                "num_hidden_layers": getattr(model.config, "num_hidden_layers", None),
                "num_experts": getattr(model.config, "num_experts", None),
                "hidden_size": getattr(model.config, "hidden_size", None),
            },
            "quant_policy": quant_policy.to_dict() if quant_policy else None,
            "q_config": dict(q_config),
            "calibration": {
                "dataset": calib_data,
                "data_path": calib_data_path,
                "requested_n_samples": n_samples,
                "n_samples": num_calib_samples,
                "seqlen": calib_seqlen,
                "seed": seed,
                "batch_size": calib_batch_size,
                "input_ids_sha256": sample_hash,
            },
        },
    }

    for i in tqdm.tqdm(range(len(layers)), desc="Running AWQ..."):
        layer = layers[i].cuda()
        named_linears = get_named_linears(layer)
        if qwen3_moe:
            validate_qwen3_moe_layer(layer, named_linears, quant_policy)

        def cache_input_hook(module, hook_input, output, name, feat_dict):
            feat_dict[name].append(hook_input[0].detach().cpu())

        input_feat = defaultdict(list)
        handles = []
        hook_names = (
            _qwen3_hook_targets(named_linears)
            if qwen3_moe
            else list(named_linears)
        )
        for name in hook_names:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        if qwen3_moe:
            handles.append(
                layer.mlp.register_forward_hook(
                    functools.partial(
                        cache_input_hook, name="mlp", feat_dict=input_feat
                    )
                )
            )

        next_inps = []
        for start in range(0, inps.shape[0], calib_batch_size):
            batch_inps = inps[start : start + calib_batch_size].to(
                next(layer.parameters()).device
            )
            batch_out = layer(batch_inps, **layer_kwargs)
            if isinstance(batch_out, (tuple, list)):
                batch_out = batch_out[0]
            next_inps.append(batch_out.detach().cpu())
        for handle in handles:
            handle.remove()
        inps = torch.cat(next_inps, dim=0)
        input_feat = {key: torch.cat(value, dim=0) for key, value in input_feat.items()}
        if qwen3_moe:
            _alias_qwen3_inputs(input_feat, named_linears)

        torch.cuda.empty_cache()
        if auto_scale:
            scales_list = auto_scale_block(
                layer,
                layer_kwargs,
                w_bit=w_bit,
                q_config=q_config,
                input_feat=input_feat,
                quant_policy=quant_policy,
                search_batch_size=calib_batch_size,
            )
            apply_scale(layer, scales_list, input_feat_dict=input_feat)
            awq_results["scale"] += append_str_prefix(
                scales_list, get_op_name(model, layer) + "."
            )

        torch.cuda.empty_cache()
        if mse_range:
            clip_list = auto_clip_block(
                layer,
                w_bit=w_bit,
                q_config=q_config,
                input_feat=input_feat,
                quant_policy=quant_policy,
            )
            apply_clip(layer, clip_list)
            awq_results["clip"] += append_str_prefix(
                clip_list, get_op_name(model, layer) + "."
            )

        layers[i] = layer.cpu()
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()

    return awq_results


def apply_awq(model, awq_results, quant_policy=None, q_config=None):
    metadata = awq_results.get("metadata")
    if quant_policy is not None:
        if not metadata:
            raise ValueError(
                "The AWQ cache has no policy metadata; rerun AWQ for Qwen3-MoE."
            )
        if metadata.get("quant_policy") != quant_policy.to_dict():
            raise ValueError(
                "AWQ cache policy does not match the requested quantization policy."
            )
        if q_config is not None and metadata.get("q_config") != dict(q_config):
            raise ValueError("AWQ cache q_config does not match the requested q_config.")
        current_shape = {
            "num_hidden_layers": getattr(model.config, "num_hidden_layers", None),
            "num_experts": getattr(model.config, "num_experts", None),
            "hidden_size": getattr(model.config, "hidden_size", None),
        }
        if metadata.get("model_type") != getattr(model.config, "model_type", None):
            raise ValueError("AWQ cache model_type does not match the loaded model.")
        if metadata.get("model_shape") != current_shape:
            raise ValueError("AWQ cache model shape does not match the loaded model.")
    apply_scale(model, awq_results["scale"])
    apply_clip(model, awq_results["clip"])
