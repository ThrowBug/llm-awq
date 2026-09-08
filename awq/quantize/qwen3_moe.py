from dataclasses import asdict, dataclass
from typing import Dict, Optional


ATTENTION_PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
)
EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
ROUTER_NAME = "mlp.gate"


@dataclass(frozen=True)
class Qwen3MoeQuantPolicy:
    expert_w_bit: int = 2
    attn_w_bit: int = 4

    def __post_init__(self):
        for name, value in (
            ("expert_w_bit", self.expert_w_bit),
            ("attn_w_bit", self.attn_w_bit),
        ):
            if value not in (2, 3, 4, 8):
                raise ValueError(f"{name} must be one of 2, 3, 4, or 8; got {value}.")

    def bit_for(self, module_name: str) -> Optional[int]:
        if module_name in ATTENTION_PROJECTIONS:
            return self.attn_w_bit
        if module_name.startswith("mlp.experts.") and module_name.rsplit(".", 1)[-1] in EXPERT_PROJECTIONS:
            return self.expert_w_bit
        return None

    def is_router(self, module_name: str) -> bool:
        return module_name == ROUTER_NAME

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


def is_qwen3_moe_model(model) -> bool:
    return getattr(getattr(model, "config", None), "model_type", None) == "qwen3_moe"


def validate_qwen3_moe_layer(layer, named_linears, policy: Qwen3MoeQuantPolicy):
    missing_attention = [name for name in ATTENTION_PROJECTIONS if name not in named_linears]
    if missing_attention:
        raise ValueError(f"Qwen3-MoE layer is missing attention projections: {missing_attention}")

    if ROUTER_NAME not in named_linears:
        raise ValueError(f"Qwen3-MoE layer is missing router module {ROUTER_NAME!r}.")

    experts = getattr(getattr(layer, "mlp", None), "experts", None)
    if experts is None:
        raise ValueError("Expected Qwen3-MoE layer.mlp.experts to be present.")

    missing_experts = []
    for expert_idx in range(len(experts)):
        for projection in EXPERT_PROJECTIONS:
            name = f"mlp.experts.{expert_idx}.{projection}"
            if name not in named_linears:
                missing_experts.append(name)
    if missing_experts:
        preview = ", ".join(missing_experts[:8])
        raise ValueError(f"Qwen3-MoE layer is missing expert projections: {preview}")

    unexpected_quantized = [
        name for name in named_linears if policy.bit_for(name) is not None
        and name not in ATTENTION_PROJECTIONS
        and not name.startswith("mlp.experts.")
    ]
    if unexpected_quantized:
        raise ValueError(f"Unexpected Qwen3-MoE quantization targets: {unexpected_quantized}")


def quantization_summary(model, policy: Qwen3MoeQuantPolicy):
    layers = model.model.layers
    num_experts = getattr(model.config, "num_experts", 0)
    return {
        "num_layers": len(layers),
        "num_experts_per_layer": num_experts,
        "attention_linears": len(layers) * len(ATTENTION_PROJECTIONS),
        "expert_linears": len(layers) * num_experts * len(EXPERT_PROJECTIONS),
        "routers": len(layers),
        "policy": policy.to_dict(),
    }
