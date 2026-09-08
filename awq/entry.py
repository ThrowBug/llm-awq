from lm_eval import evaluator, tasks
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import argparse
import os
import json
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    dispatch_model,
    load_checkpoint_in_model,
)
from accelerate.utils.modeling import get_balanced_memory
from awq.utils.parallel import auto_parallel
from awq.quantize.pre_quant import run_awq, apply_awq
from awq.quantize.quantizer import (
    pseudo_quantize_model_weight,
    real_quantize_model_weight,
)
from awq.quantize.qwen3_moe import (
    Qwen3MoeQuantPolicy,
    quantization_summary,
)
from awq.utils.fake_checkpoint import save_fake_quant_checkpoint
from awq.utils.lm_eval_adaptor import LMEvalAdaptor
from awq.utils.utils import simple_dispatch_model
from datasets import load_dataset
from torch import nn
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="path of the hf model")
parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--tasks", default=None, type=str)
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument("--num_fewshot", type=int, default=0)
# model config
parser.add_argument("--parallel", action="store_true", help="enable model parallelism")
# max memory to offload larger models to CPU
parser.add_argument(
    "--max_memory",
    type=str,
    nargs="*",
    help="List of device_id:max_memory pairs to be parsed into a dictionary; "
    + "Example: 0:10GiB 1:10GiB cpu:30GiB; "
    + "mode details here: "
    + "https://huggingface.co/docs/accelerate/usage_guides/big_modeling",
)
parser.add_argument(
    "--auto_parallel",
    action="store_true",
    help="automatically set parallel and batch_size",
)
# quantization config
parser.add_argument("--w_bit", type=int, default=None)
parser.add_argument("--expert_w_bit", type=int, default=None)
parser.add_argument("--attn_w_bit", type=int, default=None)
parser.add_argument("--q_group_size", type=int, default=-1)
parser.add_argument("--no_zero_point", action="store_true", help="disable zero_point")
parser.add_argument("--q_backend", type=str, default="fake", choices=["fake", "real"])
# save/load real quantized weights
parser.add_argument("--dump_quant", type=str, default=None, help="save quantized model")
parser.add_argument(
    "--dump_fake", type=str, default=None, help="save fake-quantized model"
)
parser.add_argument("--load_quant", type=str, default=None, help="load quantized model")
# apply/save/load awq
parser.add_argument("--run_awq", action="store_true", help="perform awq search process")
parser.add_argument(
    "--dump_awq", type=str, default=None, help="save the awq search results"
)
parser.add_argument(
    "--load_awq", type=str, default=None, help="load the awq search results"
)
parser.add_argument(
    "--calib_dataset", type=str, default="pileval", choices=["pileval", "c4"]
)
parser.add_argument("--calib_data_path", type=str, default=None)
parser.add_argument("--n_samples", type=int, default=128)
parser.add_argument("--seqlen", type=int, default=512)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--calib_batch_size", type=int, default=1)
parser.add_argument("--use_fast", action="store_true")
parser.add_argument(
    "--save_dtype",
    type=str,
    default="bfloat16",
    choices=["float16", "bfloat16"],
)
parser.add_argument(
    "--vila-15",
    action="store_true",
    help="quantizing vila 1.5",
)
parser.add_argument(
    "--vila-20",
    action="store_true",
    help="quantizing or smoothing vila 2.0 (NVILA)",
)
parser.add_argument(
    "--smooth_scale",
    action="store_true",
    help="generate the act scale of visiontower",
)
parser.add_argument(
    "--media_path",
    type=str,
    nargs="+",
    help="The input video to get act scale for visiontower",
)
parser.add_argument(
    "--act_scale_path",
    type=str,
    default=None,
    help="Path to save act scale",
)
args = parser.parse_args()
assert (
    args.act_scale_path is not None and len(args.media_path) > 0
) or not args.smooth_scale
vila_10_quant_mode = (
    ("llava" in args.model_path.lower() or "vila" in args.model_path.lower())
    and not args.vila_15
    and not args.vila_20
)

max_memory = [v.split(":") for v in (args.max_memory or [])]
max_memory = {(int(k) if k.isdigit() else k): v for k, v in max_memory}

if args.auto_parallel:
    gpu_list = auto_parallel(args)

# get quantization config (apart from w_bit)
q_config = {
    "zero_point": not args.no_zero_point,  # by default True
    "q_group_size": args.q_group_size,  # whether to use group quantization
}
print("Quantization config:", q_config)

# build model and tokenizer


def build_model_and_enc(model_path, dtype):
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
    print(f"* Building model {model_path}")

    # all hf model
    if vila_10_quant_mode:
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path

        enc, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=get_model_name_from_path(model_path),
            device="cpu",
            **{"use_cache": False},
        )
    else:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # Note (Haotian): To avoid OOM after huggingface transformers 4.36.2
        original_use_cache = getattr(config, "use_cache", True)
        config.use_cache = False
        if "mpt" in config.__class__.__name__.lower():
            enc = AutoTokenizer.from_pretrained(
                config.tokenizer_name, trust_remote_code=True
            )
        else:
            enc = AutoTokenizer.from_pretrained(
                model_path, use_fast=args.use_fast, trust_remote_code=True
            )

    quant_policy = None
    if args.expert_w_bit is not None or args.attn_w_bit is not None:
        if args.expert_w_bit is None or args.attn_w_bit is None:
            raise ValueError(
                "--expert_w_bit and --attn_w_bit must be specified together."
            )
        if getattr(config, "model_type", None) != "qwen3_moe":
            raise ValueError(
                "The heterogeneous expert/attention policy currently supports "
                "only model_type=qwen3_moe."
            )
        quant_policy = Qwen3MoeQuantPolicy(
            expert_w_bit=args.expert_w_bit,
            attn_w_bit=args.attn_w_bit,
        )

    if args.load_quant:  # directly load quantized weights
        print("Loading pre-computed quantized weights...")
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config=config, dtype=torch_dtype, trust_remote_code=True
            )
        real_quantize_model_weight(
            model, w_bit=args.w_bit, q_config=q_config, init_only=True
        )

        model.tie_weights()

        # Infer device map
        kwargs = {"max_memory": max_memory} if len(max_memory) else {}
        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=[
                "OPTDecoderLayer",
                "LlamaDecoderLayer",
                "BloomBlock",
                "MPTBlock",
                "DecoderLayer",
                "Qwen3MoeDecoderLayer",
            ],
            **kwargs,
        )
        # Load checkpoint in the model
        load_checkpoint_in_model(
            model,
            checkpoint=args.load_quant,
            device_map=device_map,
            offload_state_dict=True,
        )
        # Dispatch model
        model = simple_dispatch_model(model, device_map=device_map)

        model.eval()
    else:  # fp16 to quantized
        args.run_awq &= not args.load_awq  # if load_awq, no need to run awq
        # Init model on CPU:
        kwargs = {"dtype": torch_dtype, "low_cpu_mem_usage": True}
        if not vila_10_quant_mode:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, config=config, trust_remote_code=True, **kwargs
            )

        model.eval()
        if not vila_10_quant_mode:
            model._awq_original_use_cache = original_use_cache
        if quant_policy is not None:
            print("Qwen3-MoE quantization policy:", quantization_summary(model, quant_policy))

        if args.run_awq:
            assert args.dump_awq, "Please save the awq results with --dump_awq"

            awq_results = run_awq(
                model,
                enc,
                w_bit=args.w_bit,
                q_config=q_config,
                n_samples=args.n_samples,
                seqlen=args.seqlen,
                calib_data=args.calib_dataset,
                calib_data_path=args.calib_data_path,
                calib_batch_size=args.calib_batch_size,
                seed=args.seed,
                quant_policy=quant_policy,
            )
            if args.dump_awq:
                dirpath = os.path.dirname(args.dump_awq)
                if dirpath:
                    os.makedirs(dirpath, exist_ok=True)

                torch.save(awq_results, args.dump_awq)
                print("AWQ results saved at", args.dump_awq)

            exit(0)

        if args.load_awq:
            print("Loading pre-computed AWQ results from", args.load_awq)
            awq_results = torch.load(args.load_awq, map_location="cpu")
            apply_awq(
                model,
                awq_results,
                quant_policy=quant_policy,
                q_config=q_config,
            )

        # weight quantization
        if args.w_bit is not None or quant_policy is not None:
            if args.q_backend == "fake":
                assert (
                    args.dump_quant is None
                ), "Need to use real quantization to dump quantized weights"
                pseudo_quantize_model_weight(
                    model,
                    w_bit=args.w_bit,
                    q_config=q_config,
                    quant_policy=quant_policy,
                )
                if args.dump_fake:
                    metadata = {
                        "format": "awq-pseudo-quantized-hf-checkpoint",
                        "model_path": model_path,
                        "quant_policy": (
                            quant_policy.to_dict() if quant_policy else {"w_bit": args.w_bit}
                        ),
                        "q_config": dict(q_config),
                        "awq_search": (
                            awq_results.get("metadata") if args.load_awq else None
                        ),
                    }
                    save_fake_quant_checkpoint(
                        model,
                        enc,
                        args.dump_fake,
                        metadata=metadata,
                        save_dtype=args.save_dtype,
                    )
                    print("Pseudo-quantized models saved at", args.dump_fake)
                    exit(0)
            elif args.q_backend == "real":  # real quantization
                if quant_policy is not None:
                    raise NotImplementedError(
                        "The heterogeneous Qwen3-MoE policy supports fake quantization only."
                    )
                real_quantize_model_weight(model, w_bit=args.w_bit, q_config=q_config)
                if args.dump_quant:
                    if not args.dump_quant.endswith("v2.pt"):
                        print("[Info] Auto-change the dump_quant file name to *v2.pt")
                        args.dump_quant = args.dump_quant.replace(".pt", "-v2.pt")
                    dirpath = os.path.dirname(args.dump_quant)
                    os.makedirs(dirpath, exist_ok=True)

                    print(f"Saving the quantized model at {args.dump_quant}...")
                    torch.save(model.cpu().state_dict(), args.dump_quant)
                    exit(0)
            else:
                raise NotImplementedError

        # Move the model to GPU (as much as possible) for LM evaluation
        kwargs = {
            "max_memory": get_balanced_memory(
                model, max_memory if len(max_memory) > 0 else None
            )
        }
        device_map = infer_auto_device_map(
            model,
            # TODO: can we remove this?
            no_split_module_classes=[
                "OPTDecoderLayer",
                "LlamaDecoderLayer",
                "BloomBlock",
                "MPTBlock",
                "DecoderLayer",
                "Qwen3MoeDecoderLayer",
            ],
            **kwargs,
        )
        model = dispatch_model(model, device_map=device_map)

    return model, enc


def main():
    if args.output_path is not None and os.path.exists(args.output_path):
        # print(f"Results {args.output_path} already generated. Exit.")
        print(f"Results {args.output_path} already generated. Overwrite.")
        # exit()

    # a hack here to auto set model group
    if args.smooth_scale and args.vila_20:
        if os.path.exists(args.act_scale_path):
            print(f"Found existing Smooth Scales {args.act_scale_path}, skip.")
        else:
            from awq.quantize import get_smooth_scale

            act_scale = get_smooth_scale(args.model_path, args.media_path)
            os.makedirs(os.path.dirname(args.act_scale_path), exist_ok=True)
            torch.save(act_scale, args.act_scale_path)
            print("Save act scales at " + str(args.act_scale_path))
            args.model_path = args.model_path + "/llm"
        if args.dump_awq is None and args.dump_quant is None:
            exit()

    if args.dump_awq and os.path.exists(args.dump_awq):
        print(f"Found existing AWQ results {args.dump_awq}, exit.")
        exit()
    model, enc = build_model_and_enc(args.model_path, args.dtype)

    if args.tasks is not None:
        # https://github.com/IST-DASLab/gptq/blob/2d65066eeb06a5c9ff5184d8cebdf33662c67faf/llama.py#L206
        if args.tasks == "wikitext":
            testenc = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            testenc = enc("\n\n".join(testenc["text"]), return_tensors="pt")
            model.seqlen = 2048
            testenc = testenc.input_ids.to(model.device)
            nsamples = testenc.numel() // model.seqlen
            model = model.eval()
            nlls = []
            for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
                batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
                    model.device
                )
                with torch.no_grad():
                    lm_logits = model(batch).logits
                shift_logits = lm_logits[:, :-1, :].contiguous().float()
                shift_labels = testenc[
                    :, (i * model.seqlen) : ((i + 1) * model.seqlen)
                ][:, 1:]
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                neg_log_likelihood = loss.float() * model.seqlen
                nlls.append(neg_log_likelihood)

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
            print(ppl.item())

            results = {"ppl": ppl.item()}
            if args.output_path is not None:
                os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
                with open(args.output_path, "w") as f:
                    json.dump(results, f, indent=2)
        else:
            task_names = args.tasks.split(",")

            lm_eval_model = LMEvalAdaptor(args.model_path, model, enc, args.batch_size)
            results = evaluator.simple_evaluate(
                model=lm_eval_model,
                tasks=task_names,
                batch_size=args.batch_size,
                no_cache=True,
                num_fewshot=args.num_fewshot,
            )

            print(evaluator.make_table(results))

        if args.output_path is not None:
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
            # otherwise cannot save
            results["config"]["model"] = args.model_path
            with open(args.output_path, "w") as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
