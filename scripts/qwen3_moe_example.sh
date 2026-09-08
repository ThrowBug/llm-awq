#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

model_path="${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
c4_data_path="${C4_DATA_PATH:-${repo_root}/../GEMQ/data/c4-train.00000-of-01024.json}"
awq_cache="${AWQ_CACHE:-${repo_root}/results/awq_cache/qwen3-30b-a3b-e2-attn4-g128.pt}"
fq_model_path="${FQ_MODEL_PATH:-${repo_root}/results/fake_quant_models/Qwen3-30B-A3B-Instruct-2507/AWQ-E2-Attn4-G128}"

if [[ ! -f "${c4_data_path}" ]]; then
    echo "C4 calibration file not found: ${c4_data_path}" >&2
    exit 1
fi

common_args=(
    --model_path "${model_path}"
    --dtype bfloat16
    --expert_w_bit 2
    --attn_w_bit 4
    --q_group_size "${Q_GROUP_SIZE:-128}"
    --calib_dataset c4
    --calib_data_path "${c4_data_path}"
    --n_samples 128
    --seqlen 2048
    --seed "${CALIB_SEED:-0}"
    --calib_batch_size "${CALIB_BATCH_SIZE:-1}"
    --use_fast
)

# Stage 1: search AWQ scales/clipping values and persist only the search cache.
python -m awq.entry \
    "${common_args[@]}" \
    --run_awq \
    --dump_awq "${awq_cache}"

# Stage 2: reload pristine HF weights, apply the cache once, fake-quantize, and
# save a complete Hugging Face directory consumable as GEMQ's FQ_MODEL_PATH.
python -m awq.entry \
    "${common_args[@]}" \
    --load_awq "${awq_cache}" \
    --q_backend fake \
    --save_dtype bfloat16 \
    --dump_fake "${fq_model_path}"

echo "Fake-quant checkpoint: ${fq_model_path}"
echo "Serve from GEMQ with: cd '${repo_root}/../GEMQ' && FQ_MODEL_PATH='${fq_model_path}' bash scripts/Qwen3-30B-A3B-Instruct-2507/serve_vllm.sh"
