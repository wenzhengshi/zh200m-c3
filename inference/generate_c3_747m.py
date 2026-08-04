#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import random
from pathlib import Path

import sentencepiece as spm
import torch

from model_c3 import build_model


ROOT = Path("/root/workspace/jicreate_zh200m_c3")

CHECKPOINT = (
    ROOT
    / "runs/c3_random_100m_20260803_125316"
    / "checkpoint_tokens_747110400.pt"
)

TOKENIZER_PATH = (
    ROOT
    / "data_release_20260803"
    / "tokenizer_frozen"
    / "tokenizer.model"
)

OUTPUT_PATH = (
    ROOT
    / "eval_runs"
    / "c3_747m_generation.txt"
)

PROMPTS = [
    "中国的首都是",
    "春眠不觉晓",
    "1+1等于",
    "人工智能是",
    "今天天气很好，",
    "在很久很久以前，",
    "研究人员发现，",
    "你好，请介绍一下自己。",
]

MAX_NEW_TOKENS = 48
TEMPERATURE = 0.8
TOP_K = 40
SEED = 20260803


def generate(
    model: torch.nn.Module,
    prompt_ids: list[int],
    *,
    device: torch.device,
    eos_id: int,
    greedy: bool,
    seed: int,
) -> list[int]:
    generated = list(prompt_ids)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    for _ in range(MAX_NEW_TOKENS):
        context = generated[-2048:]

        input_ids = torch.tensor(
            [context],
            dtype=torch.long,
            device=device,
        )

        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ):
                logits = model(input_ids)[:, -1, :].float()

        if greedy:
            next_token = int(logits.argmax(dim=-1).item())
        else:
            logits = logits / TEMPERATURE
            k = min(TOP_K, logits.shape[-1])

            top_values, top_indices = torch.topk(
                logits,
                k=k,
                dim=-1,
            )

            probabilities = torch.softmax(
                top_values,
                dim=-1,
            )

            selected = torch.multinomial(
                probabilities,
                num_samples=1,
            )

            next_token = int(
                top_indices.gather(1, selected).item()
            )

        generated.append(next_token)

        if next_token == eos_id:
            break

    return generated


def main() -> None:
    assert CHECKPOINT.is_file(), f"Checkpoint不存在：{CHECKPOINT}"
    assert TOKENIZER_PATH.is_file(), f"Tokenizer不存在：{TOKENIZER_PATH}"

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    tokenizer = spm.SentencePieceProcessor(
        model_file=str(TOKENIZER_PATH)
    )

    print("正在加载Checkpoint……", flush=True)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )

    assert checkpoint["format"] == "zh200m_c3_v1"

    counters = checkpoint["counters"]

    assert counters["global_step"] == 2850
    assert counters["global_input_tokens"] == 747_110_400

    model = build_model(checkpoint["config"])
    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    del checkpoint
    gc.collect()

    device = torch.device("cuda")
    model.to(device)
    model.eval()

    print(
        f"模型加载成功：参数量={model.count_parameters():,}",
        flush=True,
    )

    output_lines = [
        json.dumps(
            {
                "checkpoint": str(CHECKPOINT),
                "global_step": counters["global_step"],
                "global_input_tokens": counters["global_input_tokens"],
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": TEMPERATURE,
                "top_k": TOP_K,
                "seed": SEED,
            },
            ensure_ascii=False,
            indent=2,
        ),
        "",
    ]

    for index, prompt in enumerate(PROMPTS):
        prompt_ids = tokenizer.encode(
            prompt,
            out_type=int,
        )

        greedy_ids = generate(
            model,
            prompt_ids,
            device=device,
            eos_id=tokenizer.eos_id(),
            greedy=True,
            seed=SEED + index,
        )

        sample_ids = generate(
            model,
            prompt_ids,
            device=device,
            eos_id=tokenizer.eos_id(),
            greedy=False,
            seed=SEED + 1000 + index,
        )

        greedy_full = tokenizer.decode(greedy_ids)
        sample_full = tokenizer.decode(sample_ids)

        greedy_new = tokenizer.decode(
            greedy_ids[len(prompt_ids):]
        )

        sample_new = tokenizer.decode(
            sample_ids[len(prompt_ids):]
        )

        block = [
            "=" * 80,
            f"提示：{prompt}",
            "",
            f"贪心完整：{greedy_full}",
            f"贪心续写：{greedy_new}",
            "",
            f"采样完整：{sample_full}",
            f"采样续写：{sample_new}",
        ]

        print("\n".join(block), flush=True)
        output_lines.extend(block)
        output_lines.append("")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        "\n".join(output_lines),
        encoding="utf-8",
    )

    print()
    print(f"生成结果已保存：{OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
