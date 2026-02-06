# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Multi-LoRA offline inference example (Qwen version) for vLLM 0.14.x.

Fix: LLMEngine.add_request() in vLLM v0.14.x does NOT accept
keyword argument `sampling_params`. Pass it positionally.

Base model: Qwen/Qwen2.5-3B-Instruct
LoRA: lleticiasilvaa/Qwen2.5-3B-Instruct-Spider-Text2SQL-SchemaLinking (rank 64)

Note: max_lora_rank must be >= LoRA rank.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from huggingface_hub import snapshot_download

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest

PromptItem = Tuple[str, SamplingParams, Optional[LoRARequest]]


def create_test_prompts(lora_path: str) -> List[PromptItem]:
    return [
        (
            "A robot may not injure a human being",
            SamplingParams(temperature=0.0, logprobs=1, max_tokens=128),
            None,
        ),
        (
            "To be or not to be,",
            SamplingParams(temperature=0.8, top_k=5, presence_penalty=0.2, max_tokens=128),
            None,
        ),
        (
            "[user] Write a SQL query to answer the question based on the table schema.\n\n"
            "context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n"
            "question: Name the ICAO for lilongwe international airport [/user] [assistant]",
            SamplingParams(temperature=0.0, logprobs=1, max_tokens=128),
            LoRARequest("sql-lora", 1, lora_path),
        ),
        (
            "[user] Write a SQL query to answer the question based on the table schema.\n\n"
            "context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n"
            "question: Name the ICAO for lilongwe international airport [/user] [assistant]",
            SamplingParams(temperature=0.0, logprobs=1, max_tokens=128),
            LoRARequest("sql-lora2", 2, lora_path),
        ),
    ]


def process_requests(engine: LLMEngine, test_prompts: List[PromptItem]) -> None:
    request_id = 0

    print("-" * 50)
    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params, lora_request = test_prompts.pop(0)

            # IMPORTANT (vLLM 0.14.x): pass sampling_params POSITIONALLY
            engine.add_request(
                str(request_id),
                prompt,
                sampling_params,
                lora_request=lora_request,
            )
            request_id += 1

        request_outputs: List[RequestOutput] = engine.step()

        for out in request_outputs:
            if out.finished:
                # out.outputs[0].text is often the generated text (depends on vLLM version)
                print(out)
                print("-" * 50)


def initialize_engine() -> LLMEngine:
    engine_args = EngineArgs(
        model="Qwen/Qwen2.5-3B-Instruct",
        enable_lora=True,
        max_loras=1,
        max_lora_rank=64,  # adapter uses r=64
        max_cpu_loras=2,
        max_num_seqs=256,
    )
    return LLMEngine.from_engine_args(engine_args)


def main() -> None:
    engine = initialize_engine()

    lora_path = snapshot_download(
        repo_id="lleticiasilvaa/Qwen2.5-3B-Instruct-Spider-Text2SQL-SchemaLinking"
    )

    test_prompts = create_test_prompts(lora_path)
    process_requests(engine, test_prompts)


if __name__ == "__main__":
    main()
