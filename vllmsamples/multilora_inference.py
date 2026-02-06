# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Multi-LoRA offline inference example (Qwen version).

Key points:
- Base model: Qwen/Qwen2.5-3B-Instruct
- LoRA: a Qwen2.5-3B Text2SQL adapter (Spider)
- max_loras=1 -> only one LoRA can be active per batch, so different LoRA requests
  will be scheduled sequentially (one after another), not mixed.

Docs:
- vLLM LoRA feature overview: https://docs.vllm.ai/en/latest/features/lora/
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from huggingface_hub import snapshot_download

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest


PromptItem = Tuple[str, SamplingParams, Optional[LoRARequest]]


def create_test_prompts(lora_path: str) -> List[PromptItem]:
    """
    Create a list of prompts with sampling params.
    We send:
      - 2 requests to the base model (no LoRA)
      - 2 requests with LoRA, but as two different LoRA "identities"
        (same adapter path, different name/id) to demonstrate multi-LoRA routing.
    With max_loras=1, those two LoRA identities won't run in the same batch.
    """
    return [
        (
            "A robot may not injure a human being",
            SamplingParams(temperature=0.0, logprobs=1, max_tokens=128),
            None,
        ),
        (
            "To be or not to be,",
            SamplingParams(
                temperature=0.8,
                top_k=5,
                presence_penalty=0.2,
                max_tokens=128,
            ),
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
    """Continuously feed prompts to the engine and print finished outputs."""
    request_id = 0

    print("-" * 50)
    while test_prompts or engine.has_unfinished_requests():
        # Feed one new request per loop iteration (demo style)
        if test_prompts:
            prompt, sampling_params, lora_request = test_prompts.pop(0)
            engine.add_request(
                request_id=str(request_id),
                prompt=prompt,
                sampling_params=sampling_params,
                lora_request=lora_request,
            )
            request_id += 1

        # Execute one decoding step for all active requests
        request_outputs: List[RequestOutput] = engine.step()

        # Print only completed requests
        for out in request_outputs:
            if out.finished:
                print(out)
                print("-" * 50)


def initialize_engine() -> LLMEngine:
    """
    Initialize vLLM engine with LoRA enabled.

    Notes:
    - max_lora_rank MUST be >= the LoRA adapter's rank (here the chosen adapter uses r=64).
    - max_loras=1 => only one LoRA can be active in a single batch (lower VRAM usage).
    """
    engine_args = EngineArgs(
        model="Qwen/Qwen2.5-3B-Instruct",
        enable_lora=True,
        max_loras=1,
        max_lora_rank=64,   # important for this adapter
        max_cpu_loras=2,
        max_num_seqs=256,
    )
    return LLMEngine.from_engine_args(engine_args)


def main() -> None:
    engine = initialize_engine()

    # Qwen2.5-3B Text2SQL LoRA (Spider)
    lora_path = snapshot_download(
        repo_id="lleticiasilvaa/Qwen2.5-3B-Instruct-Spider-Text2SQL-SchemaLinking"
    )

    test_prompts = create_test_prompts(lora_path)
    process_requests(engine, test_prompts)


if __name__ == "__main__":
    main()
