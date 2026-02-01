# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Сравнение модели с LoRA и без LoRA
"""

import gc

import torch
from huggingface_hub import snapshot_download

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest


def create_test_prompts(
        lora_path: str,
) -> list[tuple[str, SamplingParams, LoRARequest | None, str]]:
    """
    Создаём пары промптов: один без LoRA, один с LoRA
    Возвращает: (prompt, sampling_params, lora_request, label)
    """
    prompts = [
        "My name is",
        "The capital of USA is",
        "The capital of France is",
        "Explain what machine learning is:",
        "Write a short poem about the sea:",
    ]

    sampling_params = SamplingParams(temperature=0.0, logprobs=1, max_tokens=128)

    test_prompts = []
    for prompt in prompts:
        # Без LoRA (базовая модель)
        test_prompts.append((
            prompt,
            sampling_params,
            None,
            "BASE"
        ))
        # С LoRA (дообученная модель)
        test_prompts.append((
            prompt,
            sampling_params,
            LoRARequest("qlora-flan", 1, lora_path),
            "LORA"
        ))

    return test_prompts


def process_requests(
        engine: LLMEngine,
        test_prompts: list[tuple[str, SamplingParams, LoRARequest | None, str]],
):
    """Обрабатываем запросы и собираем результаты для сравнения."""
    request_id = 0
    results = {}  # {request_id: (prompt, label, output)}
    id_to_info = {}  # {request_id: (prompt, label)}

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params, lora_request, label = test_prompts.pop(0)
            engine.add_request(
                str(request_id), prompt, sampling_params, lora_request=lora_request
            )
            id_to_info[str(request_id)] = (prompt, label)
            request_id += 1

        request_outputs: list[RequestOutput] = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                req_id = request_output.request_id
                prompt, label = id_to_info[req_id]
                results[req_id] = (prompt, label, request_output.outputs[0].text)

    return results


def print_comparison(results: dict):
    """Выводим сравнение BASE vs LORA для каждого промпта."""

    # Группируем по промптам
    prompts_seen = []
    for req_id in sorted(results.keys(), key=int):
        prompt, label, output = results[req_id]
        if prompt not in prompts_seen:
            prompts_seen.append(prompt)

    for prompt in prompts_seen:
        print("=" * 70)
        print(f"PROMPT: {prompt}")
        print("=" * 70)

        # Находим BASE и LORA результаты для этого промпта
        for req_id in sorted(results.keys(), key=int):
            p, label, output = results[req_id]
            if p == prompt:
                if label == "BASE":
                    print(f"\n🔵 BASE (без LoRA):")
                    print(f"   {output.strip()}")
                else:
                    print(f"\n🟢 LORA (с LoRA):")
                    print(f"   {output.strip()}")

        print()


def initialize_engine(model: str, quantization: str) -> LLMEngine:
    """Инициализируем движок с поддержкой LoRA."""

    engine_args = EngineArgs(
        model=model,
        quantization=quantization,
        enable_lora=True,
        max_lora_rank=64,
        max_loras=4,
    )
    return LLMEngine.from_engine_args(engine_args)


def main():
    """Основная функция."""

    config = {
        "name": "qlora_inference_example",
        "model": "huggyllama/llama-7b",
        "quantization": "bitsandbytes",
        "lora_repo": "timdettmers/qlora-flan-7b",
    }

    print(f"Модель: {config['model']}")
    print(f"Квантизация: {config['quantization']}")
    print(f"LoRA адаптер: {config['lora_repo']}")
    print()

    # Инициализируем движок
    print("Загружаем модель...")
    engine = initialize_engine(config["model"], config["quantization"])

    # Скачиваем LoRA адаптер
    print("Скачиваем LoRA адаптер...")
    lora_path = snapshot_download(repo_id=config["lora_repo"])

    # Создаём тестовые промпты
    test_prompts = create_test_prompts(lora_path)

    # Обрабатываем запросы
    print("Генерируем ответы...")
    print()
    results = process_requests(engine, test_prompts)

    # Выводим сравнение
    print_comparison(results)

    # Очищаем память
    del engine
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()