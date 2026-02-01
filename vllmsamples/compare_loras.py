# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Сравнение трёх вариантов модели:
- BASE (без адаптера)
- FLAN-LoRA (timdettmers/qlora-flan-7b)
- CHAT-LoRA (наш обученный адаптер)
"""

import gc
from typing import NamedTuple

import torch
from huggingface_hub import snapshot_download

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest


class AdapterConfig(NamedTuple):
    """Конфигурация адаптера."""
    name: str
    label: str
    emoji: str
    path: str | None = None


def create_test_prompts(
        adapters: list[AdapterConfig],
) -> list[tuple[str, SamplingParams, LoRARequest | None, str, str]]:
    """
    Создаём промпты для всех адаптеров.
    Возвращает: (prompt, sampling_params, lora_request, label, emoji)
    """

    # Промпты для тестирования диалоговых способностей
    prompts = [
        # Простые вопросы
        "My name is",
        "The capital of France is",

        # Инструкции (где LoRA должен показать разницу)
        "### Human: What is machine learning?\n### Assistant:",
        "### Human: Write a short poem about the ocean.\n### Assistant:",
        "### Human: Explain quantum computing in simple terms.\n### Assistant:",

        # Диалоговый формат
        "User: Hello! How are you today?\nAssistant:",
        "User: Can you help me write a professional email?\nAssistant:",
    ]

    sampling_params = SamplingParams(
        temperature=0.7,  # Немного креативности
        top_p=0.9,
        max_tokens=150,
    )

    test_prompts = []
    lora_id = 1  # ID для LoRA запросов

    for prompt in prompts:
        for adapter in adapters:
            if adapter.path is None:
                # Базовая модель без адаптера
                lora_request = None
            else:
                lora_request = LoRARequest(adapter.name, lora_id, adapter.path)
                lora_id += 1

            test_prompts.append((
                prompt,
                sampling_params,
                lora_request,
                adapter.label,
                adapter.emoji,
            ))

    return test_prompts


def process_requests(
        engine: LLMEngine,
        test_prompts: list[tuple[str, SamplingParams, LoRARequest | None, str, str]],
) -> dict:
    """Обрабатываем запросы и собираем результаты."""

    request_id = 0
    results = {}
    id_to_info = {}

    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params, lora_request, label, emoji = test_prompts.pop(0)
            engine.add_request(
                str(request_id),
                prompt,
                sampling_params,
                lora_request=lora_request
            )
            id_to_info[str(request_id)] = (prompt, label, emoji)
            request_id += 1

        request_outputs: list[RequestOutput] = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                req_id = request_output.request_id
                prompt, label, emoji = id_to_info[req_id]
                output_text = request_output.outputs[0].text
                results[req_id] = (prompt, label, emoji, output_text)

    return results


def print_comparison(results: dict, adapters: list[AdapterConfig]):
    """Красивый вывод сравнения всех адаптеров."""

    # Собираем уникальные промпты в порядке появления
    prompts_seen = []
    for req_id in sorted(results.keys(), key=int):
        prompt, _, _, _ = results[req_id]
        if prompt not in prompts_seen:
            prompts_seen.append(prompt)

    for prompt in prompts_seen:
        print("\n" + "=" * 80)
        print(f"PROMPT: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
        print("=" * 80)

        # Выводим результаты для каждого адаптера
        for adapter in adapters:
            for req_id in sorted(results.keys(), key=int):
                p, label, emoji, output = results[req_id]
                if p == prompt and label == adapter.label:
                    print(f"\n{emoji} {label}:")
                    # Форматируем вывод с отступами
                    lines = output.strip().split('\n')
                    for line in lines[:10]:  # Ограничиваем 10 строками
                        print(f"   {line}")
                    if len(lines) > 10:
                        print(f"   ... (ещё {len(lines) - 10} строк)")
                    break


def initialize_engine(model: str, quantization: str) -> LLMEngine:
    """Инициализируем движок с поддержкой нескольких LoRA."""

    engine_args = EngineArgs(
        model=model,
        quantization=quantization,
        enable_lora=True,
        max_lora_rank=64,
        max_loras=4,  # Поддержка нескольких адаптеров одновременно
    )
    return LLMEngine.from_engine_args(engine_args)


def main():
    """Основная функция сравнения."""

    config = {
        "model": "huggyllama/llama-7b",
        "quantization": "bitsandbytes",
        "flan_lora_repo": "timdettmers/qlora-flan-7b",
        "chat_lora_path": "./chat-lora-adapter",  # Наш обученный адаптер
    }

    print("=" * 80)
    print("🔬 СРАВНЕНИЕ МОДЕЛЕЙ: BASE vs FLAN-LoRA vs CHAT-LoRA")
    print("=" * 80)
    print(f"\nМодель: {config['model']}")
    print(f"Квантизация: {config['quantization']}")
    print()

    # Инициализируем движок
    print("📦 Загружаем базовую модель...")
    engine = initialize_engine(config["model"], config["quantization"])

    # Скачиваем/находим адаптеры
    print("📥 Скачиваем FLAN-LoRA адаптер...")
    flan_lora_path = snapshot_download(repo_id=config["flan_lora_repo"])

    print(f"📂 Используем CHAT-LoRA адаптер: {config['chat_lora_path']}")

    # Конфигурация адаптеров
    adapters = [
        AdapterConfig(
            name="base",
            label="BASE",
            emoji="🔵",
            path=None,  # Без адаптера
        ),
        AdapterConfig(
            name="flan-lora",
            label="FLAN-LoRA",
            emoji="🟢",
            path=flan_lora_path,
        ),
        AdapterConfig(
            name="chat-lora",
            label="CHAT-LoRA",
            emoji="🟣",
            path=config["chat_lora_path"],
        ),
    ]

    # Создаём тестовые промпты
    test_prompts = create_test_prompts(adapters)

    # Обрабатываем запросы
    print("\n🚀 Генерируем ответы...")
    results = process_requests(engine, test_prompts)

    # Выводим сравнение
    print_comparison(results, adapters)

    # Итоговая статистика
    print("\n" + "=" * 80)
    print("📊 ИТОГО:")
    print("=" * 80)
    print(f"  🔵 BASE     - базовая LLaMA-7B без дообучения")
    print(f"  🟢 FLAN-LoRA - дообучена на FLAN инструкциях")
    print(f"  🟣 CHAT-LoRA - наш адаптер на Guanaco диалогах")
    print()

    # Очищаем память
    del engine
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()