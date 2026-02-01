# SPDX-License-Identifier: Apache-2.0
"""
Обучение QLoRA адаптера для диалогов/чата
Датасет: mlabonne/guanaco-llama2-1k (1000 качественных диалогов)
Время обучения: ~5-10 минут на 24GB GPU
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig

# Определяем версию trl и импортируем правильно
import trl

TRL_VERSION = tuple(map(int, trl.__version__.split('.')[:2]))

from trl import SFTTrainer

if TRL_VERSION >= (0, 12):
    from trl import SFTConfig

    USE_NEW_API = True
else:
    USE_NEW_API = False

print(f"TRL version: {trl.__version__} (USE_NEW_API={USE_NEW_API})")


def create_bnb_config():
    """Конфигурация 4-bit квантизации."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def create_lora_config():
    """Конфигурация LoRA адаптера."""
    return LoraConfig(
        r=32,  # Ранг (баланс качества и скорости)
        lora_alpha=64,  # Масштабирование
        lora_dropout=0.1,  # Dropout для регуляризации
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[  # Слои для адаптации
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )


def load_model_and_tokenizer(model_name: str, bnb_config: BitsAndBytesConfig):
    """Загружаем модель с квантизацией и токенизатор."""

    print(f"Загружаем модель: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def load_chat_dataset(dataset_name: str):
    """Загружаем датасет для обучения."""

    print(f"Загружаем датасет: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")

    print(f"Размер датасета: {len(dataset)} примеров")
    print(f"Пример диалога:\n{dataset[0]['text'][:500]}...")

    return dataset


def create_training_config(output_dir: str, max_seq_length: int):
    """Настройки обучения для ~10 минут на 24GB GPU."""

    base_args = dict(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        logging_steps=25,
        save_strategy="epoch",
        report_to="none",
    )

    if USE_NEW_API:
        # trl >= 0.12: используем SFTConfig
        return SFTConfig(
            **base_args,
            max_length=max_seq_length,
            packing=False,
            dataset_text_field="text",
        )
    else:
        # trl < 0.12: используем TrainingArguments
        return TrainingArguments(**base_args)


def train_lora(
        model_name: str = "huggyllama/llama-7b",
        dataset_name: str = "mlabonne/guanaco-llama2-1k",
        output_dir: str = "./chat-lora-adapter",
        max_seq_length: int = 512,
):
    """Основная функция обучения."""

    print("=" * 60)
    print("QLoRA Training для диалогов")
    print("=" * 60)

    # Конфигурации
    bnb_config = create_bnb_config()
    lora_config = create_lora_config()

    # Загружаем модель
    model, tokenizer = load_model_and_tokenizer(model_name, bnb_config)

    # Статистика параметров (до применения LoRA)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nВсего параметров базовой модели: {total_params:,}")

    # Загружаем датасет
    dataset = load_chat_dataset(dataset_name)

    # Настройки обучения
    training_config = create_training_config(output_dir, max_seq_length)

    # Создаём тренер (разные API для разных версий trl)
    if USE_NEW_API:
        # trl >= 0.12
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            processing_class=tokenizer,
            args=training_config,
            peft_config=lora_config,
        )
    else:
        # trl < 0.12
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            dataset_text_field="text",
            tokenizer=tokenizer,
            args=training_config,
            peft_config=lora_config,
            max_seq_length=max_seq_length,
            packing=False,
        )

    # Обучаем
    print("\nНачинаем обучение...")
    print("-" * 60)
    trainer.train()

    # Сохраняем адаптер
    print(f"\nСохраняем адаптер в: {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n" + "=" * 60)
    print("Обучение завершено!")
    print(f"Адаптер сохранён в: {output_dir}")
    print("=" * 60)

    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Обучение QLoRA адаптера")
    parser.add_argument(
        "--model",
        type=str,
        default="huggyllama/llama-7b",
        help="Базовая модель"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mlabonne/guanaco-llama2-1k",
        help="Датасет для обучения"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./chat-lora-adapter",
        help="Директория для сохранения адаптера"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Максимальная длина последовательности"
    )

    args = parser.parse_args()

    train_lora(
        model_name=args.model,
        dataset_name=args.dataset,
        output_dir=args.output,
        max_seq_length=args.max_seq_length,
    )
