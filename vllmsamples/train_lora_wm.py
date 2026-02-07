"""
Обучение Qwen LoRA адаптера на датасете из Redmine Wiki
Затем запуск через vLLM.

Установка:
  pip install torch transformers peft datasets accelerate bitsandbytes trl

Использование:
  # Обучение
  python train_lora.py --dataset dataset.json --model Qwen/Qwen2.5-7B-Instruct

  # После обучения — запуск через vLLM:
  python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --enable-lora \
    --lora-modules wiki-lora=./output/qwen-wiki-lora \
    --port 8000
"""

import json
import argparse
import os
import torch
from pathlib import Path


def load_dataset_sharegpt(dataset_path: str):
    """Загружает ShareGPT датасет и конвертирует в формат для trl."""
    from datasets import Dataset

    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Конвертируем ShareGPT → messages формат
    processed = []
    for entry in raw_data:
        convs = entry["conversations"]
        messages = []
        for msg in convs:
            role_map = {"system": "system", "human": "user", "gpt": "assistant"}
            role = role_map.get(msg["from"], msg["from"])
            messages.append({"role": role, "content": msg["value"]})
        processed.append({"messages": messages})

    dataset = Dataset.from_list(processed)
    print(f"📄 Загружено примеров: {len(dataset)}")
    return dataset


def train(args):
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
    from trl import SFTTrainer, SFTConfig

    print(f"🔧 Модель:   {args.model}")
    print(f"📄 Датасет:  {args.dataset}")
    print(f"💾 Выход:    {args.output_dir}")
    print(f"📊 Эпохи:    {args.epochs}")
    print(f"📊 LoRA r:   {args.lora_rank}")
    print(f"📊 Batch:    {args.batch_size}")
    print()

    # ---- Токенизатор ----
    print("📥 Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Квантизация (опционально) ----
    bnb_config = None
    if args.use_4bit:
        print("⚡ Используется 4-bit квантизация (QLoRA)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # ---- Модель ----
    print("📥 Загрузка модели...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not args.use_4bit else None,
        attn_implementation="flash_attention_2" if args.flash_attn else None,
    )

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    model.config.use_cache = False

    # ---- LoRA ----
    print("🔗 Настройка LoRA...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Датасет ----
    print("📄 Загрузка датасета...")
    dataset = load_dataset_sharegpt(args.dataset)

    # Разделение train/eval
    if len(dataset) > 20:
        split = dataset.train_test_split(test_size=min(0.1, 50 / len(dataset)), seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"   Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    else:
        train_dataset = dataset
        eval_dataset = None
        print(f"   Train: {len(train_dataset)}, Eval: нет (мало данных)")

    # ---- Параметры обучения ----
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_steps=50,
        save_total_limit=3,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=50 if eval_dataset else None,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        max_grad_norm=1.0,
        seed=42,
        max_seq_length=args.max_seq_length,
    )

    # ---- Trainer ----
    print("\n🚀 Начало обучения...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    # ---- Сохранение ----
    print(f"\n💾 Сохранение LoRA адаптера в {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\n✅ Обучение завершено!")
    print(f"\n{'='*60}")
    print(f"📦 LoRA адаптер: {args.output_dir}")
    print(f"\n🚀 Запуск через vLLM:")
    print(f"   python -m vllm.entrypoints.openai.api_server \\")
    print(f"     --model {args.model} \\")
    print(f"     --enable-lora \\")
    print(f"     --lora-modules wiki-lora={args.output_dir} \\")
    print(f"     --port 8000")
    print(f"\n📡 Запрос к API:")
    print(f'   curl http://localhost:8000/v1/chat/completions \\')
    print(f'     -d \'{{"model": "wiki-lora", "messages": [{{"role": "user", "content": "Что такое X9?"}}]}}\'')
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Обучение Qwen LoRA на wiki датасете")

    # Основные
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Базовая модель (по умолчанию: Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--dataset", default="dataset.json",
                        help="Путь к датасету")
    parser.add_argument("--output-dir", default="./output/qwen-wiki-lora",
                        help="Директория для сохранения LoRA")

    # Гиперпараметры обучения
    parser.add_argument("--epochs", type=int, default=3,
                        help="Количество эпох (по умолчанию: 3)")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Размер батча (по умолчанию: 2)")
    parser.add_argument("--gradient-accumulation", type=int, default=8,
                        help="Gradient accumulation steps (по умолчанию: 8)")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate (по умолчанию: 1e-4)")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                        help="Макс. длина последовательности (по умолчанию: 2048)")

    # LoRA
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank (по умолчанию: 16)")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha (по умолчанию: 32)")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout (по умолчанию: 0.05)")

    # Оптимизации
    parser.add_argument("--use-4bit", action="store_true",
                        help="Использовать QLoRA (4-bit квантизация)")
    parser.add_argument("--flash-attn", action="store_true",
                        help="Использовать Flash Attention 2")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()