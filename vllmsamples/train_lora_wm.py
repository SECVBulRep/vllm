"""
Обучение Qwen LoRA адаптера на датасете из Redmine Wiki
Совместимо с: trl==0.27.1, transformers==4.57.6, peft==0.18.1

Установка:
  pip install torch transformers peft datasets accelerate bitsandbytes trl

Использование:
  python train_lora.py --dataset dataset.json --model Qwen/Qwen2.5-7B-Instruct --use-4bit

После обучения — запуск через vLLM:
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
from datasets import Dataset
from transformers import BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


def load_dataset_sharegpt(dataset_path: str) -> Dataset:
    """Загружает ShareGPT датасет и конвертирует в messages формат."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    processed = []
    role_map = {"system": "system", "human": "user", "gpt": "assistant"}

    for entry in raw_data:
        messages = []
        for msg in entry["conversations"]:
            role = role_map.get(msg["from"], msg["from"])
            messages.append({"role": role, "content": msg["value"]})
        processed.append({"messages": messages})

    dataset = Dataset.from_list(processed)
    print(f"📄 Загружено примеров: {len(dataset)}")
    return dataset


def train(args):
    print(f"🔧 Модель:   {args.model}")
    print(f"📄 Датасет:  {args.dataset}")
    print(f"💾 Выход:    {args.output_dir}")
    print(f"📊 Эпохи:    {args.epochs}")
    print(f"📊 LoRA r:   {args.lora_rank}")
    print(f"📊 Batch:    {args.batch_size}")
    print()

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

    # ---- Квантизация (QLoRA) ----
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }

    if args.use_4bit:
        print("⚡ Используется 4-bit квантизация (QLoRA)")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # ---- LoRA ----
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    # ---- SFTConfig (trl 0.27+) ----
    sft_config = SFTConfig(
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
        # trl 0.27: max_length вместо max_seq_length
        max_length=args.max_length,
        # Передаём kwargs для AutoModelForCausalLM.from_pretrained
        model_init_kwargs=model_kwargs,
    )

    # ---- Trainer (trl 0.27 API) ----
    print("\n🚀 Загрузка модели и начало обучения...")
    trainer = SFTTrainer(
        model=args.model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    trainer.model.print_trainable_parameters()
    trainer.train()

    # ---- Сохранение ----
    print(f"\n💾 Сохранение LoRA адаптера в {args.output_dir}...")
    trainer.save_model(args.output_dir)

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

    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dataset", default="dataset.json")
    parser.add_argument("--output-dir", default="./output/qwen-wiki-lora")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=2048)

    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--use-4bit", action="store_true")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
