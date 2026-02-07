"""
Тест LoRA адаптера через vLLM API.

Использование:
  1. Запустите vLLM с LoRA:
     python -m vllm.entrypoints.openai.api_server \
       --model Qwen/Qwen2.5-7B-Instruct \
       --enable-lora \
       --lora-modules wiki-lora=./output/qwen-wiki-lora \
       --port 8000

  2. Запустите тест:
     python test_lora.py
     python test_lora.py --url http://172.16.29.232:8000
"""

import requests
import argparse
import json

TEST_QUESTIONS = [
    "Which XML interface is used to create a new purse?",
    "What is Capitaller passport?",
    "How to add funds to BA purses?",
    "What is X9 interface?",
]


def ask(url: str, model: str, question: str) -> str:
    resp = requests.post(
        f"{url}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "Ты — ассистент по внутренней базе знаний компании. "
                               "Отвечай на вопросы точно, опираясь на документацию."
                },
                {"role": "user", "content": question},
            ],
            "temperature": 0.1,
            "max_tokens": 500,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--model", default="wiki-lora",
                        help="Имя модели (wiki-lora для LoRA, или базовая)")
    parser.add_argument("--base-model", default="openai/gpt-oss-20b",
                        help="Базовая модель для сравнения")
    parser.add_argument("--compare", action="store_true",
                        help="Сравнить LoRA с базовой моделью")
    args = parser.parse_args()

    print(f"🧪 Тестирование модели: {args.model}")
    print(f"🔗 URL: {args.url}")
    print(f"{'='*60}\n")

    for q in TEST_QUESTIONS:
        print(f"❓ {q}")

        answer = ask(args.url, args.model, q)
        print(f"✅ [{args.model}]: {answer}\n")

        if args.compare:
            base_answer = ask(args.url, args.base_model, q)
            print(f"🔵 [{args.base_model}]: {base_answer}\n")

        print("-" * 60)


if __name__ == "__main__":
    main()