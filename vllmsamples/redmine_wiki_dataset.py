"""
Скрипт для построения Q&A датасета из Redmine Wiki (PostgreSQL)
Формат: ShareGPT (для Qwen + LLaMA-Factory)

Локальная LLM: openai/gpt-oss-20b на kurchatov-mini:8000
Страницы анализируются ЦЕЛИКОМ.

Поддержка инкрементальной работы:
  - Запоминает обработанные страницы в progress.json
  - При повторном запуске пропускает уже обработанные
  - Новые Q&A дописываются к существующему датасету
  - --reset для очистки прогресса и начала с нуля

Использование:
  pip install psycopg2-binary requests
  python redmine_wiki_dataset.py --output dataset.json
  python redmine_wiki_dataset.py --output dataset.json          # повторный запуск — обработает только новые
  python redmine_wiki_dataset.py --output dataset.json --reset  # начать с нуля
"""

import json
import argparse
import re
import textwrap
import time
import sys
import os
import hashlib
from pathlib import Path

# ============================================================
# 1. НАСТРОЙКИ
# ============================================================
DB_CONFIG = {
    "host": "irinka.webs.ru",
    "port": 5432,
    "dbname": "wiki_production",
    "user": "bulat",
    "password": "1234567809",
}

LLM_CONFIG = {
    "url": "http://kurchatov-mini:8000/v1/chat/completions",
    "model": "openai/gpt-oss-20b",
    "temperature": 0.1,
    "max_tokens": 10000,
    "top_p": 1.0,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}

SYSTEM_PROMPT = (
    "Ты — ассистент по внутренней базе знаний компании. "
    "Отвечай на вопросы точно, опираясь на документацию. "
    "Если информация отсутствует в базе знаний, сообщи об этом."
)

QA_PER_PAGE_MIN = 3
QA_PER_PAGE_MAX = 10
LLM_DELAY = 1.0
LLM_RETRIES = 2  # повторные попытки при ошибке парсинга JSON

# ============================================================
# 2. ПРОГРЕСС — ЗАПОМИНАНИЕ ОБРАБОТАННЫХ СТРАНИЦ
# ============================================================
class ProgressTracker:
    """
    Хранит информацию об обработанных страницах.
    Ключ = page_id (из БД) + hash содержимого.
    Если содержимое страницы изменилось — она будет переобработана.
    """

    def __init__(self, progress_file: str):
        self.progress_file = progress_file
        self.data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {"processed": {}}
        return {"processed": {}}

    def save(self):
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def is_processed(self, page_key: str, content_hash: str) -> bool:
        """Проверяет, обработана ли страница с таким же содержимым."""
        entry = self.data.get("processed", {}).get(page_key)
        if entry and entry.get("content_hash") == content_hash:
            return True
        return False

    def mark_processed(self, page_key: str, content_hash: str, qa_count: int):
        """Отмечает страницу как обработанную."""
        if "processed" not in self.data:
            self.data["processed"] = {}
        self.data["processed"][page_key] = {
            "content_hash": content_hash,
            "qa_count": qa_count,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.save()

    def reset(self):
        """Очищает весь прогресс."""
        self.data = {"processed": {}}
        self.save()

    @property
    def total_processed(self) -> int:
        return len(self.data.get("processed", {}))


def content_hash(text: str) -> str:
    """MD5-хеш содержимого страницы."""
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()


def page_key(page: dict) -> str:
    """Уникальный ключ страницы: project_id/page_title."""
    return f"{page['project_id']}/{page['page_title']}"


# ============================================================
# 3. ИЗВЛЕЧЕНИЕ ДАННЫХ ИЗ POSTGRESQL
# ============================================================
WIKI_QUERY = """
SELECT DISTINCT ON (wp.id)
    wp.id                           AS wp_id,
    p.name                          AS project_name,
    p.identifier                    AS project_id,
    wp.title                        AS page_title,
    wc.text                         AS page_content,
    wc.updated_on                   AS updated_on,
    COALESCE(u.firstname || ' ' || u.lastname, '') AS author,
    wp.parent_id                    AS parent_page_id,
    parent_wp.title                 AS parent_page_title
FROM wiki_contents wc
JOIN wiki_pages wp        ON wc.page_id = wp.id
JOIN wikis w              ON wp.wiki_id = w.id
JOIN projects p           ON w.project_id = p.id
LEFT JOIN users u         ON wc.author_id = u.id
LEFT JOIN wiki_pages parent_wp ON wp.parent_id = parent_wp.id
WHERE wp.deleted_at IS NULL
  AND wc.text IS NOT NULL
  AND LENGTH(TRIM(wc.text)) > 50
ORDER BY wp.id, wc.version DESC;
"""


def fetch_wiki_pages(db_config: dict) -> list[dict]:
    """Извлекает wiki-страницы из Redmine PostgreSQL."""
    import psycopg2
    import psycopg2.extras

    conn = psycopg2.connect(**db_config)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(WIKI_QUERY)
            rows = cur.fetchall()
            return [dict(r) for r in rows]
    finally:
        conn.close()


# ============================================================
# 4. ОЧИСТКА REDMINE WIKI-РАЗМЕТКИ
# ============================================================
def clean_wiki_text(text: str) -> str:
    """Убирает Redmine/Textile разметку, оставляя чистый текст."""
    if not text:
        return ""

    text = re.sub(r'\{\{.*?\}\}', '', text)
    text = re.sub(r'h[1-6]\.\s*', '', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    text = re.sub(r'\[\[([^|\]]+)\|([^\]]+)\]\]', r'\2', text)
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)
    text = re.sub(r'"([^"]+)":\S+', r'\1', text)
    text = re.sub(r'</?(?:pre|code)[^>]*>', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'^[*#]+\s*', '- ', text, flags=re.MULTILINE)
    text = re.sub(r'\|_\.', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    return text


# ============================================================
# 5. ГЕНЕРАЦИЯ Q&A ЧЕРЕЗ ЛОКАЛЬНУЮ LLM
# ============================================================
def call_llm(prompt: str) -> str:
    """Отправляет запрос к локальной LLM."""
    import requests

    payload = {
        "model": LLM_CONFIG["model"],
        "messages": [
            {
                "role": "system",
                "content": "Ты — эксперт по созданию обучающих датасетов. Генерируй только валидный JSON."
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": LLM_CONFIG["temperature"],
        "max_tokens": LLM_CONFIG["max_tokens"],
        "top_p": LLM_CONFIG["top_p"],
        "frequency_penalty": LLM_CONFIG["frequency_penalty"],
        "presence_penalty": LLM_CONFIG["presence_penalty"],
    }

    resp = requests.post(
        LLM_CONFIG["url"],
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_json_from_llm(text: str) -> list[dict]:
    """Извлекает JSON-массив из ответа LLM."""
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        text = match.group(0)

    return json.loads(text)


def generate_qa_for_page(page: dict) -> list[dict]:
    """Отправляет ПОЛНЫЙ текст страницы в LLM и получает Q&A пары."""
    title = page["page_title"].replace("_", " ")
    project = page["project_name"]
    raw_text = page["page_content"] or ""
    clean_text = clean_wiki_text(raw_text)

    if len(clean_text) < 30:
        return []

    text_len = len(clean_text)
    if text_len < 500:
        num_qa = QA_PER_PAGE_MIN
    elif text_len < 2000:
        num_qa = 5
    else:
        num_qa = QA_PER_PAGE_MAX

    prompt = textwrap.dedent(f"""\
        Ты создаёшь обучающий датасет для AI-ассистента по внутренней базе знаний.

        Ниже — полная статья из внутренней wiki.
        Проект: {project}
        Страница: {title}

        === ТЕКСТ СТАТЬИ ===
        {clean_text}
        === КОНЕЦ СТАТЬИ ===

        На основе этой статьи сгенерируй ровно {num_qa} пар "вопрос-ответ".

        Требования:
        1. Вопросы должны быть разнообразными: общие, конкретные, практические.
        2. Вопросы — такие, какие реально задал бы сотрудник компании.
        3. Ответы должны быть полными и основываться ТОЛЬКО на тексте статьи.
        4. Ответы — развёрнутые, информативные, не менее 2-3 предложений.
        5. Не выдумывай информацию, которой нет в тексте.

        ВАЖНО: Верни ТОЛЬКО валидный JSON массив. Никакого текста до или после.
        Убедись что все строки корректно закрыты кавычками.
        [
          {{"question": "...", "answer": "..."}},
          {{"question": "...", "answer": "..."}}
        ]
    """)

    # Попытки с ретраями
    for attempt in range(LLM_RETRIES + 1):
        try:
            response = call_llm(prompt)
            qa_pairs = parse_json_from_llm(response)

            results = []
            for qa in qa_pairs:
                q = qa.get("question", "").strip()
                a = qa.get("answer", "").strip()
                if q and a and len(a) > 20:
                    results.append(make_sharegpt_entry(
                        question=q,
                        answer=a,
                        project=project,
                        page_title=title,
                    ))
            if results:
                return results

        except json.JSONDecodeError as e:
            if attempt < LLM_RETRIES:
                print(f"\n    ⚠ JSON ошибка (попытка {attempt+1}/{LLM_RETRIES+1}), повтор...", end="")
                time.sleep(LLM_DELAY)
            else:
                print(f"\n    ⚠ Не удалось распарсить JSON для '{title}' после {LLM_RETRIES+1} попыток")
                return fallback_template(page)
        except Exception as e:
            print(f"\n    ⚠ Ошибка LLM для '{title}': {e}")
            return fallback_template(page)

    return fallback_template(page)


# ============================================================
# 6. ФОЛЛБЭК — ШАБЛОННЫЕ ВОПРОСЫ
# ============================================================
FALLBACK_TEMPLATES = [
    "Что такое {title}?",
    "Расскажи про {title}.",
    "Какая информация есть по теме «{title}»?",
]


def fallback_template(page: dict) -> list[dict]:
    title = page["page_title"].replace("_", " ")
    project = page["project_name"]
    clean_text = clean_wiki_text(page["page_content"] or "")

    if len(clean_text) < 30:
        return []

    results = []
    for tmpl in FALLBACK_TEMPLATES:
        results.append(make_sharegpt_entry(
            question=tmpl.format(title=title),
            answer=clean_text,
            project=project,
            page_title=title,
        ))
    return results


# ============================================================
# 7. ФОРМИРОВАНИЕ ShareGPT ЗАПИСИ
# ============================================================
def make_sharegpt_entry(question: str, answer: str, project: str, page_title: str) -> dict:
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer},
        ],
        "metadata": {
            "project": project,
            "page": page_title,
        }
    }


# ============================================================
# 8. ЗАГРУЗКА СУЩЕСТВУЮЩЕГО ДАТАСЕТА
# ============================================================
def load_existing_dataset(output_path: str) -> list[dict]:
    """Загружает существующий датасет для дописывания."""
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"📂 Загружен существующий датасет: {len(data)} записей")
            return data
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠ Не удалось загрузить {output_path}: {e}")
            return []
    return []


# ============================================================
# 9. ОСНОВНОЙ ПАЙПЛАЙН
# ============================================================
def build_dataset(skip_llm: bool, output_path: str, progress_file: str) -> list[dict]:
    # Инициализация прогресса
    tracker = ProgressTracker(progress_file)
    print(f"📦 Подключение к PostgreSQL (irinka.webs.ru / wiki_production)...")

    pages = fetch_wiki_pages(DB_CONFIG)
    print(f"📄 Найдено wiki-страниц: {len(pages)}")

    if not pages:
        print("❌ Страниц не найдено.")
        return []

    # Фильтрация: пропускаем уже обработанные
    pages_to_process = []
    pages_skipped = 0
    for page in pages:
        pk = page_key(page)
        ch = content_hash(page["page_content"] or "")
        if tracker.is_processed(pk, ch):
            pages_skipped += 1
        else:
            pages_to_process.append(page)

    print(f"   Уже обработано ранее: {pages_skipped}")
    print(f"   Новых / изменённых:   {len(pages_to_process)}")

    if not pages_to_process:
        print("✅ Все страницы уже обработаны. Нечего делать.")
        print("   Используйте --reset чтобы начать заново.")
        return load_existing_dataset(output_path)

    # Проверяем LLM
    if not skip_llm:
        print(f"🤖 Проверка LLM ({LLM_CONFIG['url']})...")
        try:
            test = call_llm("Ответь одним словом: работает?")
            print(f"   ✅ LLM доступна: {test[:50]}...")
        except Exception as e:
            print(f"   ❌ LLM недоступна: {e}")
            print("   Переключаюсь на шаблонный режим.")
            skip_llm = True

    # Загружаем существующий датасет для дописывания
    dataset = load_existing_dataset(output_path)
    new_count = 0
    errors = 0

    for i, page in enumerate(pages_to_process):
        title = page["page_title"]
        text = page["page_content"] or ""
        pk = page_key(page)
        ch = content_hash(text)

        print(f"  [{i+1}/{len(pages_to_process)}] {page['project_name']} / {title} ({len(text)} симв.) ", end="")

        if skip_llm:
            examples = fallback_template(page)
        else:
            examples = generate_qa_for_page(page)
            if not examples:
                errors += 1
            time.sleep(LLM_DELAY)

        dataset.extend(examples)
        new_count += len(examples)

        # Отмечаем как обработанную
        tracker.mark_processed(pk, ch, len(examples))

        print(f"→ {len(examples)} Q&A ✓")

    print(f"\n{'='*50}")
    print(f"✅ Новых Q&A примеров:  {new_count}")
    print(f"✅ Всего в датасете:    {len(dataset)}")
    if errors:
        print(f"⚠  Ошибок LLM (фоллбэк): {errors}")

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Redmine Wiki → Q&A Dataset для Qwen LoRA")
    parser.add_argument("--output", default="dataset.json",
                        help="Путь к выходному файлу")
    parser.add_argument("--progress", default="progress.json",
                        help="Файл прогресса (по умолчанию: progress.json)")
    parser.add_argument("--llm-url", default=None,
                        help="URL LLM API (например http://172.16.29.232:8000/v1/chat/completions)")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Пропустить LLM, использовать только шаблоны")
    parser.add_argument("--reset", action="store_true",
                        help="Очистить прогресс и начать заново")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Пауза между запросами к LLM, сек")
    parser.add_argument("--qa-min", type=int, default=3,
                        help="Минимум Q&A пар на страницу")
    parser.add_argument("--qa-max", type=int, default=7,
                        help="Максимум Q&A пар на страницу")
    parser.add_argument("--retries", type=int, default=2,
                        help="Повторные попытки при ошибке JSON")
    args = parser.parse_args()

    if args.llm_url:
        LLM_CONFIG["url"] = args.llm_url
        print(f"🔗 LLM URL: {args.llm_url}")

    global LLM_DELAY, QA_PER_PAGE_MIN, QA_PER_PAGE_MAX, LLM_RETRIES
    LLM_DELAY = args.delay
    QA_PER_PAGE_MIN = args.qa_min
    QA_PER_PAGE_MAX = args.qa_max
    LLM_RETRIES = args.retries

    # Сброс прогресса
    if args.reset:
        tracker = ProgressTracker(args.progress)
        tracker.reset()
        if os.path.exists(args.output):
            os.remove(args.output)
        print("🔄 Прогресс и датасет очищены.")

    dataset = build_dataset(
        skip_llm=args.skip_llm,
        output_path=args.output,
        progress_file=args.progress,
    )

    if not dataset:
        print("❌ Датасет пуст.")
        sys.exit(1)

    # Сохранение
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Датасет сохранён: {args.output}")

    # Статистика
    projects = set(e["metadata"]["project"] for e in dataset)
    pages_set = set(e["metadata"]["page"] for e in dataset)
    avg_q = sum(len(e["conversations"][1]["value"]) for e in dataset) / len(dataset)
    avg_a = sum(len(e["conversations"][2]["value"]) for e in dataset) / len(dataset)

    print(f"\n📊 Статистика:")
    print(f"   Проектов:              {len(projects)}")
    print(f"   Wiki-страниц:          {len(pages_set)}")
    print(f"   Q&A пар:               {len(dataset)}")
    print(f"   Средняя длина вопроса: {avg_q:.0f} символов")
    print(f"   Средняя длина ответа:  {avg_a:.0f} символов")

    # Пример
    if dataset:
        print(f"\n📝 Пример записи:")
        example = dataset[0].copy()
        ans = example["conversations"][2]["value"]
        if len(ans) > 300:
            example["conversations"][2]["value"] = ans[:300] + "..."
        print(json.dumps(example, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()