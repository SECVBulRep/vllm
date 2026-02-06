"""
Скрипт для построения Q&A датасета из Redmine Wiki (PostgreSQL)
Формат: ShareGPT (для Qwen + LLaMA-Factory)

Локальная LLM: openai/gpt-oss-20b на kurchatov-mini:8000
Страницы анализируются ЦЕЛИКОМ (без разбивки на чанки).

Использование:
  pip install psycopg2-binary requests
  python redmine_wiki_dataset.py --output dataset.json
"""

import json
import argparse
import re
import textwrap
import time
import sys

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
    "max_tokens": 2000,
    "top_p": 1.0,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}

# Системный промпт для итогового Q&A бота
SYSTEM_PROMPT = (
    "Ты — ассистент по внутренней базе знаний компании. "
    "Отвечай на вопросы точно, опираясь на документацию. "
    "Если информация отсутствует в базе знаний, сообщи об этом."
)

# Сколько Q&A пар просить у LLM на одну страницу
QA_PER_PAGE_MIN = 3
QA_PER_PAGE_MAX = 7

# Пауза между запросами к LLM (секунды)
LLM_DELAY = 1.0

# ============================================================
# 2. ИЗВЛЕЧЕНИЕ ДАННЫХ ИЗ POSTGRESQL
# ============================================================
WIKI_QUERY = """
SELECT
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
ORDER BY p.name, wp.title LIMIT 5;
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
# 3. ОЧИСТКА REDMINE WIKI-РАЗМЕТКИ
# ============================================================
def clean_wiki_text(text: str) -> str:
    """Убирает Redmine/Textile разметку, оставляя чистый текст."""
    if not text:
        return ""

    # Макросы {{...}}
    text = re.sub(r'\{\{.*?\}\}', '', text)

    # Заголовки h1. h2. h3.
    text = re.sub(r'h[1-6]\.\s*', '', text)

    # Жирный *text* и _курсив_
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)

    # Ссылки [[Page|text]] и [[Page]]
    text = re.sub(r'\[\[([^|\]]+)\|([^\]]+)\]\]', r'\2', text)
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)

    # Внешние ссылки "text":url
    text = re.sub(r'"([^"]+)":\S+', r'\1', text)

    # Код <pre>, <code>
    text = re.sub(r'</?(?:pre|code)[^>]*>', '', text)

    # HTML-теги
    text = re.sub(r'<[^>]+>', '', text)

    # Списки: * или # в начале строки
    text = re.sub(r'^[*#]+\s*', '- ', text, flags=re.MULTILINE)

    # Таблицы |_. (заголовки)
    text = re.sub(r'\|_\.', '', text)

    # Лишние пробелы
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    return text


# ============================================================
# 4. ГЕНЕРАЦИЯ Q&A ЧЕРЕЗ ЛОКАЛЬНУЮ LLM
# ============================================================
def call_llm(prompt: str) -> str:
    """Отправляет запрос к локальной LLM на kurchatov-mini."""
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
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_json_from_llm(text: str) -> list[dict]:
    """Извлекает JSON-массив из ответа LLM (даже если обёрнут в markdown)."""
    # Убираем ```json ... ```
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    # Ищем массив [ ... ]
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        text = match.group(0)

    return json.loads(text)


def generate_qa_for_page(page: dict) -> list[dict]:
    """
    Отправляет ПОЛНЫЙ текст страницы в LLM и получает Q&A пары.
    """
    title = page["page_title"].replace("_", " ")
    project = page["project_name"]
    raw_text = page["page_content"] or ""
    clean_text = clean_wiki_text(raw_text)

    if len(clean_text) < 30:
        return []

    # Кол-во пар в зависимости от длины текста
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
           Примеры типов вопросов:
           - "Что такое ...?"
           - "Как сделать ...?"
           - "Какие шаги нужны для ...?"
           - "Кто отвечает за ...?"
           - "Где найти информацию о ...?"
        2. Вопросы — такие, какие реально задал бы сотрудник компании.
        3. Ответы должны быть полными и основываться ТОЛЬКО на тексте статьи.
        4. Ответы — развёрнутые, информативные, не менее 2-3 предложений.
        5. Не выдумывай информацию, которой нет в тексте.

        Верни ТОЛЬКО валидный JSON массив, без пояснений и комментариев:
        [
          {{"question": "...", "answer": "..."}},
          {{"question": "...", "answer": "..."}}
        ]
    """)

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
        return results

    except json.JSONDecodeError as e:
        print(f"\n    ⚠ Не удалось распарсить JSON для '{title}': {e}")
        return fallback_template(page)
    except Exception as e:
        print(f"\n    ⚠ Ошибка LLM для '{title}': {e}")
        return fallback_template(page)


# ============================================================
# 5. ФОЛЛБЭК — ШАБЛОННЫЕ ВОПРОСЫ (если LLM не ответила)
# ============================================================
FALLBACK_TEMPLATES = [
    "Что такое {title}?",
    "Расскажи про {title}.",
    "Какая информация есть по теме «{title}»?",
]


def fallback_template(page: dict) -> list[dict]:
    """Генерация по шаблонам, если LLM недоступна."""
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
# 6. ФОРМИРОВАНИЕ ShareGPT ЗАПИСИ
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
# 7. ОСНОВНОЙ ПАЙПЛАЙН
# ============================================================
def build_dataset(skip_llm: bool = False) -> list[dict]:
    print("📦 Подключение к PostgreSQL (irinka.webs.ru / wiki_production)...")
    pages = fetch_wiki_pages(DB_CONFIG)
    print(f"📄 Найдено wiki-страниц: {len(pages)}")

    if not pages:
        print("❌ Страниц не найдено. Проверьте подключение и данные в БД.")
        return []

    # Проверяем доступность LLM
    if not skip_llm:
        print(f"🤖 Проверка LLM (kurchatov-mini:8000)...")
        try:
            test = call_llm("Ответь одним словом: работает?")
            print(f"   ✅ LLM доступна: {test[:50]}...")
        except Exception as e:
            print(f"   ❌ LLM недоступна: {e}")
            print("   Переключаюсь на шаблонный режим.")
            skip_llm = True

    dataset = []
    errors = 0

    for i, page in enumerate(pages):
        title = page["page_title"]
        text = page["page_content"] or ""
        print(f"  [{i+1}/{len(pages)}] {page['project_name']} / {title} ({len(text)} симв.) ", end="")

        if skip_llm:
            examples = fallback_template(page)
        else:
            examples = generate_qa_for_page(page)
            if not examples:
                errors += 1
            time.sleep(LLM_DELAY)

        dataset.extend(examples)
        print(f"→ {len(examples)} Q&A")

    print(f"\n{'='*50}")
    print(f"✅ Всего Q&A примеров: {len(dataset)}")
    if errors:
        print(f"⚠  Ошибок LLM (использован фоллбэк): {errors}")

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Redmine Wiki → Q&A Dataset для Qwen LoRA")
    parser.add_argument("--output", default="dataset.json",
                        help="Путь к выходному файлу (по умолчанию: dataset.json)")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Пропустить LLM, использовать только шаблоны")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Пауза между запросами к LLM, сек (по умолчанию: 1.0)")
    parser.add_argument("--qa-min", type=int, default=3,
                        help="Минимум Q&A пар на страницу (по умолчанию: 3)")
    parser.add_argument("--qa-max", type=int, default=7,
                        help="Максимум Q&A пар на страницу (по умолчанию: 7)")
    args = parser.parse_args()

    global LLM_DELAY, QA_PER_PAGE_MIN, QA_PER_PAGE_MAX
    LLM_DELAY = args.delay
    QA_PER_PAGE_MIN = args.qa_min
    QA_PER_PAGE_MAX = args.qa_max

    dataset = build_dataset(skip_llm=args.skip_llm)

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
        # Обрезаем ответ для вывода
        ans = example["conversations"][2]["value"]
        if len(ans) > 300:
            example["conversations"][2]["value"] = ans[:300] + "..."
        print(json.dumps(example, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()