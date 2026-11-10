
import os
import json
import time
import urllib.request
import urllib.error


def get_api_config():

    base_url = (
        os.getenv("LLM_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
    )

    api_key = (
        os.getenv("LLM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )

    model = (
        os.getenv("LLM_MODEL")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("GEN_MODEL")
    )

    if not base_url:
        raise RuntimeError("Не задан LLM_BASE_URL")

    if not api_key:
        raise RuntimeError("Не задан LLM_API_KEY")

    if not model:
        raise RuntimeError("Не задана модель LLM_MODEL")

    return base_url.rstrip("/"), api_key, model


def format_documents(documents):

    formatted = []

    for i, doc in enumerate(documents, start=1):
        if isinstance(doc, dict):
            title = doc.get("title", f"Документ {i}")
            content = (
                doc.get("content")
                or doc.get("text")
                or doc.get("description")
                or ""
            )
        else:
            title = f"Документ {i}"
            content = str(doc)

        formatted.append(
            f"Источник: {title}\n"
            f"Содержание: {content}"
        )

    return "\n\n".join(formatted)


def generate_answer(question, documents):

    base_url, api_key, model = get_api_config()

    context = format_documents(documents)

    system_prompt = """
Ты — ИИ-консультант сайта MTS Energytool.

Твоя задача — помогать пользователям разбираться
в продуктах и возможностях Energytool.

Правила:

1. Отвечай на основании предоставленных документов.
2. Не придумывай функции, характеристики, цены и условия.
3. Если информации недостаточно, честно скажи об этом.
4. Если вопрос неоднозначный, задай один конкретный
   уточняющий вопрос.
5. Объясняй сложные технические вещи простым языком.
6. Не утверждай, что продукт обладает возможностью,
   если этого нет в документах.
7. Отвечай по-русски.
"""

    user_prompt = f"""
ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{question}

НАЙДЕННЫЕ ДОКУМЕНТЫ:
{context}

Подготовь ответ пользователю.
"""

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": float(os.getenv("TEMPERATURE", "0.2")),
        "max_tokens": int(os.getenv("MAX_TOKENS", "700"))
    }

    url = f"{base_url}/chat/completions"

    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        method="POST"
    )

    # Ограничиваем число попыток при временных ошибках.
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))

            return result["choices"][0]["message"]["content"].strip()

        except urllib.error.HTTPError as error:
            # Ошибки авторизации повторять бессмысленно.
            if error.code in (401, 403):
                raise RuntimeError(
                    f"Ошибка авторизации API: HTTP {error.code}. "
                    "Проверьте настройки подключения у организаторов."
                ) from error

            if error.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(
                    f"Ошибка API: HTTP {error.code}"
                ) from error

            if attempt == 2:
                raise RuntimeError(
                    "API не ответил после трёх попыток."
                ) from error

            time.sleep(2 ** attempt)

        except (TimeoutError, urllib.error.URLError) as error:
            if attempt == 2:
                raise RuntimeError(
                    "Не удалось подключиться к API после трёх попыток."
                ) from error

            time.sleep(2 ** attempt)

    raise RuntimeError("Не удалось получить ответ от модели.")

