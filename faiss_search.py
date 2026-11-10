
import json
import numpy as np

from pathlib import Path
from FlagEmbedding import BGEM3FlagModel


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "knowledge.json"
INDEX_DIR = BASE_DIR / "indexes"

EMBEDDINGS_PATH = INDEX_DIR / "document_embeddings.npy"
DOCUMENTS_PATH = INDEX_DIR / "documents.json"


print("Загружаем базу знаний...", flush=True)

with open(DATA_PATH, "r", encoding="utf-8") as file:
    documents = json.load(file)

if not documents:
    raise ValueError("База знаний пуста!")

print(f"Загружено документов: {len(documents)}", flush=True)



if not EMBEDDINGS_PATH.exists():
    raise FileNotFoundError(
        "Не найдены эмбеддинги. Сначала запусти bge_search.py"
    )

if not DOCUMENTS_PATH.exists():
    raise FileNotFoundError(
        "Не найден documents.json. Сначала запусти bge_search.py"
    )

with open(DOCUMENTS_PATH, "r", encoding="utf-8") as file:
    saved_documents = json.load(file)

if saved_documents != documents:
    raise ValueError(
        "База знаний изменилась. Сначала обнови эмбеддинги через bge_search.py"
    )

document_embeddings = np.load(
    EMBEDDINGS_PATH,
    allow_pickle=False
)

if (
    document_embeddings.ndim != 2
    or document_embeddings.shape[0] != len(documents)
):
    raise ValueError(
        "Количество эмбеддингов не совпадает с количеством документов"
    )

document_embeddings = np.asarray(
    document_embeddings,
    dtype=np.float32
)

if not np.isfinite(document_embeddings).all():
    raise ValueError("В эмбеддингах обнаружены некорректные значения")

print("Эмбеддинги загружены!", flush=True)



document_norms = np.linalg.norm(
    document_embeddings,
    axis=1,
    keepdims=True
)

if np.any(document_norms == 0):
    raise ValueError("Обнаружен нулевой эмбеддинг документа")

document_embeddings = document_embeddings / document_norms

print("Эмбеддинги подготовлены!", flush=True)


print("Загружаем BGE-M3...", flush=True)

model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=False
)

print("BGE-M3 успешно загружена!", flush=True)



def search_documents(query, top_k=2):

    print("\nСоздаём эмбеддинг вопроса...", flush=True)

    result = model.encode(
        [query],
        max_length=512
    )

    query_embedding = np.asarray(
        result["dense_vecs"],
        dtype=np.float32
    )

    if query_embedding.shape[1] != document_embeddings.shape[1]:
        raise ValueError(
            "Размерность эмбеддинга вопроса не совпадает с документами"
        )

    if not np.isfinite(query_embedding).all():
        raise ValueError("Эмбеддинг вопроса содержит некорректные значения")

    print(
        "Размер эмбеддинга:",
        query_embedding.shape,
        flush=True
    )

    # Нормализуем эмбеддинг вопроса
    query_norm = np.linalg.norm(
        query_embedding,
        axis=1,
        keepdims=True
    )

    if np.any(query_norm == 0):
        raise ValueError("Получен нулевой эмбеддинг вопроса")

    query_embedding = query_embedding / query_norm

    # Вычисляем косинусное сходство
    print("Выполняем семантический поиск...", flush=True)

    scores = query_embedding @ document_embeddings.T

    # Выбираем наиболее подходящие документы
    top_indices = np.argsort(scores[0])[::-1][:top_k]

    results = []

    for idx in top_indices:

        results.append({
            "title": documents[idx]["title"],
            "text": documents[idx]["text"],
            "score": round(float(scores[0][idx]), 4)
        })

    print("Поиск завершён!", flush=True)

    return results



if __name__ == "__main__":

    print(" ENERGYTOOL: СЕМАНТИЧЕСКИЙ ПОИСК")

    while True:

        question = input(
            "\nВведите вопрос (0 для выхода): "
        ).strip()

        if question.lower() == "0":
            print("Программа завершена.")
            break

        if not question:
            print("Введите непустой вопрос.")
            continue

        try:

            results = search_documents(question)

            print("\nНАЙДЕННЫЕ ДОКУМЕНТЫ:\n")

            for number, result in enumerate(results, start=1):

                print(f"Документ №{number}")
                print("Название:", result["title"])
                print("Текст:", result["text"])
                print("Сходство:", result["score"])
                print("-" * 45)

        except Exception as error:

            print("Ошибка при поиске:", error)
