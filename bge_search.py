
import json
import numpy as np

from pathlib import Path
from FlagEmbedding import BGEM3FlagModel
from sklearn.metrics.pairwise import cosine_similarity



BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "knowledge.json"

INDEX_DIR = BASE_DIR / "indexes"

EMBEDDINGS_PATH = INDEX_DIR / "document_embeddings.npy"

DOCUMENTS_PATH = INDEX_DIR / "documents.json"



with open(DATA_PATH, "r", encoding="utf-8") as file:
    documents = json.load(file)

if not documents:
    raise ValueError("База знаний пуста!")

texts = [doc["text"] for doc in documents]

print(f"Загружено документов: {len(documents)}")


print("Загружаем BGE-M3...")

model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=False
)

print("Модель загружена!")


def load_or_create_index():

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    if EMBEDDINGS_PATH.exists() and DOCUMENTS_PATH.exists():

        try:

            with open(DOCUMENTS_PATH, "r", encoding="utf-8") as file:
                saved_documents = json.load(file)

            embeddings = np.load(
                EMBEDDINGS_PATH,
                allow_pickle=False
            )

            if (
                saved_documents == documents
                and embeddings.ndim == 2
                and embeddings.shape[0] == len(documents)
            ):

                print("Загружаем сохранённые эмбеддинги...")
                print("Эмбеддинги успешно загружены!")

                return embeddings

            print("Документы изменились. Создаём эмбеддинги заново.")

        except (OSError, ValueError, json.JSONDecodeError):

            print("Не удалось загрузить сохранённые эмбеддинги.")

    print("Создаём эмбеддинги документов...")

    embeddings = model.encode(
        texts,
        batch_size=4,
        max_length=512
    )["dense_vecs"]

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    np.save(EMBEDDINGS_PATH, embeddings)

    with open(DOCUMENTS_PATH, "w", encoding="utf-8") as file:

        json.dump(
            documents,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("Эмбеддинги сохранены!")
    print("Папка:", INDEX_DIR)

    return embeddings


document_embeddings = load_or_create_index()


def search_documents(query, top_k=2):

    query_embedding = model.encode(
        [query],
        max_length=512
    )["dense_vecs"]

    scores = cosine_similarity(
        query_embedding,
        document_embeddings
    )[0]

    ranked_indices = np.argsort(scores)[::-1]

    results = []

    for index in ranked_indices[:top_k]:

        results.append({
            "title": documents[index]["title"],
            "text": documents[index]["text"],
            "score": round(float(scores[index]), 4)
        })

    return results


if __name__ == "__main__":

    print("\nBGE-M3 поиск запущен!")

    while True:

        question = input("\nВведите вопрос (exit для выхода): ")

        if question.strip().lower() == "exit":
            print("Завершение программы.")
            break

        if not question.strip():
            print("Введите непустой вопрос.")
            continue

        results = search_documents(question)

        print("\nНайденные документы:\n")

        for result in results:

            print("Документ:", result["title"])
            print("Текст:", result["text"])
            print("Сходство:", result["score"])
