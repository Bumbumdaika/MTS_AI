import json
import numpy as np

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from FlagEmbedding import BGEM3FlagModel


DATA_PATH = Path("data/knowledge.json")
EMBEDDINGS_PATH = Path("indexes/document_embeddings.npy")
DOCUMENTS_PATH = Path("indexes/documents.json")

with open(DATA_PATH, "r", encoding="utf-8") as file:
    documents = json.load(file)


if not EMBEDDINGS_PATH.exists() or not DOCUMENTS_PATH.exists():
    raise FileNotFoundError(
        "Эмбеддинги не найдены. Сначала запусти bge_search.py"
    )

with open(DOCUMENTS_PATH, "r", encoding="utf-8") as file:
    saved_documents = json.load(file)

if documents != saved_documents:
    raise ValueError(
        "База знаний изменилась. Пересоздай эмбеддинги через bge_search.py"
    )

document_embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)

if document_embeddings.ndim != 2:
    raise ValueError("Неправильная размерность эмбеддингов")

if document_embeddings.shape[0] != len(documents):
    raise ValueError("Количество эмбеддингов не совпадает с количеством документов")

if not np.isfinite(document_embeddings).all():
    raise ValueError("В эмбеддингах есть некорректные значения")


# 5. Нормализуем эмбеддинги BGE-M3
norms = np.linalg.norm(document_embeddings, axis=1, keepdims=True)
document_embeddings /= np.maximum(norms, 1e-12)


# 6. Создаём TF-IDF индекс
document_texts = [
    document["title"] + ". " + document["text"]
    for document in documents
]

tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    sublinear_tf=True
)

tfidf_matrix = tfidf_vectorizer.fit_transform(document_texts)


# 7. Загружаем BGE-M3
print("Загружаю BGE-M3...")
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
print("BGE-M3 загружена!")


# 8. Функция гибридного поиска
def hybrid_search(query, top_k=2, rrf_k=60):

    # Получаем эмбеддинг вопроса
    query_embedding = model.encode(
        [query],
        max_length=512
    )["dense_vecs"].astype(np.float32)

    # Нормализуем эмбеддинг вопроса
    query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
    query_embedding /= np.maximum(query_norm, 1e-12)

    # Считаем сходство BGE-M3
    semantic_scores = (query_embedding @ document_embeddings.T)[0]

    # Получаем оценки TF-IDF
    query_tfidf = tfidf_vectorizer.transform([query])
    lexical_scores = cosine_similarity(
        query_tfidf,
        tfidf_matrix
    )[0]

    # Ранжируем документы по каждому методу
    semantic_order = np.argsort(semantic_scores)[::-1]
    lexical_order = np.argsort(lexical_scores)[::-1]

    # Объединяем ранги через Reciprocal Rank Fusion
    rrf_scores = np.zeros(len(documents), dtype=np.float32)

    for rank, index in enumerate(semantic_order, start=1):
        rrf_scores[index] += 1 / (rrf_k + rank)

    if np.any(lexical_scores > 0):
        for rank, index in enumerate(lexical_order, start=1):
            rrf_scores[index] += 1 / (rrf_k + rank)

    # Выбираем лучшие документы
    best_indices = np.argsort(rrf_scores)[::-1][:top_k]

    results = []

    for index in best_indices:
        results.append({
            "title": documents[index]["title"],
            "text": documents[index]["text"],
            "semantic_score": float(semantic_scores[index]),
            "tfidf_score": float(lexical_scores[index]),
            "rrf_score": float(rrf_scores[index])
        })

    return results

print("Введи вопрос или 0 для выхода.\n")

while True:
    query = input("Твой вопрос: ").strip()

    if query.lower() == "0":
        print("Поиск завершён.")
        break

    if not query:
        print("Введи вопрос.")
        continue

    try:
        results = hybrid_search(query)

        print("\nЛучшие результаты:\n")

        for i, result in enumerate(results, start=1):
            print(f"{i}. {result['title']}")
            print(f"BGE-M3: {result['semantic_score']:.4f}")
            print(f"TF-IDF: {result['tfidf_score']:.4f}")
            print(f"RRF: {result['rrf_score']:.4f}")
            print(result["text"])
            print()

    except Exception as error:
        print(f"Ошибка поиска: {error}")