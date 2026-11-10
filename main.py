
import json

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "knowledge.json"


with open(DATA_PATH, "r", encoding="utf-8") as file:
    documents = json.load(file)

texts = [doc["text"] for doc in documents]

print(f"Загружено документов: {len(documents)}")


vectorizer = TfidfVectorizer()

document_vectors = vectorizer.fit_transform(texts)

print("TF-IDF модель создана!")



def search_documents(query, top_k=2):

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        document_vectors
    )[0]

    ranked_indices = scores.argsort()[::-1]

    results = []

    for index in ranked_indices[:top_k]:

        results.append({
            "title": documents[index]["title"],
            "text": documents[index]["text"],
            "score": round(float(scores[index]), 4)
        })

    return results


if __name__ == "__main__":

    print("\nTF-IDF поиск запущен!")

    while True:

        question = input("\nВведите вопрос (0 для выхода): ")

        if question.strip().lower() == "0":
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
