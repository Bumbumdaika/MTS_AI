"""Small dependency-free baseline; optional original BGE search."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def tokens(text):
    return set(re.findall(r'[а-яёa-z0-9]+', text.lower()))


class Retriever:
    def __init__(self, backend='basic'):
        self.backend = backend
        self.documents = json.loads((ROOT / 'data/knowledge.json').read_text(encoding='utf-8'))
        self.search_bge = None
        if backend == 'bge':
            from bge_search import search_documents
            self.search_bge = search_documents
        elif backend != 'basic':
            raise ValueError('SEARCH_BACKEND: допустимы basic и bge')

    def search(self, question, history):
        # Use prior user topics for contextual follow-ups, but exclude contact fields.
        previous = [m['content'] for m in history if m.get('kind') == 'consultation' and m['role'] == 'user'][-4:]
        query = question + ' ' + ' '.join(previous)
        if self.search_bge:
            return self.search_bge(query, top_k=2)
        words = tokens(query)
        scored = []
        for doc in self.documents:
            matched = words & tokens(doc['title'] + ' ' + doc['text'])
            if matched:
                scored.append(dict(doc, score=len(matched)))
        return sorted(scored, key=lambda d: d['score'], reverse=True)[:2]
