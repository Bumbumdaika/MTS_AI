"""Composition root; independent of HTTP or CLI."""
import os
import time
from config import ROOT, load_env
from dialogue import Dialogue
from retrieval import Retriever
from storage import Store


def build_service(db_path=None):
    load_env()
    mode = os.getenv('BOT_MODE', 'demo')
    if mode not in ('demo', 'llm'):
        raise ValueError('BOT_MODE: допустимы demo и llm')
    retriever = Retriever(os.getenv('SEARCH_BACKEND', 'basic'))
    if mode == 'llm':
        from rag_generator import generate_answer, get_api_config
        get_api_config()  # Fail early for missing configuration.

    def answer(question, history):
        started = time.perf_counter()
        documents = retriever.search(question, history)
        if not documents:
            content = ('В предоставленной базе знаний нет ответа на этот вопрос. '
                       'Уточните задачу или свяжитесь со специалистом кнопкой «Связаться с менеджером».')
        elif mode == 'llm':
            content = generate_answer(question, documents, history)
        else:
            # Explicit fallback, no pretending this is a model response.
            content = ('Демонстрационный режим без LLM. Найденные материалы:\n\n'
                       + '\n\n'.join(d['title'] + ': ' + d['text'] for d in documents)
                       + '\n\nДля обсуждения внедрения можно оставить заявку кнопкой «Связаться с менеджером».')
        return {
            'content': content,
            'debug': {
                'elapsed_ms': round((time.perf_counter() - started) * 1000, 1),
                'documents': [
                    {
                        'title': d.get('title', 'Без названия'),
                        'score': d.get('score'),
                        'excerpt': (d.get('text') or d.get('content') or '')[:300],
                    }
                    for d in documents
                ],
            },
        }

    return Dialogue(Store(db_path or os.getenv('BOT_DB') or ROOT / 'runtime/bot.sqlite3'), answer)


if __name__ == '__main__':
    service = build_service()
    state = service.new()
    print('Energytool. /lead — заявка, /new — новый диалог, /exit — выход.')
    while True:
        try:
            text = input('Вы: ').strip()
            if text == '/exit':
                break
            if text == '/new':
                state = service.new()
                print('Начат новый диалог.')
                continue
            state = service.message(state['id'], text)
            print('Бот:', state['history'][-1]['content'])
        except (ValueError, RuntimeError) as error:
            print('Ошибка:', error)
        except (EOFError, KeyboardInterrupt):
            break
