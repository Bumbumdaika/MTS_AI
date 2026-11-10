"""Composition root; independent of HTTP or CLI."""
import os
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
        documents = retriever.search(question, history)
        if not documents:
            return 'В базе знаний не нашлось подходящей информации. Уточните вопрос или оставьте заявку менеджеру кнопкой «Связаться с менеджером».'
        if mode == 'llm':
            return generate_answer(question, documents, history)
        # Explicit fallback, no pretending this is a model response.
        return ('Демонстрационный режим без LLM. Найденные материалы:\n\n'
                + '\n\n'.join(d['title'] + ': ' + d['text'] for d in documents)
                + '\n\nДля обсуждения внедрения можно оставить заявку кнопкой «Связаться с менеджером».')

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
