"""Tasks 4/5: isolated history, resumable contact collection, lead summary."""
import re
import threading
import uuid
from storage import now

PROMPTS = {
    'full_name': 'Укажите, пожалуйста, ФИО для заявки.',
    'phone': 'Укажите телефон с кодом страны, например +7 999 123-45-67.',
    'company': 'Как называется ваша компания?',
}
CONFIRM = {'да', 'подтверждаю', 'подтвердить', 'подтвердить заявку', 'отправить', '/confirm'}
CANCEL = {'отмена', '/cancel', 'отменить заявку'}


def valid_name(text):
    parts = text.split()
    return 2 <= len(parts) <= 5 and all(re.fullmatch(r"[^\W\d_]+(?:[-’'][^\W\d_]+)*", p) for p in parts)


def phone_number(text):
    if not re.fullmatch(r'\+?[\d\s()\-]+', text):
        return None
    digits = re.sub(r'\D', '', text)
    if len(digits) == 11 and digits.startswith('8'):
        digits = '7' + digits[1:]
    if not 10 <= len(digits) <= 15 or digits[0] == '0':
        return None
    return '+' + digits


def summarize(history):
    # Only consultation turns are used. Name/phone/company aren't mistaken for interests.
    questions = [m['content'] for m in history if m['role'] == 'user' and m.get('kind') == 'consultation']
    if not questions:
        return 'Пользователь запросил связь с менеджером. Потребности пока не уточнены.'
    return 'Обращения пользователя в ходе консультации:\n' + '\n'.join(
        '- ' + q[:350] for q in questions[-8:])


class Dialogue:
    def __init__(self, store, answer):
        self.store, self.answer = store, answer
        # One-process demo: serialize state transitions to prevent duplicate confirmations.
        self.lock = threading.RLock()

    def new(self):
        return self.view(self.store.create())

    def view(self, state):
        return {k: state[k] for k in ('id', 'history', 'stage', 'fields', 'lead_id')}

    def message(self, session_id, text):
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise ValueError('Сообщение должно содержать от 1 до 4000 символов.')
        text = text.strip()
        with self.lock:
            state = self.store.get(session_id)
            if len(state['history']) >= 400:
                raise ValueError('Диалог достиг лимита. Начните новый диалог.')
            lower = text.lower().strip(' .!')
            kind, lead = 'lead', None
            if lower in CANCEL:
                state['stage'], state['fields'] = None, {}
                reply = 'Сбор заявки отменён. Можем продолжить консультацию.'
            elif lower == '/lead' or (not state['stage'] and any(
                    phrase in lower for phrase in ('связаться с менеджером', 'оставить заявку', 'хочу менеджера'))):
                if state['lead_id']:
                    reply = 'В этом диалоге заявка уже сохранена: ' + state['lead_id']
                else:
                    state['stage'] = 'full_name'
                    state['fields'] = {}
                    reply = PROMPTS['full_name'] + ' Можно отменить сбор командой /cancel. Для вопроса во время сбора используйте /ask ваш вопрос.'
            elif text.startswith('/ask ') or self.is_question(text, state):
                kind = 'consultation'
                question = text[5:].strip() if text.startswith('/ask ') else text
                if not question:
                    raise ValueError('Введите вопрос после /ask.')
                reply, debug = self._answer(question, state['history'])
                if state['stage']:
                    reply += '\n\n' + (PROMPTS.get(state['stage']) or 'Для сохранения заявки нажмите «Подтвердить заявку».')
            elif state['stage'] == 'full_name':
                if not valid_name(text):
                    reply = 'Введите фамилию и имя, а также отчество, если оно есть. Например: Иванов Иван Иванович.'
                else:
                    state['fields']['full_name'] = text
                    state['stage'] = 'phone'
                    reply = PROMPTS['phone']
            elif state['stage'] == 'phone':
                phone = phone_number(text)
                if phone is None:
                    reply = 'Не удалось распознать телефон. Введите 10–15 цифр с кодом страны, например +7 999 123-45-67.'
                else:
                    state['fields']['phone'] = phone
                    state['stage'] = 'company'
                    reply = PROMPTS['company']
            elif state['stage'] == 'company':
                if len(text) < 2 or len(text) > 200 or not any(c.isalnum() for c in text):
                    reply = 'Введите название компании длиной от 2 до 200 символов.'
                else:
                    state['fields']['company'] = text
                    state['stage'] = 'confirm'
                    f = state['fields']
                    reply = (f"Проверьте заявку:\nФИО: {f['full_name']}\nТелефон: {f['phone']}\nКомпания: {f['company']}\n\n"
                             + summarize(state['history']) + '\n\nПодтвердите сохранение заявки и согласие на передачу этих данных менеджеру для связи: «Подтвердить заявку» или «да». Для исправления начните сбор заново: /lead.')
            elif state['stage'] == 'confirm':
                if lower in CONFIRM:
                    lead_id = uuid.uuid4().hex
                    lead = dict(id=lead_id, session_id=state['id'], **state['fields'],
                                summary=summarize(state['history']), created_at=now())
                    state['lead_id'], state['stage'] = lead_id, None
                    reply = 'Заявка сохранена. Номер: ' + lead_id + '. Автоматическая отправка менеджеру в этой версии не подключена.'
                else:
                    reply = 'Для сохранения нажмите «Подтвердить заявку» или напишите «да». /lead — заполнить заново; /cancel — отменить.'
            elif state['lead_id'] and lower in CONFIRM:
                reply = 'Заявка уже сохранена. Номер: ' + state['lead_id']
            else:
                kind = 'consultation'
                reply, debug = self._answer(text, state['history'])
            assistant = dict(role='assistant', content=reply, kind=kind)
            if kind == 'consultation' and debug:
                assistant['debug'] = debug
            state['history'].extend([
                dict(role='user', content=text, kind=kind),
                assistant,
            ])
            self.store.save(state, lead)
            return self.view(state)

    @staticmethod
    def is_question(text, state):
        if not state['stage']:
            return False
        return '?' in text or bool(re.match(r'^(как|что|сколько|можно ли|расскажите|а какие|какие|зачем|почему)\b', text.lower()))

    def _answer(self, question, history):
        result = self.answer(question, history)
        if isinstance(result, dict):
            return str(result.get('content', '')), result.get('debug')
        return str(result), None
