import json
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dialogue import Dialogue, phone_number
from storage import Store
from retrieval import Retriever
from server import make_handler
import rag_generator


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'bot.sqlite3'
        self.calls = []
        def answer(question, history):
            self.calls.append((question, [dict(m) for m in history]))
            return 'Ответ на: ' + question
        self.bot = Dialogue(Store(self.path), answer)
        self.sid = self.bot.new()['id']

    def tearDown(self):
        self.tmp.cleanup()

    def send(self, text):
        return self.bot.message(self.sid, text)

    def collect(self):
        for text in ['/lead', 'Иванов Иван Иванович', '8 (999) 123-45-67', 'ООО Пример']:
            state = self.send(text)
        return state

    def count_leads(self):
        with self.bot.store.connect() as db:
            return db.execute('SELECT COUNT(*) FROM leads').fetchone()[0]

    def test_confirmation_and_persistent_summary(self):
        self.send('Нужен контроль энергопотребления цеха')
        state = self.collect()
        self.assertEqual(state['stage'], 'confirm')
        self.assertEqual(self.count_leads(), 0)
        state = self.send('да')
        self.assertIsNotNone(state['lead_id'])
        with Store(self.path).connect() as db:
            lead = dict(db.execute('SELECT * FROM leads').fetchone())
        self.assertEqual(lead['phone'], '+79991234567')
        self.assertIn('энергопотребления цеха', lead['summary'])
        self.assertNotIn('Иванов', lead['summary'])
        self.assertEqual(Store(self.path).get(self.sid)['lead_id'], lead['id'])

    def test_repeated_confirmation_and_concurrent_requests(self):
        self.collect()
        threads = [threading.Thread(target=self.send, args=('/confirm',)) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(self.count_leads(), 1)

    def test_invalid_fields_and_cancel(self):
        self.send('/lead')
        self.assertEqual(self.send('Иван123')['stage'], 'full_name')
        self.send('Иванов Иван')
        self.assertEqual(self.send('123')['stage'], 'phone')
        self.send('+7 999 123-45-67')
        self.assertEqual(self.send('!')['stage'], 'company')
        self.assertIsNone(self.send('/cancel')['stage'])
        self.assertEqual(self.count_leads(), 0)

    def test_question_interrupts_collection(self):
        self.send('Хочу мониторинг оборудования')
        self.send('/lead')
        self.send('Иванов Иван')
        state = self.send('/ask А что с анализом расходов?')
        self.assertEqual(state['stage'], 'phone')
        self.assertIn('Укажите телефон', state['history'][-1]['content'])
        self.assertTrue(any('мониторинг' in m['content'] for m in self.calls[-1][1]))
        self.assertEqual(self.send('+79991234567')['stage'], 'company')

    def test_new_session_is_clean_and_other_is_preserved(self):
        self.send('Потребление на заводе')
        other = self.bot.new()
        self.assertEqual(other['history'], [])
        self.assertNotEqual(other['id'], self.sid)
        self.assertEqual(len(self.bot.store.get(self.sid)['history']), 2)
        self.bot.message(other['id'], 'Оборудование')
        self.assertEqual(self.calls[-1][1], [])

    def test_revisit_previous_topic_and_restart(self):
        for question in ['Расходы на заводе', 'А оборудование?', 'Вернёмся к расходам']:
            self.send(question)
        self.assertIn('Расходы на заводе', self.calls[-1][1][0]['content'])
        restored = Store(self.path).get(self.sid)
        self.assertEqual(len(restored['history']), 6)

    def test_unknown_session_and_limits(self):
        with self.assertRaises(KeyError): self.bot.message('unknown', 'Привет')
        for value in ['', ' ' * 2, 'x' * 4001, None]:
            with self.assertRaises(ValueError): self.send(value)

    def test_reenter_contact_fields(self):
        self.collect()
        self.send('/lead')
        state = self.send('Петров Пётр')
        self.assertEqual(state['fields'], {'full_name': 'Петров Пётр'})

    def test_confirmation_by_button_label(self):
        self.collect()
        self.send('Подтвердить заявку')
        self.assertEqual(self.count_leads(), 1)

    def test_old_topics_are_in_llm_context(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def read(self): return b'{"choices":[{"message":{"content":"ok"}}]}'
        history = [dict(role='user' if i % 2 == 0 else 'assistant',
                        content='Ранняя тема' if i == 0 else str(i), kind='consultation')
                   for i in range(40)]
        env = {'LLM_BASE_URL':'https://example.invalid/v1','LLM_API_KEY':'test','LLM_MODEL':'test'}
        with patch.dict('os.environ', env), patch('urllib.request.urlopen', return_value=Response()) as call:
            rag_generator.generate_answer('Вернемся к ранней теме', [], history)
        payload = json.loads(call.call_args.args[0].data)
        self.assertIn('Ранняя тема', payload['messages'][0]['content'])

    def test_phone_validation(self):
        self.assertEqual(phone_number('8 (999) 123-45-67'), '+79991234567')
        self.assertIsNone(phone_number('номер 79991234567'))
        self.assertIsNone(phone_number('0000000000'))

    def test_summary_without_consultation(self):
        state = self.collect()
        self.assertIn('Потребности пока не уточнены', state['history'][-1]['content'])

    def test_retrieval_followup_uses_earlier_topics(self):
        retriever = Retriever()
        history = [dict(role='user', kind='consultation', content='Мониторинг оборудования')]
        self.assertTrue(retriever.search('Расскажите подробнее', history))
        self.assertEqual(retriever.search('zxqv123', []), [])

    def test_llm_receives_only_consultation_history(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def read(self): return json.dumps({'choices':[{'message':{'content':'Ответ'}}]}).encode()
        env = {'LLM_BASE_URL':'https://example.invalid/v1','LLM_API_KEY':'test','LLM_MODEL':'test'}
        history = [dict(role='user',content='Первый вопрос',kind='consultation'),
                   dict(role='user',content='+79991234567',kind='lead')]
        with patch.dict('os.environ', env), patch('urllib.request.urlopen', return_value=Response()) as call:
            self.assertEqual(rag_generator.generate_answer('Второй вопрос', [], history), 'Ответ')
        payload = json.loads(call.call_args.args[0].data)
        self.assertEqual(payload['messages'][1]['content'], 'Первый вопрос')
        self.assertNotIn('+79991234567', json.dumps(payload))

    def test_http_session_roundtrip_and_origin_block(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(self.bot))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = 'http://127.0.0.1:' + str(server.server_port)
        try:
            with urllib.request.urlopen(url + '/') as r:
                self.assertIn('Новый диалог', r.read().decode())
            req = urllib.request.Request(url+'/api/sessions', data=b'{}', headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(req) as r: state = json.load(r)
            headers = {'Authorization':'Bearer '+state['id'], 'Content-Type':'application/json'}
            req = urllib.request.Request(url+'/api/chat', data=json.dumps({'message':'/lead'}).encode(), headers=headers)
            with urllib.request.urlopen(req) as r: self.assertEqual(json.load(r)['stage'], 'full_name')
            with urllib.request.urlopen(urllib.request.Request(url+'/api/session', headers=headers)) as r:
                self.assertEqual(json.load(r)['stage'], 'full_name')
            req = urllib.request.Request(url+'/api/chat', data=b'{}', headers={**headers,'Origin':'https://other.invalid'})
            with self.assertRaises(urllib.error.HTTPError) as error: urllib.request.urlopen(req)
            self.assertEqual(error.exception.code, 403)
        finally:
            server.shutdown(); server.server_close(); thread.join()


if __name__ == '__main__':
    unittest.main()
