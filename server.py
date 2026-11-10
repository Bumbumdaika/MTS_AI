"""Local single-process HTTP adapter; no third-party dependencies."""
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from config import ROOT
from solution import build_service


def make_handler(service):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass  # Do not write session secrets or contact data to access logs.

        def respond(self, status, data, content_type='application/json; charset=utf-8'):
            body = json.dumps(data, ensure_ascii=False).encode() if isinstance(data, dict) else data
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/':
                return self.respond(200, (ROOT / 'web/index.html').read_bytes(), 'text/html; charset=utf-8')
            if path == '/api/health':
                return self.respond(200, dict(status='ok', mode=os.getenv('BOT_MODE', 'demo')))
            if path == '/api/session':
                try:
                    session_id = self.session_id()
                    return self.respond(200, service.view(service.store.get(session_id)))
                except (ValueError, KeyError):
                    return self.respond(404, dict(error='Диалог не найден.'))
            self.respond(404, dict(error='Адрес не найден.'))

        def session_id(self):
            auth = self.headers.get('Authorization', '')
            if not re.fullmatch(r'Bearer [a-f0-9]{32}', auth):
                raise ValueError('Требуется Authorization: Bearer <session_id>.')
            return auth[7:]

        def do_POST(self):
            # Block cross-origin browser submissions even when running on localhost.
            origin = self.headers.get('Origin')
            if origin and origin != 'http://' + self.headers.get('Host', ''):
                return self.respond(403, dict(error='Недопустимый источник запроса.'))
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 <= size <= 65536:
                    raise ValueError('Слишком большой запрос.')
                payload = json.loads(self.rfile.read(size) or b'{}')
                if not isinstance(payload, dict):
                    raise ValueError('Ожидается JSON-объект.')
                path = urlsplit(self.path).path
                if path == '/api/sessions':
                    return self.respond(201, service.new())
                if path == '/api/chat':
                    state = service.message(self.session_id(), payload.get('message'))
                    return self.respond(200, state)
                self.respond(404, dict(error='Адрес не найден.'))
            except (ValueError, UnicodeError) as error:
                self.respond(400, dict(error=str(error)))
            except KeyError:
                self.respond(404, dict(error='Диалог не найден.'))
            except RuntimeError:
                self.respond(502, dict(error='Не удалось получить ответ LLM. Проверьте подключение и повторите вопрос.'))
            except Exception:
                self.respond(500, dict(error='Внутренняя ошибка. Проверьте настройки сервера.'))
    return Handler


if __name__ == '__main__':
    service = build_service()
    address = ('127.0.0.1', int(os.getenv('PORT', '8000')))
    server = ThreadingHTTPServer(address, make_handler(service))
    print(f'Energytool: http://{address[0]}:{address[1]} — режим {os.getenv("BOT_MODE", "demo")}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
