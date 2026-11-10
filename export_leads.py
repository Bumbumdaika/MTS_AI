"""Local operator export; deliberately no unauthenticated HTTP list of contacts."""
import argparse
import csv
import os
from config import ROOT, load_env
from storage import Store


def main():
    load_env()
    parser = argparse.ArgumentParser(description='Выгрузить заявки в CSV для менеджера')
    parser.add_argument('--output', required=True, help='Путь к CSV с персональными данными')
    args = parser.parse_args()
    store = Store(os.getenv('BOT_DB') or ROOT / 'runtime/bot.sqlite3')
    with store.connect() as db:
        rows = db.execute('SELECT id,created_at,full_name,phone,company,summary FROM leads ORDER BY created_at').fetchall()
    with open(args.output, 'w', encoding='utf-8-sig', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'created_at', 'full_name', 'phone', 'company', 'summary'])
        for row in rows:
            # Prevent user-provided values from being executed as spreadsheet formulas.
            writer.writerow(["'" + str(v) if str(v).startswith(('=', '+', '-', '@', '\t', '\r', '\n')) else v for v in row])
    print(f'Выгружено заявок: {len(rows)}. Файл: {args.output}')


if __name__ == '__main__':
    main()
