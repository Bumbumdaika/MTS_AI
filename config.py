import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_env():
    path = ROOT / '.env'
    if path.exists():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            key, sep, value = line.partition('=')
            if sep:
                os.environ.setdefault(key.strip(), value.strip().strip('\"\''))
