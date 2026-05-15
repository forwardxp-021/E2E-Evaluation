#!/usr/bin/env python3
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'tools'

HARD_PATTERNS = [
    '/' + 'tmp/old.py',
    "Path('/" + 'tmp',
    'Path("/' + 'tmp',
]

files = sorted(TOOLS.glob('*.py'))
violations = []
for fp in files:
    if fp.name == Path(__file__).name:
        continue
    text = fp.read_text(encoding='utf-8', errors='ignore')
    for pat in HARD_PATTERNS:
        if pat in text:
            violations.append((fp, f'contains forbidden pattern: {pat}'))
    if 'exec(' in text and ('/' + 'tmp') in text:
        violations.append((fp, 'contains exec(...) with /tmp reference'))
    if re.search(r"Path\((['\"])\/tmp.*?\1\)\.read_text\(", text, flags=re.S):
        violations.append((fp, 'contains read_text() on hard-coded /tmp path'))

if violations:
    print('FAIL: forbidden tmp dependency patterns found:')
    for fp, msg in violations:
        print(f' - {fp.relative_to(ROOT)}: {msg}')
    sys.exit(1)

print('OK: no forbidden /tmp dependency patterns found in tools/*.py')
