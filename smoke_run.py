"""Dry-run smoke script: validates CSV parsing and filename creation without running PsychoPy or audio.
"""
import os
from MR_AUT_utils import read_items, sanitize_filename

ROOT = os.getcwd()
CSV_PATH = os.path.join(ROOT, 'stim', 'MR_AUT_items.csv')


def main():
    if not os.path.exists(CSV_PATH):
        print('Missing CSV:', CSV_PATH)
        return 2
    items = read_items(CSV_PATH)
    print(f'Read {len(items)} items; sample: {items[:3]}')
    for i, it in enumerate(items[:5]):
        print(i, sanitize_filename(it))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
