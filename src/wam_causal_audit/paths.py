"""Explicit relocation of historical paths; no access to the original machine."""
import os
from pathlib import Path


def resolve(value: str) -> str:
    root = Path(__file__).resolve().parents[2]
    workspace = os.environ.get('WAM_WORKSPACE', str(root / 'reference/workspace'))
    data = os.environ.get('WAM_DATA', str(root / 'reference/data'))
    return value.replace('@WORKSPACE@', workspace).replace('@DATA@', data)
