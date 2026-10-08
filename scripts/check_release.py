"""Static source, content-integrity, size, and accidental-secret checks."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest=json.loads((ROOT/'provenance/source_export.json').read_text())
    errors=[]
    for row in manifest['files']:
        p=ROOT/row['path']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['release_sha256']:
            errors.append('HASH_MISMATCH:'+row['path'])
    patterns=[r'/home/[A-Za-z0-9_.-]+/', r'/data/users/[A-Za-z0-9_.-]+/',
              r'gh[pousr]_[A-Za-z0-9]{25,}',r'github_pat_[A-Za-z0-9_]{30,}',
              r'AKIA[A-Z0-9]{16}',r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----']
    count=0
    paths=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    for name in sorted(set(paths)-{''}):
        p=ROOT/name
        if not p.is_file() or '.git' in p.parts or '__pycache__' in p.parts:continue
        if p.suffix=='.py':ast.parse(p.read_text());count+=1
        if p.stat().st_size>10*1024*1024:errors.append('OVERSIZE:'+str(p.relative_to(ROOT)))
        if p.suffix in {'.py','.md','.json','.csv','.toml','.txt','.tex'}:
            text=p.read_text()
            if any(re.search(pattern,text) for pattern in patterns):
                errors.append('SENSITIVE_PATTERN:'+str(p.relative_to(ROOT)))
    print(json.dumps({'python_files_parsed':count,'exported_files_verified':len(manifest['files']),
                      'errors':errors,'scope':'Static checks only; not proof of anonymity or dynamic correctness'},indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
