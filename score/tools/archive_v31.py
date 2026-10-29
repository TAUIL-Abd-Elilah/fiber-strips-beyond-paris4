"""Archive pre-registration v3.1 into superseded_v31/ before the v3.2 erratum (PREREG 10.8): every file v3.2 changes,
as it was frozen, each checked against the v3.1 FREEZE sha256; the relaunch wrapper as it stood at v3.1 (not pinned
then); and the wrapper's first version, rebuilt from the session record (the Write of 28 Oct 14:44:45 UTC).

    python -I tools/archive_v31.py --first-version <file with the first version's text>
"""
import argparse
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'superseded_v31'
FILES = [('PREREG.md', 'PREREG_v31.md'), ('tools/write_freeze.py', 'write_freeze_v31.py'),
         ('FREEZE.txt', 'FREEZE_v31.txt'), ('runs/relaunch_v3.ps1', 'relaunch_v3_at_v31.ps1')]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--first-version', required=True)
    a = ap.parse_args()
    if DEST.exists():
        raise SystemExit(f'{DEST} exists')
    frozen = {}
    for line in (ROOT/'FREEZE.txt').read_text().splitlines():
        line = line.strip()
        if len(line) > 66 and line[64:66] == ' *':
            frozen[line[66:]] = line[:64]
    if not (ROOT/'FREEZE.txt').read_text().startswith('Freeze of pre-registration v3.1 '):
        raise SystemExit('FREEZE.txt is not the v3.1 freeze')
    DEST.mkdir()
    rows = []
    for src, name in FILES:
        shutil.copy2(ROOT/src, DEST/name)
        h = sha(DEST/name)
        if src == 'FREEZE.txt':
            check = f'(the v3.1 freeze record itself; {len(frozen)} files)'
        elif src in frozen:
            check = 'yes' if frozen[src] == h else 'NO'
        else:
            check = '(not pinned in v3.1)'
        rows.append((name, src, h, check))
    text = Path(a.first_version).read_bytes()
    (DEST/'relaunch_v3_first_version_from_session_record.ps1').write_bytes(text)
    rows.append(('relaunch_v3_first_version_from_session_record.ps1', 'runs/relaunch_v3.ps1 (28 Oct 14:44:45 UTC)',
                 sha(DEST/'relaunch_v3_first_version_from_session_record.ps1'), '(rebuilt from the session record)'))
    if any(r[3] == 'NO' for r in rows):
        raise SystemExit(f'an archived file differs from the v3.1 FREEZE: {rows}')
    w = max(len(r[0]) for r in rows)+2
    v = max(len(r[1]) for r in rows)+2
    lines = ['superseded_v31: pre-registration v3.1 (frozen 2026-10-29 00:34:06), archived on 29 Oct 2026 before the v3.2',
             'erratum (PREREG 10.8). Every file that v3.2 changes is here as it was frozen; each is checked against the',
             'v3.1 FREEZE sha256 below. Files v3.2 does not change keep their v3.1 sha256 in the v3.2 FREEZE.txt.',
             'runs/relaunch_v3.ps1 was not pinned in v3.1: it is kept here as it stood at v3.1, and its first version',
             '(28 Oct 14:44:45 UTC) is rebuilt from the session record of the author\'s AI assistant, which wrote it.',
             '',
             f"{'archived file':<{w}}{'path':<{v}}{'sha256 (archived)':<66}= v3.1 FREEZE"]
    lines += [f'{n:<{w}}{s:<{v}}{h}  {c}' for n, s, h, c in rows]
    lines += ['', 'changed in v3.2: PREREG.md, tools/write_freeze.py, FREEZE.txt (rewritten by tools/write_freeze.py);',
              'runs/relaunch_v3.ps1 (comments, and an unused function removed) is pinned from v3.2 on']
    (DEST/'ARCHIVE.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
