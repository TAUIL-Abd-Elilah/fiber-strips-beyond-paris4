"""Archive pre-registration v2 into superseded_v2/ before v3 overwrites it (run once, 28 Oct 2026).

Every file in FREEZE.txt (v2) is copied to superseded_v2/<same relative path> (``../tools/X`` -> ``setup_tools/X``;
PREREG.md, FREEZE.txt and score_strips.py as PREREG_v2.md, FREEZE_v2.txt, score_strips_v2.py; superseded_v1/* stays
where it is) and its sha256 is checked against the v2 freeze. score_strips.py had already been edited for v3 when
this ran; its v2 text was rebuilt by undoing those edits exactly (runs/v2_scorer_rebuilt.py, CRLF line ends as
frozen) and is accepted only when its sha256 is the frozen one.
"""
import hashlib
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'superseded_v2'
RENAME = {'PREREG.md': 'PREREG_v2.md', 'score_strips.py': 'score_strips_v2.py'}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    freeze = (ROOT/'FREEZE.txt').read_bytes()
    entries = []
    for line in freeze.decode().splitlines():
        line = line.strip()
        if len(line) > 66 and line[64:66] == ' *':
            entries.append((line[:64], line[66:]))
    assert len(entries) == 33, len(entries)
    DEST.mkdir(exist_ok=True)
    rows = []
    for digest, rel in entries:
        if rel.startswith('superseded_v1/'):
            rows.append(f'{digest}  {rel}  (kept in place)')
            continue
        if rel == 'score_strips.py':
            data = (ROOT/'runs'/'v2_scorer_rebuilt.py').read_bytes().replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
        else:
            data = (ROOT/rel).resolve().read_bytes()
        if sha(data) != digest:
            raise SystemExit(f'{rel}: sha256 {sha(data)} != frozen {digest}')
        out = RENAME.get(rel, rel.replace('../tools/', 'setup_tools/'))
        (DEST/out).parent.mkdir(parents=True, exist_ok=True)
        (DEST/out).write_bytes(data)
        rows.append(f'{digest}  {rel}  -> superseded_v2/{out}')
    (DEST/'FREEZE_v2.txt').write_bytes(freeze)
    rows.append(f'{sha(freeze)}  FREEZE.txt  -> superseded_v2/FREEZE_v2.txt')
    (DEST/'ARCHIVE.txt').write_text('Pre-registration v2 (frozen 2026-10-28 12:27:47), archived before v3. Every file '
                                    'below matches its v2 FREEZE sha256.\n'+'\n'.join(rows)+'\n')
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
