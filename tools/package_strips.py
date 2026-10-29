"""Package the scored run's strips for a public release (CC BY-NC 4.0, LICENSE-DATA.md), every local path removed.

    python tools/package_strips.py --strips C:/VesuviusTemp/strips/score/runs/v3/strips \
        --plans C:/VesuviusTemp/strips/score/results --out release/ [--dry-run] [--sets primary,secondary,0332]
        [--overwrite] [--require-complete] [--freeze FREEZE.txt] [--no-done-check]

Input: one villa ``sheet_strip/infer.py`` output folder per piece, STRIPS/<set>/<piece id>/, as
score/runs/v3/run_all.ps1 writes them (sets primary, secondary, 0332): ``strip/`` (the tifxyz: x.tif, y.tif, z.tif,
mask.tif, meta.json, trace.json, and expected_error.tif where there is one), ``strip.npz`` and ``summary.json``.
Nothing else of a piece folder is packaged (no render, no patch sheet, no temporary folder); what is left out is
listed. A piece folder without all of those files is incomplete: listed, not packaged.

Pieces are mapped to fibers, annotators, families and the stretches or crossing windows they cover from the frozen
plan files only (PLANS/runplan_*.tsv and the plan JSONs, checked against FREEZE.txt beside PLANS when it is there),
never from the strips. A strip's own summary.json and trace.json are only checked against its plan row (AFV id,
decisions) and the run's checkpoint and settings; a mismatch stops the release.

Output (OUT): one zip per set, stored members, <set>/<piece>/strip/..., <set>/<piece>/strip.npz,
<set>/<piece>/summary.json (extract the zips into one folder and it is the run's strips folder again, as
score/score_strips.py reads it); README.md; manifest.json (per strip: set, piece, fiber, annotator, family, the
stretches or crossing windows it covers, the size and sha256 of every member; the scrub report); SHA256SUMS.txt (the
zips, README.md and manifest.json).

Scrub: in every text member (JSON, txt) and in every string array of strip.npz, each absolute local path (drive
letters in any form, MSYS/WSL/Cygwin drives, UNC paths, POSIX home and temp roots, file URIs) is replaced by the token
'<local>', and so is any path fragment naming the author's workspace folders. Nothing else changes: a JSON member
keeps every byte outside the strings it changes, an npz keeps every other array byte for byte, and a member with
nothing to replace is copied byte for byte. TIFFs are never rewritten. Then every member written is read back and
scanned (JSON strings and raw text, TIFF tags, npz headers and string arrays, and every byte of every member for the
workspace folder names in ASCII, UTF-16 and UTF-32); the release fails if anything is left, and also if a path sits
where it can't be replaced (a TIFF tag, a numeric or pickled array, a JSON key whose removal would merge two keys).
The fields that held paths are reported (stdout, manifest.json, README.md).

The release is built in OUT/.staging and moved into OUT only when every check passed. --dry-run reads, maps, scrubs
and scans everything in memory and writes nothing. The run must be finished (DONE or V3_FINAL beside STRIPS, when
run_all.ps1 is there). Exit codes: 0 done; 1 a local path left or not removable, or a written file failed its
check; 2 bad input (arguments, layout, plans, a strip that does not match its plan); 3 planned pieces without a
strip under --require-complete.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import time
import zipfile

import numpy as np
import tifffile

TOKEN = '<local>'
REPO = 'https://github.com/TAUIL-Abd-Elilah/fiber-strips-beyond-paris4'
CHECKPOINT = 'scrollprize/fiber_strip_small_v1@7bd163a22c839ff312879ae1916195d0c9729c0e'
CHECKPOINT_SOURCE = f'hf://{CHECKPOINT}/'  # infer.py records the checkpoint as hf://REPO@COMMIT/FILE
VILLA = '6f2024569404a46b0f008eee292773945c07a684'
SETTINGS = dict(decision_step=8.0, blend=16.0, trim_patch_voxels=75.0)  # run_all.ps1: --step 8 --blend 16 --trim 75
INFER_ARGS = '--device cpu --index none --no-render --step 8 --blend 16 --trim 75 --batch 32'
VOLUMES = {
    'PHerc0139': 's3://vesuvius-challenge-open-data/PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr',
    'PHerc0332': 's3://vesuvius-challenge-open-data/PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr',
}
# As run_all.ps1's $Sets: the run plan, the plan JSON and the piece AFV of each set (paths relative to score/).
SETS = {
    'primary': dict(scroll='PHerc0139', kind='stretches', runplan='runplan_0139_primary.tsv',
                    plan='matched_stretches.json', afv='data/matched_0139.afv', zip='PHerc0139_primary_strips.zip',
                    title='PHerc0139 primary set: the case-study stretches of 2.5 mm or more'),
    'secondary': dict(scroll='PHerc0139', kind='stretches', runplan='runplan_0139_secondary_run40.tsv',
                      plan='matched_stretches_secondary_run40.json', afv='data/matched_0139_secondary_run40.afv',
                      zip='PHerc0139_secondary_strips.zip',
                      title='PHerc0139 secondary set: the run subsample of the stretches of 40 sv or more'),
    '0332': dict(scroll='PHerc0332', kind='crossings', runplan='runplan_0332_crossings.tsv',
                 plan='crossings_0332_plan.json', afv='data/crossings_0332.afv', zip='PHerc0332_crossing_strips.zip',
                 title='PHerc0332: both windows of the planned H/V crossings (no ground truth)'),
}
STRIP_FILES = ('mask.tif', 'meta.json', 'trace.json', 'x.tif', 'y.tif', 'z.tif')  # strip/, required
OPTIONAL_STRIP_FILES = ('expected_error.tif',)                                   # strip/, when written
PIECE_FILES = ('strip.npz', 'summary.json')                                      # beside strip/, required
PIECE_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*')
RELEASE_FILES = ('README.md', 'manifest.json', 'SHA256SUMS.txt')
STAGING = '.staging'
CROSS_TOL = 3.0  # sv: the tight crossings (PREREG 7), unless the plan records another CROSS_TOL
GITHUB_ASSET_LIMIT = 2*10**9


class PackageError(Exception):
    def __init__(self, message, code=2):
        super().__init__(message)
        self.code = code


class ScrubError(PackageError):
    def __init__(self, message):
        super().__init__(message, 1)


# ---- Local paths --------------------------------------------------------------------------------------------------

_DELIM = r'\s"\'`=:,;(){}\[\]<>|'
_AT_ROOT = r'(?<![^' + _DELIM + r'])'  # at the start of the text or after a delimiter
_ROOT = (r'(?:file:/{0,3})?(?:\\\\[?.]\\)?(?<![A-Za-z0-9])[A-Za-z]:[\\/]'  # C:/ C:\ c:/ \\?\C:\ file:///C:/
         r'|(?<![A-Za-z0-9])[A-Za-z]%3a(?:%2f|%5c)'                      # C%3A%2F (URL-encoded)
         r'|(?<![A-Za-z0-9])file:/{1,3}'                                 # any other file URI
         r'|' + _AT_ROOT + r'\\\\[A-Za-z0-9._$-]+\\'                      # \\server\share
         r'|' + _AT_ROOT + r'/(?:mnt/|cygdrive/)?[A-Za-z]/'               # /c/ /mnt/c/ /cygdrive/c/
         r'|' + _AT_ROOT + r'/(?:home|users|root|tmp)/')                  # POSIX home and temp roots
# A space belongs to a path unless command-line syntax follows it (an option, &&, |, ;, >, more blanks).
_SPACE = r'[ \t](?![-\s&|;>])'
_MID = r'(?:[^\\/\x00-\x1f"<>|?*:\s]|' + _SPACE + r')*[\\/]+'  # a folder (spaces allowed) and its separator(s)
_LAST = r'[^\\/\x00-\x1f"<>|?*:\s,;\'`(){}\[\]]*'               # the last name: no space or punctuation after it
# The author's workspace folder names: replaced wherever they appear (also in relative paths), and the only
# patterns looked for in binary data (long enough not to occur there by chance).
_WORKSPACE = (r'users(?:[\\/]|%2f|%5c)+pc|vesuviustemp|vesuviusctcache|miniconda|prizetemp|hf_home'
              r'|vesuvius(?:\s|%20|_|-|\+)+progress(?:\s|%20|_|-|\+)+prizes')
EMBEDDED = re.compile(r'(?:' + _ROOT + r')[\\/]*(?:' + _MID + r')*' + _LAST, re.I)
WHOLE = re.compile(r'(?:' + _ROOT + r')(?:[^\x00-\x1f"<>|?*:\s]|' + _SPACE + r')*', re.I)  # fullmatch: one path
WORKSPACE = re.compile(_WORKSPACE, re.I)
LOCAL = re.compile(r'(?:' + _ROOT + r')|' + _WORKSPACE, re.I)       # anything local, in text
_FRAGMENT_STOP = frozenset(' \t\r\n\f\v"\'`<>|?*,;(){}[]=')
_RELATIVE_PATH = re.compile(r'(?:\.{1,2}[\\/])?[\w.-]+(?:[\\/][\w.-]+)+')


def local_spans(s):
    """The spans (start, end) of ``s`` to replace by TOKEN: absolute local paths (a string that is one path as a
    whole, spaces included), and the path fragment around each workspace folder name; merged when they overlap."""
    if WHOLE.fullmatch(s):
        return [(0, len(s))]
    spans = [m.span() for m in EMBEDDED.finditer(s) if m.end() > m.start()]
    for m in WORKSPACE.finditer(s):
        a, b = m.span()
        while a > 0 and s[a-1] not in _FRAGMENT_STOP:
            a -= 1
        while b < len(s) and s[b] not in _FRAGMENT_STOP:
            b += 1
        spans.append((a, b))
    merged = []
    for a, b in sorted(spans):
        if merged and a < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


def scrub_string(s):
    """``s`` with every local path replaced by TOKEN, and the spans replaced (in ``s``)."""
    spans = local_spans(s)
    if not spans:
        return s, spans
    parts, at = [], 0
    for a, b in spans:
        parts += [s[at:a], TOKEN]
        at = b
    new = ''.join(parts+[s[at:]])
    left = LOCAL.search(new)
    if left:
        raise ScrubError(f'a local path is left after scrubbing ({left.group(0)!r})')
    return new, spans


# ---- JSON: only the strings that hold a path change, every other byte stays -------------------------------------

_JSON_STRING = re.compile(r'"((?:[^"\\]|\\.)*)"', re.S)
_ESCAPES = {'"': '"', '\\': '\\', '/': '/', 'b': '\b', 'f': '\f', 'n': '\n', 'r': '\r', 't': '\t'}


class _Pairs(list):
    """A parsed JSON object: its (key, value) pairs in order, duplicates kept."""


def _decode_json_string(body):
    """A JSON string literal's body (between the quotes) decoded, and the body offset of each character (plus the
    end), so that a span of the decoded text maps back to the raw literal."""
    chars, starts, i = [], [], 0
    while i < len(body):
        starts.append(i)
        if body[i] != '\\':
            chars.append(body[i])
            i += 1
        elif body[i+1] == 'u':
            code = int(body[i+2:i+6], 16)
            i += 6
            if 0xD800 <= code < 0xDC00 and body[i:i+2] == '\\u':
                low = int(body[i+2:i+6], 16)
                if 0xDC00 <= low < 0xE000:
                    code = 0x10000+((code-0xD800) << 10)+(low-0xDC00)
                    i += 6
            chars.append(chr(code))
        else:
            chars.append(_ESCAPES[body[i+1]])
            i += 2
    starts.append(len(body))
    return ''.join(chars), starts


def _pointer(parent, part):
    if not parent:
        return part
    return parent+part if part.startswith('[') else f'{parent}.{part}'


def _shown(key):
    """A key as it may appear in a report: scrubbed."""
    try:
        return scrub_string(key)[0]
    except ScrubError:
        return '<key with a local path>'


def _json_strings(node, pointer=''):
    """(pointer, string, is_key) for every key and string value of a parsed JSON value (keys in pointers scrubbed)."""
    if isinstance(node, _Pairs):
        for key, value in node:
            shown = _shown(key)
            yield _pointer(pointer, '{'+shown+'}'), key, True
            yield from _json_strings(value, _pointer(pointer, shown))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _json_strings(value, _pointer(pointer, f'[{i}]'))
    elif isinstance(node, str):
        yield pointer or '(top)', node, False


def _check_keys(node, pointer=''):
    """Refuse a scrub that would give an object two equal keys (a JSON reader would keep only one of them)."""
    if isinstance(node, _Pairs):
        keys = [key for key, _ in node]
        new = [scrub_string(key)[0] for key in keys]
        if len(set(new)) < len(new) and len(set(keys)) == len(keys):
            raise ScrubError(f'removing local paths would give the object at {pointer or "the top"} equal keys')
        for key, value in node:
            _check_keys(value, _pointer(pointer, scrub_string(key)[0]))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _check_keys(value, _pointer(pointer, f'[{i}]'))


def scrub_json(data):
    """A JSON member with every local path in its strings (keys and values) replaced by TOKEN, each changed string
    edited in place in the raw text, so that every other byte (layout, numbers, escapes) stays. Returns the bytes
    (``data`` itself when nothing changes), the pointers of the strings that held a path, and the pointers of
    strings left as they are that look like relative paths."""
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError as error:
        raise ScrubError(f'not UTF-8 ({error})') from None
    bom = '\ufeff' if text.startswith('\ufeff') else ''
    body = text[len(bom):]
    try:
        tree = json.loads(body, object_pairs_hook=_Pairs)
    except ValueError as error:
        raise ScrubError(f'not JSON ({error})') from None
    changed, kept = [], []
    for pointer, s, is_key in _json_strings(tree):
        if local_spans(s):
            changed.append(pointer)
        elif not is_key and '://' not in s and _RELATIVE_PATH.fullmatch(s):
            kept.append(pointer)
    if not changed:
        return data, [], kept
    _check_keys(tree)
    parts, at, edited = [], 0, 0
    for m in _JSON_STRING.finditer(body):
        raw = m.group(1)
        try:
            value, starts = _decode_json_string(raw)
            same = json.loads(m.group(0)) == value
        except (ValueError, KeyError, IndexError):
            same = False
        if not same:
            raise ScrubError(f'could not read the JSON string at offset {m.start()} exactly')
        spans = local_spans(value)
        if not spans:
            continue
        pieces, pos = [], 0
        for a, b in spans:
            pieces += [raw[pos:starts[a]], TOKEN]
            pos = starts[b]
        new_raw = ''.join(pieces+[raw[pos:]])
        if json.loads('"'+new_raw+'"') != scrub_string(value)[0]:
            raise ScrubError(f'the edited JSON string at offset {m.start()} does not read back as scrubbed')
        parts += [body[at:m.start(1)], new_raw]
        at = m.end(1)
        edited += 1
    new_body = ''.join(parts+[body[at:]])
    if edited != len(changed):
        raise ScrubError(f'{edited} strings edited, {len(changed)} expected')
    json.loads(new_body)
    return (bom+new_body).encode('utf-8'), changed, kept


def scrub_text(data):
    """A plain-text member with every local path replaced, line by line (line ends kept); the lines changed."""
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError as error:
        raise ScrubError(f'not UTF-8 ({error})') from None
    out, changed = [], []
    for n, line in enumerate(text.splitlines(keepends=True), 1):
        content = line.rstrip('\r\n')
        new, spans = scrub_string(content)
        if spans:
            changed.append(f'line {n}')
        out.append(new+line[len(content):])
    return (''.join(out).encode('utf-8') if changed else data), changed, []


# ---- npz: string arrays scrubbed, every other array kept byte for byte -------------------------------------------

def _npy_header(raw):
    """(version, header text, data offset) of a .npy member."""
    if raw[:6] != b'\x93NUMPY':
        raise ScrubError('an npz member is not a .npy array')
    if raw[6] == 1:
        (n,), start = struct.unpack('<H', raw[8:10]), 10
    else:
        (n,), start = struct.unpack('<I', raw[8:12]), 12
    return (raw[6], raw[7]), raw[start:start+n].decode('latin-1'), start+n


def _read_npy(raw):
    """The array of a .npy member, or None when numpy would have to unpickle it (never done here)."""
    try:
        return np.lib.format.read_array(io.BytesIO(raw), allow_pickle=False)
    except ValueError as error:
        if 'pickle' in str(error):
            return None
        raise


def _scrub_strings_array(arr, name):
    """A string array (U or S) with its local paths replaced, or None when none has one; the elements changed."""
    kind = arr.dtype.kind
    values = [v.decode('latin-1') if kind == 'S' else v for v in arr.reshape(-1).tolist()]
    new_values, changed = [], []
    for i, v in enumerate(values):
        new = scrub_string(v)[0]
        if new != v:
            changed.append(f'{name}[{i}]' if arr.ndim else name)
        new_values.append(new)
    if not changed:
        return None, []
    if kind == 'S':
        new_values = [v.encode('latin-1') for v in new_values]
    width = arr.dtype.itemsize//(4 if kind == 'U' else 1)
    need = max([len(v) for v in new_values]+[1])
    dtype = arr.dtype if need <= width else np.dtype(arr.dtype.str[:2]+str(need))
    new = np.array(new_values, dtype=dtype).reshape(arr.shape)
    return new, changed


def scrub_npz(data):
    """An npz with its string arrays scrubbed. Numeric arrays never change; object and structured arrays are only
    scanned (a path there stops the release). With nothing to replace the npz is returned as it is; otherwise only
    the changed .npy members are rewritten (same dtype unless a value needs a wider one, same order and .npy
    version) and every other member's array bytes are checked to be the same."""
    try:
        src = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, OSError, ValueError) as error:
        raise ScrubError(f'unreadable npz ({error})') from None
    replaced, changed = {}, []
    for info in src.infolist():
        if not info.filename.endswith('.npy'):
            continue
        try:
            raw = src.read(info)
            version, header, _ = _npy_header(raw)
            arr = _read_npy(raw)
        except ScrubError:
            raise
        except Exception as error:  # noqa: BLE001 - an array that can't be read can't be shown free of paths
            raise ScrubError(f'unreadable array {info.filename} ({type(error).__name__}: {error})') from None
        if arr is None or arr.dtype.fields is not None or arr.dtype.kind not in 'US':
            continue
        new, hit = _scrub_strings_array(arr, info.filename[:-4])
        if new is None:
            continue
        if "'fortran_order': True" in header:
            new = np.asfortranarray(new)
        buf = io.BytesIO()
        try:
            np.lib.format.write_array(buf, new, version=version, allow_pickle=False)
        except ValueError:
            buf = io.BytesIO()
            np.lib.format.write_array(buf, new, allow_pickle=False)
        replaced[info.filename] = buf.getvalue()
        changed += hit
    if not replaced:
        return data, []
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', allowZip64=True) as dst:
        for info in src.infolist():
            new_info = zipfile.ZipInfo(info.filename, info.date_time)
            new_info.compress_type, new_info.external_attr = info.compress_type, info.external_attr
            new_info.create_system, new_info.comment = info.create_system, info.comment
            dst.writestr(new_info, replaced.get(info.filename) or src.read(info))
        dst.comment = src.comment
    new_data = out.getvalue()
    with zipfile.ZipFile(io.BytesIO(new_data)) as check:
        if check.namelist() != src.namelist():
            raise ScrubError('the rewritten npz does not hold the same arrays')
        for info in src.infolist():
            if info.filename not in replaced and check.read(info.filename) != src.read(info):
                raise ScrubError(f'the rewritten npz changed {info.filename}')
    return new_data, changed


def scrub_member(name, data):
    """(bytes, fields that held a path, relative-path-like fields kept) of one member. TIFFs are never changed."""
    suffix = name.rsplit('.', 1)[-1].lower()
    if suffix == 'json':
        return scrub_json(data)
    if suffix == 'txt':
        return scrub_text(data)
    if suffix == 'npz':
        new, changed = scrub_npz(data)
        return new, changed, []
    return data, [], []


# ---- Scan of the written bytes --------------------------------------------------------------------------------------

def _hits(text, regex, where):
    return [(where, m.group(0)[:80]) for m in regex.finditer(text)]


def scan_raw(data, where):
    """The workspace folder names anywhere in the bytes: as ASCII/UTF-8/Latin-1, and as UTF-16 or UTF-32 (any byte
    order or alignment) once the NUL bytes are dropped. The drive and root patterns are not looked for in binary
    data, where three bytes such as 'C:/' occur by chance; they are looked for in every text, tag and string."""
    hits = _hits(data.decode('latin-1'), WORKSPACE, where)
    if b'\0' in data:
        hits += _hits(data.replace(b'\0', b'').decode('latin-1'), WORKSPACE, where+' (UTF-16/32)')
    return hits


def scan_json(name, data):
    text = data.decode('utf-8', 'replace')
    hits = _hits(text, LOCAL, name)
    try:
        tree = json.loads(text.lstrip('\ufeff'), object_pairs_hook=_Pairs)
    except ValueError as error:
        return hits+[(name, f'not JSON ({error})')]
    for pointer, s, _ in _json_strings(tree):
        hits += _hits(s, LOCAL, f'{name}:{pointer}')
    return hits


def scan_tiff(name, data):
    """Every tag of every IFD (SubIFDs too): text-like tags (BYTE, ASCII, SBYTE, UNDEFINED) for any local path;
    numeric tags and pixel data are covered by scan_raw."""
    hits = []

    def pages_of(pages, label):
        for p, page in enumerate(pages):
            yield f'{label}{p}', page
            sub = getattr(page, 'pages', None)
            if sub:
                yield from pages_of(sub, f'{label}{p}.sub')
    try:
        with tifffile.TiffFile(io.BytesIO(data)) as tif:
            for label, page in pages_of(tif.pages, 'page '):
                for tag in page.tags.values():
                    if int(tag.dtype) not in (1, 2, 6, 7):
                        continue
                    where = f'{name} {label} tag {tag.code} ({tag.name})'
                    raw = data[tag.valueoffset:tag.valueoffset+tag.valuebytecount]
                    hits += _hits(raw.decode('latin-1'), LOCAL, where)
                    hits += _hits(raw.replace(b'\0', b'').decode('latin-1'), LOCAL, where)
                    if isinstance(tag.value, str):
                        hits += _hits(tag.value, LOCAL, where)
    except Exception as error:  # noqa: BLE001 - an unreadable TIFF can't be shown free of paths
        hits.append((name, f'unreadable TIFF ({type(error).__name__}: {error})'))
    return hits


def scan_npz(name, data):
    """Every array of the npz: member names and comments, the .npy headers, every element of string arrays, the raw
    bytes of object and structured arrays as text, and (scan_raw) every byte for the workspace folder names."""
    hits = []
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            hits += _hits(z.comment.decode('latin-1'), LOCAL, f'{name} (zip comment)')
            for info in z.infolist():
                where = f'{name}:{info.filename}'
                hits += _hits(info.filename, LOCAL, where+' (name)')
                hits += _hits(info.comment.decode('latin-1'), LOCAL, where+' (comment)')
                raw = z.read(info)
                hits += scan_raw(raw, where)
                if not info.filename.endswith('.npy'):
                    hits += _hits(raw.decode('latin-1'), LOCAL, where)
                    continue
                _, header, offset = _npy_header(raw)
                hits += _hits(header, LOCAL, where+' (header)')
                arr = _read_npy(raw)
                if arr is None or arr.dtype.fields is not None or arr.dtype.kind in 'OV':
                    body = raw[offset:]
                    hits += _hits(body.decode('latin-1'), LOCAL, where)
                    hits += _hits(body.replace(b'\0', b'').decode('latin-1'), LOCAL, where)
                elif arr.dtype.kind in 'US':
                    for i, v in enumerate(arr.reshape(-1).tolist()):
                        hits += _hits(v.decode('latin-1') if isinstance(v, bytes) else v, LOCAL, f'{where}[{i}]')
    except Exception as error:  # noqa: BLE001 - an unreadable npz can't be shown free of paths
        hits.append((name, f'unreadable npz ({type(error).__name__}: {error})'))
    return hits


def scan_member(name, data):
    """Local paths left in one member (its name included): a list of (where, what)."""
    hits = scan_raw(data, name)+_hits(name, LOCAL, f'{name} (name)')
    suffix = name.rsplit('.', 1)[-1].lower() if '.' in name.rsplit('/', 1)[-1] else ''
    if suffix == 'json':
        hits += scan_json(name, data)
    elif suffix in ('txt', 'md', 'tsv', 'csv'):
        hits += _hits(data.decode('utf-8', 'replace'), LOCAL, name)
    elif suffix in ('tif', 'tiff'):
        hits += scan_tiff(name, data)
    elif suffix == 'npz':
        hits += scan_npz(name, data)
    else:
        hits += _hits(data.decode('latin-1'), LOCAL, name)
    return list(dict.fromkeys(hits))


# ---- Plans ----------------------------------------------------------------------------------------------------------

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_freeze(path):
    """FREEZE.txt's sha256 entries: {path relative to its folder: sha256}."""
    entries = {}
    for line in Path(path).read_text(encoding='utf-8', errors='replace').splitlines():
        m = re.match(r'\s+([0-9a-f]{64}) \*(.+?)\s*$', line)
        if m:
            entries[m[2]] = m[1]
    return entries


def load_plans(plans_dir, sets, freeze_path=None):
    """The planned (run = True) pieces of each set, from the run plans and plan JSONs only, cross-checked against
    each other, and the plan files' provenance (sha256, FREEZE.txt check)."""
    plans_dir = Path(plans_dir)
    freeze_path = Path(freeze_path) if freeze_path else plans_dir.parent/'FREEZE.txt'
    frozen = read_freeze(freeze_path) if freeze_path.is_file() else None
    files, plans = {}, {}
    for set_name in sets:
        cfg = SETS[set_name]
        paths = [plans_dir/cfg['runplan'], plans_dir/cfg['plan']]
        for p in paths:
            if not p.is_file():
                raise PackageError(f'plan file missing: {p.name} (in --plans)')
            digest = sha256_file(p)
            entry = dict(sha256=digest)
            if frozen is not None:
                try:
                    rel = Path(os.path.relpath(p.resolve(), freeze_path.resolve().parent)).as_posix()
                except ValueError:  # another drive
                    rel = p.name
                if frozen.get(rel) != digest:
                    raise PackageError(f'{p.name} is not as frozen: FREEZE.txt has '
                                       f'{frozen.get(rel, "no entry")} for {rel}, the file {digest}')
                entry['frozen'] = True
            files[p.name] = entry
        with open(paths[0], newline='', encoding='utf-8') as fh:
            rows = [r for r in csv.DictReader(fh, delimiter='\t') if r.get('run') == 'True']
        plan = json.loads(paths[1].read_text(encoding='utf-8'))
        cross_tol = float(plan.get('provenance', {}).get('constants', {}).get('CROSS_TOL', CROSS_TOL))
        by_id = {p['id']: p for p in plan.get('pieces', [])}
        stretches = {s['id']: s for s in plan.get('stretches', [])}
        windows = {w['id']: w for w in plan.get('windows', [])}
        crossings = {c['id']: c for c in plan.get('crossings', [])}
        pieces = {}
        for row in rows:
            pid = row['piece']
            if not PIECE_ID.fullmatch(pid) or pid in pieces:
                raise PackageError(f'{cfg["runplan"]}: bad or repeated piece id {pid!r}')
            jp = by_id.get(pid)
            win = [w for w in row['windows'].split(',') if w]
            piece = dict(set=set_name, piece=pid, afv_id=int(row['afv_id']), family=row['family'],
                         fiber=row['fiber'], arc_patch_vox=[float(row['arc_start_patch_vox']),
                                                            float(row['arc_end_patch_vox'])],
                         decisions=int(row['decisions']), windows=win, parent_afv_id=int(row['parent_afv_id']),
                         head_k=[int(row['head_k_first']), int(row['head_k_last'])],
                         max_head_dev_sv=float(row['max_head_dev_sv']))
            problems = []
            if jp is None:
                problems.append('not in the plan JSON')
            else:
                for key in ('afv_id', 'fiber', 'family', 'decisions', 'windows'):
                    if jp.get(key) != piece[key]:
                        problems.append(f'{key} {piece[key]!r} in the run plan, {jp.get(key)!r} in the plan JSON')
                if jp.get('run') is not True:
                    problems.append('not run = true in the plan JSON')
                piece.update(annotator=jp.get('annotator') or piece['fiber'].split('_')[0],
                             fiber_file=jp.get('parent_file'), fiber_uuid=jp.get('parent_uuid'),
                             piece_uuid=jp.get('uuid'), line_index_range=jp.get('line_index_range'))
                if piece['annotator'] != piece['fiber'].split('_')[0]:
                    problems.append(f"annotator {piece['annotator']!r} is not the fiber name's prefix")
            covers = []
            for wid in win:
                if cfg['kind'] == 'stretches':
                    s = stretches.get(wid)
                    if s is None or s.get('piece', pid) != pid or s.get('fiber') != piece['fiber']:
                        problems.append(f'stretch {wid} is not a stretch of this piece in the plan JSON')
                        continue
                    covers.append(dict(id=wid, segment=s.get('segment'), s0_sv=s.get('s0'), s1_sv=s.get('s1'),
                                       length_sv=s.get('length_sv'), length_mm=s.get('length_mm')))
                else:
                    w = windows.get(wid)
                    c = crossings.get(w.get('crossing')) if w else None
                    if w is None or c is None or w.get('piece') != pid or w.get('fiber') != piece['fiber']:
                        problems.append(f'window {wid} is not a crossing window of this piece in the plan JSON')
                        continue
                    other = [x for x in c.get('windows', []) if x != wid]
                    pw = windows.get(other[0], {}) if other else {}
                    covers.append(dict(id=wid, crossing=c['id'], family=w.get('family'), s0_sv=w.get('s0'),
                                       s1_sv=w.get('s1'), partner_window=other[0] if other else None,
                                       partner_piece=pw.get('piece'), partner_fiber=pw.get('fiber'),
                                       closest_approach_sv=c.get('closest_approach_sv'),
                                       tight=bool(c.get('closest_approach_sv', float('inf')) <= cross_tol),
                                       z_tercile=c.get('z_tercile')))
            if problems:
                raise PackageError(f'{set_name} piece {pid}: the plan files disagree: '+'; '.join(problems))
            piece['covers'] = covers
            pieces[pid] = piece
        table = []
        if cfg['kind'] == 'crossings':
            for c in plan.get('crossings', []):
                if c.get('run') is True:
                    table.append(dict(id=c['id'], h_fiber=c.get('h_fiber'), v_fiber=c.get('v_fiber'),
                                      closest_approach_sv=c.get('closest_approach_sv'),
                                      tight=bool(c.get('closest_approach_sv', float('inf')) <= cross_tol),
                                      z_tercile=c.get('z_tercile'), point_sv=c.get('point_sv'),
                                      windows={w: windows.get(w, {}).get('piece') for w in c.get('windows', [])}))
        plans[set_name] = dict(pieces=pieces, crossings=table, cross_tol=cross_tol)
    freeze = dict(found=frozen is not None)
    if frozen is not None:
        freeze.update(file=freeze_path.name, sha256=sha256_file(freeze_path), plans_verified=True,
                      prereg_sha256=frozen.get('PREREG.md'),
                      afv_sha256={SETS[s]['afv']: frozen.get(SETS[s]['afv']) for s in sets},
                      runner_sha256={k: frozen.get(k) for k in ('runs/v3/run_all.ps1', 'runs/v3/infer_cpu.py')})
    return plans, files, freeze


# ---- Strips ---------------------------------------------------------------------------------------------------------

def _regular_file(p):
    return p.is_file() and not p.is_symlink()


def _plain_dir(p):
    junction = getattr(p, 'is_junction', lambda: False)()
    return p.is_dir() and not p.is_symlink() and not junction


def inventory_piece(piece_dir):
    """(members [(name, path)] in zip order, required names missing, names left out) of a piece folder."""
    strip = piece_dir/'strip'
    members, missing, ignored = [], [], []
    for name in sorted(STRIP_FILES+OPTIONAL_STRIP_FILES):
        if _plain_dir(strip) and _regular_file(strip/name):
            members.append((f'strip/{name}', strip/name))
        elif name in STRIP_FILES:
            missing.append(f'strip/{name}')
    for name in PIECE_FILES:
        if _regular_file(piece_dir/name):
            members.append((name, piece_dir/name))
        else:
            missing.append(name)
    for entry in sorted(piece_dir.iterdir()):
        if entry.name not in ('strip',)+PIECE_FILES or not (_plain_dir(entry) if entry.name == 'strip'
                                                             else _regular_file(entry)):
            ignored.append(entry.name+('/' if entry.is_dir() else ''))
    if _plain_dir(strip):
        for entry in sorted(strip.iterdir()):
            if entry.name not in STRIP_FILES+OPTIONAL_STRIP_FILES or not _regular_file(entry):
                ignored.append('strip/'+entry.name+('/' if entry.is_dir() else ''))
    return members, missing, ignored


def check_strip(piece, summary, trace):
    """Problems of a strip's own record against its plan row and the run's checkpoint and settings."""
    problems = []
    for where, record in (('summary.json', summary), ('strip/trace.json', trace)):
        if record.get('fiber') != piece['afv_id']:
            problems.append(f"{where} fiber {record.get('fiber')!r} is not the plan's AFV id {piece['afv_id']}")
        checkpoint = str(record.get('checkpoint', ''))
        if not checkpoint.startswith(CHECKPOINT_SOURCE):
            problems.append(f'{where} checkpoint {checkpoint!r} is not {CHECKPOINT}')
        for key, value in SETTINGS.items():
            try:
                same = float(record.get(key)) == value
            except (TypeError, ValueError):
                same = False
            if not same:
                problems.append(f'{where} {key} {record.get(key)!r} is not {value:g}')
    if summary.get('decisions') != piece['decisions']:
        problems.append(f"summary.json decisions {summary.get('decisions')!r} is not the plan's {piece['decisions']}")
    return problems


def run_status(strips_root):
    """The run's DONE line when the runner sits beside the strips folder (run_all.ps1 writes DONE when it ends and
    moves an old one away when it starts); refuses an unfinished run."""
    run_dir = strips_root.parent
    if not (run_dir/'run_all.ps1').is_file():
        return None
    done = run_dir/'DONE'
    if not done.is_file():
        raise PackageError('the run is not finished: no DONE beside the strips folder (--no-done-check to package '
                           'anyway)')
    lines = done.read_text(encoding='utf-8-sig', errors='replace').strip().splitlines()
    return f"DONE: {lines[0] if lines else ''}"


# ---- Writing --------------------------------------------------------------------------------------------------------

def _date_time(path):
    t = time.gmtime(os.stat(path).st_mtime)
    return max(t[:6], (1980, 1, 1, 0, 0, 0)) if t.tm_year < 2108 else (2107, 12, 31, 23, 59, 58)


def _file_info(name, date_time):
    info = zipfile.ZipInfo(name, date_time)
    info.compress_type, info.create_system, info.external_attr = zipfile.ZIP_STORED, 3, 0o100644 << 16
    return info


def _dir_info(name, date_time):
    info = zipfile.ZipInfo(name.rstrip('/')+'/', date_time)
    info.compress_type, info.create_system, info.external_attr = zipfile.ZIP_STORED, 3, (0o40755 << 16) | 0x10
    return info


def _inside(child, parent):
    try:
        Path(child).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def prepare_out(out, overwrite):
    """OUT/.staging, fresh. OUT may hold nothing but a previous release (replaced only with --overwrite)."""
    ours = set(RELEASE_FILES) | {cfg['zip'] for cfg in SETS.values()} | {STAGING}
    if out.exists():
        if not out.is_dir():
            raise PackageError(f'--out {out} is not a folder')
        others = sorted(p.name for p in out.iterdir() if p.name not in ours)
        if others:
            raise PackageError(f'--out {out} holds other files ({", ".join(others[:5])}); give an empty or new folder')
        if any(p.name in RELEASE_FILES or p.suffix == '.zip' for p in out.iterdir()) and not overwrite:
            raise PackageError(f'--out {out} holds a release already; pass --overwrite to replace it')
    staging = out/STAGING
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    return staging


def verify_zip(path, expected):
    """(local paths found, other problems) of a written zip read back: every member stored, its CRC and sha256 as
    recorded, no member missing or extra, and the scan of every member."""
    hits, problems = [], []
    with zipfile.ZipFile(path) as z:
        bad = z.testzip()
        if bad is not None:
            problems.append(f'{path.name}: CRC error in {bad}')
        names = []
        for info in z.infolist():
            if info.compress_type != zipfile.ZIP_STORED:
                problems.append(f'{path.name}: {info.filename} is not stored')
            if info.is_dir():
                hits += _hits(info.filename, LOCAL, f'{path.name}:{info.filename} (name)')
                continue
            data = z.read(info)
            names.append(info.filename)
            if sha256_bytes(data) != expected.get(info.filename):
                problems.append(f'{path.name}: {info.filename} differs from what was packaged')
            hits += [(f'{path.name}:{w}', m) for w, m in scan_member(info.filename, data)]
        if sorted(names) != sorted(expected):
            problems.append(f'{path.name}: members differ from the manifest')
    return hits, problems


# ---- README ---------------------------------------------------------------------------------------------------------

def _count(n, word, plural=None):
    return f'{n:,} {word}' if n == 1 else f'{n:,} {plural or word+"s"}'


def _and(items):
    items = list(items)
    return ', '.join(items[:-1])+' and '+items[-1] if len(items) > 1 else ''.join(items)


def readme(manifest, planned_annotators=()):
    sets, strips = manifest['sets'], manifest['strips']
    rows = []
    for name, s in sets.items():
        if s['kind'] == 'stretches':
            ids = [c['id'] for x in strips if x['set'] == name for c in x['stretches']]
            covers = ', '.join(ids) if 0 < len(ids) <= 4 else _count(len(ids), 'stretch', 'stretches')
        else:
            crossings = {c['crossing'] for x in strips if x['set'] == name for c in x['crossing_windows']}
            covers = f"{_count(s['covers_packaged'], 'window')} of {_count(len(crossings), 'crossing')}"
        zip_name = f"`{s['zip']}`" if s.get('zip') else '(none: no strip)'
        rows.append(f"| {zip_name} | {s['title']} | {s['packaged_pieces']} of {s['planned_pieces']} | "
                    f"{s['decisions_packaged']:,} | {covers} |")
    gaps = []
    for name, s in sets.items():
        if s['missing_pieces']:
            gaps.append(f"- {name}: no strip for {', '.join(s['missing_pieces'])} (the run did not write one).")
        for x in s['incomplete_pieces']:
            gaps.append(f"- {name}: {x['piece']} left out, its output is incomplete (no {', '.join(x['missing'])}).")
    gap_text = ('Planned pieces without a packaged strip:\n'+'\n'.join(gaps)) if gaps else \
        'Every planned piece of these sets has a strip.'
    fields = manifest['scrub']['fields']
    field_lines = [f"- `{member}` `{field}`: {_count(n, 'strip')}" for member, f in sorted(fields.items())
                   for field, n in sorted(f.items())]
    field_text = ('Fields that held a local path:\n'+'\n'.join(field_lines)) if field_lines else \
        'No member held a local path.'
    if any(field == 'afv' for f in fields.values() for field in f):
        packaged = [n for n, s in sets.items() if s['packaged_pieces']]
        if packaged and all(sets[n]['afv_is_runner_file'] for n in packaged):
            named = _and(f"`score/{SETS[n]['afv']}` ({n})" for n in packaged)
            field_text += (f'\n\n`afv` held the path of the piece AFV that infer.py read: {named}, which the\n'
                           'repository rebuilds (`score/runs/plan_v3/run_afv.sh`) and does not ship.')
        else:
            field_text += '\n\n`afv` held the path of the AFV file that infer.py read.'
    kept = manifest['scrub']['relative_paths_kept']
    if kept:
        field_text += ('\n\nKept as they are (relative paths, no local root): ' +
                       ', '.join(f'`{m}` `{f}`' for m, fs in sorted(kept.items()) for f in sorted(fs)) + '.')
    grids = manifest['grid']
    grid_text = (f"rows run along the fiber, one every {grids['row_step_voxels']:g} sv, and its "
                 f"{grids['columns']} columns run across it, {grids['column_step_voxels']:g} sv apart"
                 if grids else 'rows run along the fiber and columns across it (spacing in `summary.json`)')
    expected = sum(1 for x in strips if any(m['name'].endswith('/strip/expected_error.tif') for m in x['members']))
    expected_text = (f" `expected_error.tif` ({_count(expected, 'strip')}: the model's expected |error| per node, "
                     'voxels)') if expected else ''
    annotators = sorted(set(planned_annotators) or {x['annotator'] for x in strips})
    vols = sorted({SETS[n]['scroll'] for n in sets})
    volume_lines = '\n'.join(f'  - {v}: `{VOLUMES[v]}`' for v in vols)
    run_line = f" The run's own record: `{manifest['run_status']}`." if manifest.get('run_status') else ''
    return f"""# Fiber strips beyond Paris 4: `fiber_strip_small_v1` on PHerc0139 and PHerc0332

Sheet strips predicted by sean's (bruniss) strip model `fiber_strip_small_v1` (villa #2110) along pieces of
human-traced fibers of PHerc0139 and PHerc0332, two scrolls outside its training data as far as public records show
(the checkpoint's `run.json` fine-tunes on PHerc Paris 4 fibers from an earlier checkpoint, and no training config
in villa's `tracers/fibers/configs/`, for the fiber follower, frame model and sheet strip, lists PHerc0139 or
PHerc0332; configs of other villa models do). They are the strips of the pre-registered CPU evaluation in
{REPO}
(`score/PREREG.md` v3.2, frozen in `score/FREEZE.txt`), as its runner `score/runs/v3/run_all.ps1` wrote them.{run_line}
This release holds the strips and their mapping to the fibers. The scores are in the repository, not here.

| Zip | Set | Pieces | Decisions | Covers |
|---|---|---|---|---|
{chr(10).join(rows)}

{gap_text}

`SHA256SUMS.txt` holds the sha256 of every file of this release; `manifest.json` the size and sha256 of every zip
member.

## What a strip is

For each piece, villa's `sheet_strip/infer.py` makes a decision every 8 trace voxels (16 sv) along the fiber; each
predicts the height of the papyrus sheet the fiber lies on, across the fiber, and the decisions are stitched into one
strip: a tifxyz surface whose {grid_text}.
- Coordinates are xyz in patch voxels: sv, level 2 of the 2.399 um scans (4 level-0 voxels, 9.596 um).
- These are pieces, not whole fibers: each covers the stretches or crossing windows listed in `manifest.json`, with
  margins (PREREG 4.6). Nodes within 75 sv of a piece end are left out.
- Volumes:
{volume_lines}

## Layout

Each zip holds one folder per piece, `<set>/<piece id>/`, as the run wrote it. Extract the zips into one folder to
get the run's strips folder; `score/score_strips.py` reads a set with `--strips <folder>/<set>`.
- `strip/`: `x.tif`, `y.tif`, `z.tif` (float32, -1 off the strip), `mask.tif` (255 on the strip), `meta.json`
  (tifxyz metadata; its `area_cm2` uses infer.py's 9.6 um default), `trace.json` (checkpoint, settings, the fiber's
  AFV id, arclength of the first row).{expected_text}
- `strip.npz`: per decision `arcs` (trace voxels), `positions`, `frames`, `heights`, `flipped`; the model grid `f`,
  `v`; per node `spread`; `trace_per_patch`; and `expected_error_vox` (the model's expected |error| per decision and
  grid point).
- `summary.json`: the piece's settings and counts.
- Members are stored, not compressed.

## Which strip is which

`manifest.json` is built from the frozen plan files only (`score/results/runplan_*.tsv` and the plan JSONs), never
from the strips. Per strip: its set, piece id, fiber (the name of the human fiber JSON), annotator, family (H or V),
the stretches (id, segment, arclength range on the fiber in sv, length) or crossing windows (id, crossing, partner
window and piece, closest approach) it covers, its AFV ids, `--arc` and decisions, and every member's size and
sha256. For PHerc0332 it also lists the planned crossings with the pieces of their two windows. Each strip's own
`summary.json` was checked against its plan row (AFV id, decisions) and the run's checkpoint and settings.

## How they were made

- Code: villa `{VILLA[:9]}` + PR #2163 (the native Windows port by slchicoineau-creator and Baz248) + a 20-line
  patch that binds the per-crop z-score normalization record to the CT actually read (a no-op on Paris 4); the
  diffs are in `patches/` in the repository.
- Checkpoint: `{CHECKPOINT}`
  (update 15000), with its frame model.
- One `infer.py` process per piece, on the CPU (8 torch threads, fp32):
  `{INFER_ARGS}`,
  with `--afv`, `--fiber` and `--arc` from the frozen run plans, the scroll's CT (`--ct`, the volumes above) and its
  level-2 umbilicus. A piece whose run failed twice has no strip.
- Fibers: the PHerc0139 and PHerc0332 human fiber scaffolds, cut into pieces at their own line points (PREREG 4.6).

## Local paths removed

Every absolute local path in the JSON members (and in any string array of `strip.npz`) was replaced by the token
`<local>`; nothing else changed. A member with nothing to replace is byte for byte as the run wrote it, and the TIFFs
never change. After writing, every member was read back and scanned for local paths (JSON strings, TIFF tags, npz
headers and arrays, and every byte for the workspace folder names); none was left.

{field_text}

## Not included

No CT, no raw human fiber JSON, no segment mesh, no model weights, no published Paris 4 strip, no render.

## License

CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/): the strips are derived from Vesuvius Challenge open
data and from the Vesuvius Challenge human fiber annotations, by running `fiber_strip_small_v1` on the human fibers.
See `LICENSE-DATA.md` in the repository. No restriction is added beyond CC BY-NC 4.0.

## Credits

- sean (bruniss, Sean Johnson): the strip model `fiber_strip_small_v1` and the sheet-strip method, villa PR #2110;
  also the frame model and `match_fibers_to_patches.py` (villa PR #2045), whose +/-2 sv window and `--min-run 40`
  the PHerc0139 stretch rules use.
- slchicoineau-creator (Sebastien Chicoineau) and Baz248 (Chris Thomas): villa PR #2163, the Windows port the strips
  were made with.
- The fiber annotators {_and(annotators)}: the PHerc0139 and PHerc0332 human fiber scaffolds
  (https://dl.ash2txt.org/other/annotators/0139_fibers/ and /0332_fibers/).
- hja (surfaces built from H/V fiber intersections) and Qual (surfaces from H and V fiber splines that constrain each
  other), both without the strip model, and sean (the remark that strips could be used to estimate winding counts
  from H-on-V crossings), in the Discord thread on automated fiber volumes, 20 Oct 2026: related work for the
  PHerc0332 crossing windows.
- Vesuvius Challenge open data (s3://vesuvius-challenge-open-data, CC BY-NC 4.0): the PHerc0139 and PHerc0332 scans.
  Data citation (https://scrollprize.org/data#how-to-cite): Giorgio Angelotti, Stephen Parsons, Sean Johnson, Elian
  Rafael Dal Prà, Johannes Rudolph, Paul Tafforeau, Alessandro Mirone, Paul Henderson, Hendrik Schilling, Forrest
  McDonald, David Josey, Youssef Nader, C. Seth Parker, W. Brent Seales. Vesuvius Challenge - CT Scans of
  Herculaneum Papyri. Vesuvius Challenge.

Author: TAUIL Abd Elilah. Developed with AI assistance under my direction.
"""


# ---- Main -------------------------------------------------------------------------------------------------------

def build(args):
    strips_root, plans_dir, out = Path(args.strips), Path(args.plans), Path(args.out)
    sets = [s for s in SETS if s in [x.strip() for x in args.sets.split(',')]]
    unknown = [x for x in args.sets.split(',') if x.strip() not in SETS]
    if unknown or not sets:
        raise PackageError(f'--sets: unknown set(s) {unknown}; known: {", ".join(SETS)}')
    if not strips_root.is_dir():
        raise PackageError(f'--strips {strips_root} is not a folder')
    if not plans_dir.is_dir():
        raise PackageError(f'--plans {plans_dir} is not a folder')
    if not args.dry_run and (_inside(out, strips_root) or _inside(out, plans_dir) or _inside(strips_root, out)):
        raise PackageError('--out must be outside --strips and --plans, and must not hold them')
    status = None if args.no_done_check else run_status(strips_root)
    plans, plan_files, freeze = load_plans(plans_dir, sets, args.freeze)
    if not freeze['found']:
        print('warning: no FREEZE.txt beside --plans: the plan files are not checked against the freeze',
              file=sys.stderr)

    # what is there
    inventory, unexpected = {}, []
    for set_name in sets:
        planned = plans[set_name]['pieces']
        set_dir = strips_root/set_name
        found, incomplete, ignored, stray = {}, [], {}, []
        if _plain_dir(set_dir):
            for entry in sorted(set_dir.iterdir()):
                if not _plain_dir(entry):
                    stray.append(entry.name)
                elif entry.name not in planned:
                    unexpected.append(f'{set_name}/{entry.name}')
                else:
                    members, missing, left_out = inventory_piece(entry)
                    if missing:
                        incomplete.append(dict(piece=entry.name, missing=missing))
                    else:
                        found[entry.name] = members
                    if left_out:
                        ignored[entry.name] = left_out
        missing_pieces = sorted(set(planned)-set(found)-{x['piece'] for x in incomplete})
        inventory[set_name] = dict(found=found, incomplete=incomplete, ignored=ignored, stray=stray,
                                   missing=missing_pieces)
    if unexpected:
        raise PackageError('piece folders that are in no plan of their set: '+', '.join(unexpected))
    gaps = [(s, p) for s in sets for p in inventory[s]['missing']+[x['piece'] for x in inventory[s]['incomplete']]]
    if gaps and args.require_complete:
        raise PackageError(f'{len(gaps)} planned pieces have no complete strip: '
                           + ', '.join(f'{s}/{p}' for s, p in gaps), code=3)

    # check every strip against its plan before anything is written
    problems = []
    for set_name in sets:
        for pid, members in inventory[set_name]['found'].items():
            files = dict(members)
            try:
                summary = json.loads(files['summary.json'].read_text(encoding='utf-8-sig'))
                trace = json.loads(files['strip/trace.json'].read_text(encoding='utf-8-sig'))
            except ValueError as error:
                problems.append(f'{set_name}/{pid}: unreadable JSON ({error})')
                continue
            if not isinstance(summary, dict) or not isinstance(trace, dict):
                problems.append(f'{set_name}/{pid}: summary.json or strip/trace.json is not a JSON object')
                continue
            problems += [f'{set_name}/{pid}: {p}' for p in check_strip(plans[set_name]['pieces'][pid], summary, trace)]
            inventory[set_name].setdefault('grid', {})[pid] = (summary.get('row_step_voxels'),
                                                              summary.get('column_step_voxels'),
                                                              summary.get('columns'))
            afv = str(summary.get('afv', '')).replace('\\', '/')  # for the README: is it run_all.ps1's AFV?
            inventory[set_name].setdefault('afv_is_runner_file', []).append(
                afv == SETS[set_name]['afv'] or afv.endswith('/score/'+SETS[set_name]['afv']))
    if problems:
        raise PackageError('strips that do not match their plan or the run:\n  '+'\n  '.join(problems))

    staging = None
    created_out = not out.exists()
    if not args.dry_run:
        staging = prepare_out(out, args.overwrite)
    try:
        manifest, written = package(args, sets, plans, plan_files, freeze, inventory, staging, status)
        if staging is not None:
            for p in list(out.iterdir()):
                if p.name in RELEASE_FILES or (p.suffix == '.zip' and p.is_file()):
                    p.unlink()
            for name in written:
                os.replace(staging/name, out/name)
            shutil.rmtree(staging)
    except BaseException:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
            if created_out and out.exists() and not any(out.iterdir()):
                out.rmdir()
        raise
    return manifest


def package(args, sets, plans, plan_files, freeze, inventory, staging, status):
    """Scrub, scan, write (unless dry run) and verify; returns the manifest and the files written to staging."""
    strips, fields, kept, hits, set_info, written = [], {}, {}, [], {}, []
    for set_name in sets:
        cfg, inv, plan = SETS[set_name], inventory[set_name], plans[set_name]
        zip_path = staging/cfg['zip'] if staging is not None and inv['found'] else None
        zf = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED, allowZip64=True) if zip_path else None
        expected = {}
        try:
            if zf is not None:
                zf.writestr(_dir_info(set_name, (1980, 1, 1, 0, 0, 0)), b'')
            for pid in sorted(inv['found']):
                piece, members = plan['pieces'][pid], inv['found'][pid]
                base = f'{set_name}/{pid}'
                stamp = max(_date_time(path) for _, path in members)
                if zf is not None:
                    zf.writestr(_dir_info(base, stamp), b'')
                    zf.writestr(_dir_info(base+'/strip', stamp), b'')
                entries, scrubbed = [], []
                for rel, path in members:
                    data = path.read_bytes()
                    try:
                        new, changed, relative = scrub_member(rel, data)
                    except ScrubError as error:
                        raise ScrubError(f'{base}/{rel}: {error}') from None
                    found = scan_member(f'{base}/{rel}', new)
                    hits += found
                    for f in changed:
                        fields.setdefault(rel, {}).setdefault(f, 0)
                        fields[rel][f] += 1
                    for f in relative:
                        kept.setdefault(rel, {}).setdefault(f, 0)
                        kept[rel][f] += 1
                    scrubbed += [f'{rel}:{f}' for f in changed]
                    name = f'{base}/{rel}'
                    digest = sha256_bytes(new)
                    expected[name] = digest
                    entries.append(dict(name=name, bytes=len(new), sha256=digest, scrubbed=new != data))
                    if zf is not None and not found:
                        zf.writestr(_file_info(name, _date_time(path)), new)
                record = dict(set=set_name, piece=pid, zip=cfg['zip'], dir=base+'/', scroll=cfg['scroll'],
                              fiber=piece['fiber'], fiber_file=piece.get('fiber_file'),
                              fiber_uuid=piece.get('fiber_uuid'), annotator=piece['annotator'],
                              family=piece['family'], afv_id=piece['afv_id'], parent_afv_id=piece['parent_afv_id'],
                              piece_uuid=piece.get('piece_uuid'), line_index_range=piece.get('line_index_range'),
                              arc_patch_vox=piece['arc_patch_vox'], decisions=piece['decisions'],
                              head_k=piece['head_k'], max_head_dev_sv=piece['max_head_dev_sv'])
                record['stretches' if cfg['kind'] == 'stretches' else 'crossing_windows'] = piece['covers']
                record.update(members=entries, scrubbed_fields=scrubbed,
                              left_out=inv['ignored'].get(pid, []))
                strips.append(record)
        finally:
            if zf is not None:
                zf.close()
        if hits:
            break
        planned = plan['pieces']
        packaged = [planned[p] for p in sorted(inv['found'])]
        info = dict(title=cfg['title'], kind=cfg['kind'], scroll=cfg['scroll'], volume=VOLUMES[cfg['scroll']],
                    afv=f"score/{cfg['afv']}", runplan=f"score/results/{cfg['runplan']}",
                    plan=f"score/results/{cfg['plan']}", zip=cfg['zip'] if inv['found'] else None,
                    planned_pieces=len(planned), packaged_pieces=len(packaged),
                    decisions_planned=sum(p['decisions'] for p in planned.values()),
                    decisions_packaged=sum(p['decisions'] for p in packaged),
                    covers_planned=sum(len(p['covers']) for p in planned.values()),
                    covers_packaged=sum(len(p['covers']) for p in packaged),
                    fibers_packaged=len({p['fiber'] for p in packaged}),
                    annotators=_tally(p['annotator'] for p in packaged),
                    families=_tally(p['family'] for p in packaged),
                    missing_pieces=inv['missing'], incomplete_pieces=inv['incomplete'],
                    left_out=inv['ignored'], stray_files=inv['stray'],
                    afv_is_runner_file=bool(inv.get('afv_is_runner_file')) and all(inv['afv_is_runner_file']))
        if zip_path is not None:
            size = zip_path.stat().st_size
            if size >= GITHUB_ASSET_LIMIT:
                raise PackageError(f'{zip_path.name} is {size:,} bytes: a GitHub release asset must be under 2 GB',
                                   code=1)
            zip_hits, zip_problems = verify_zip(zip_path, expected)
            hits += zip_hits
            if zip_problems:
                raise PackageError('a written zip failed its check: '+'; '.join(zip_problems), code=1)
            info.update(zip_bytes=size, zip_sha256=sha256_file(zip_path))
            written.append(zip_path.name)
        set_info[set_name] = info
    if hits:
        raise ScrubError('local paths left (nothing was released):\n  '
                         + '\n  '.join(f'{w}: {m!r}' for w, m in hits[:50])
                         + (f'\n  ... and {len(hits)-50} more' if len(hits) > 50 else ''))
    grids = {g for s in sets for g in inventory[s].get('grid', {}).values()}
    grid = dict(zip(('row_step_voxels', 'column_step_voxels', 'columns'), grids.pop())) \
        if len(grids) == 1 and None not in next(iter(grids)) else None
    tool = Path(__file__)
    manifest = dict(
        release='fiber_strip_small_v1 strips on PHerc0139 and PHerc0332 (fiber-strips-beyond-paris4)',
        created_utc=time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()), repository=REPO,
        license='CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/); see LICENSE-DATA.md',
        packager=dict(tool='tools/package_strips.py', sha256=sha256_file(tool)),
        run_status=status,
        provenance=dict(prereg='score/PREREG.md v3.2', freeze=freeze,
                        plan_files={f'score/results/{k}': v for k, v in plan_files.items()},
                        runner='score/runs/v3/run_all.ps1 (one infer.py process per piece)',
                        villa=f'{VILLA} + PR #2163 + patches/strip_bind_normalization_other_volume.diff',
                        checkpoint=CHECKPOINT, infer_args=INFER_ARGS,
                        settings_checked=dict(SETTINGS, checkpoint=CHECKPOINT_SOURCE+'*')),
        layout=dict(zip_method='stored', piece_dir='<set>/<piece id>/',
                    members=[f'strip/{f}' for f in sorted(STRIP_FILES+OPTIONAL_STRIP_FILES)]+list(PIECE_FILES),
                    optional=[f'strip/{f}' for f in OPTIONAL_STRIP_FILES]),
        grid=grid,
        scrub=dict(token=TOKEN, members_scanned=sum(len(x['members']) for x in strips),
                   rule='every absolute local path in JSON and text members and in npz string arrays replaced by '
                        'the token, nothing else changed; TIFFs and numeric arrays never rewritten; every member '
                        'read back and scanned after writing',
                   fields=fields, relative_paths_kept=kept, left=0),
        sets=set_info,
        crossings=[dict(c, windows={w: dict(piece=p, packaged=any(x['set'] == s and x['piece'] == p
                                                                  for x in strips))
                                    for w, p in c['windows'].items()})
                   for s in sets for c in plans[s]['crossings']],
        strips=strips)
    planned_annotators = {p['annotator'] for s in sets for p in plans[s]['pieces'].values()}
    texts = {'manifest.json': (json.dumps(manifest, indent=1)+'\n').encode('utf-8'),
             'README.md': readme(manifest, planned_annotators).encode('utf-8')}
    for name, data in texts.items():
        found = scan_member(name, data)
        if found:
            raise ScrubError(f'{name} would hold a local path: {found[:5]}')
    if staging is not None:
        for name, data in texts.items():
            (staging/name).write_bytes(data)
            written.append(name)
        sums = ''.join(f'{sha256_file(staging/n)}  {n}\n' for n in written)
        (staging/'SHA256SUMS.txt').write_bytes(sums.encode('utf-8'))
        written.append('SHA256SUMS.txt')
        for name in written:  # the files as they will be released, read back
            if name.endswith('.zip'):
                continue
            found = scan_member(name, (staging/name).read_bytes())
            if found:
                raise ScrubError(f'{name} holds a local path: {found[:5]}')
    return manifest, written


def _tally(values):
    out = {}
    for v in values:
        out[v] = out.get(v, 0)+1
    return dict(sorted(out.items()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--strips', required=True, help='the run\'s strips folder (STRIPS/<set>/<piece id>/)')
    ap.add_argument('--plans', required=True, help='the frozen plan files (score/results)')
    ap.add_argument('--out', required=True, help='the release folder (new, empty, or a previous release)')
    ap.add_argument('--sets', default=','.join(SETS), help=f'comma-separated, default {",".join(SETS)}')
    ap.add_argument('--dry-run', action='store_true', help='read, map, scrub and scan; write nothing')
    ap.add_argument('--overwrite', action='store_true', help='replace a previous release in --out')
    ap.add_argument('--require-complete', action='store_true', help='fail when a planned piece has no strip')
    ap.add_argument('--freeze', help='FREEZE.txt to check the plan files against (default: beside --plans)')
    ap.add_argument('--no-done-check', action='store_true', help='package even if the run is not finished')
    args = ap.parse_args(argv)
    try:
        manifest = build(args)
    except PackageError as error:
        print(f'package_strips: {error}', file=sys.stderr)
        return error.code
    report = dict(dry_run=args.dry_run, out=None if args.dry_run else str(args.out),
                  sets={k: {f: v.get(f) for f in ('zip', 'packaged_pieces', 'planned_pieces', 'decisions_packaged',
                                                  'covers_packaged', 'missing_pieces', 'incomplete_pieces',
                                                  'left_out', 'zip_bytes', 'zip_sha256')}
                        for k, v in manifest['sets'].items()},
                  plans_frozen=manifest['provenance']['freeze'].get('plans_verified', False),
                  run_status=manifest['run_status'], members_scanned=manifest['scrub']['members_scanned'],
                  fields_that_held_paths=manifest['scrub']['fields'],
                  relative_paths_kept=manifest['scrub']['relative_paths_kept'], local_paths_left=0)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
