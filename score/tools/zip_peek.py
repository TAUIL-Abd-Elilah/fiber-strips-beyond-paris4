"""Read a remote zip's end records and the start of its central directory via HTTP range requests (no full download)."""
import struct, sys, urllib.request, json, zlib

URL = sys.argv[1]
N_ENTRIES = int(sys.argv[2]) if len(sys.argv) > 2 else 60

def rng(a, b):
    req = urllib.request.Request(URL, headers={'Range': f'bytes={a}-{b}'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read(), r.headers.get('Content-Range')

def size():
    req = urllib.request.Request(URL, method='HEAD')
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers['Content-Length'])

total = size()
tail, _ = rng(total - 65536 - 22, total - 1)
i = tail.rfind(b'PK\x05\x06')
eocd = tail[i:i+22]
_, disk, cdisk, n_disk, n_total, cd_size, cd_off, clen = struct.unpack('<IHHHHIIH', eocd)
info = dict(total=total, n_total=n_total, cd_size=cd_size, cd_off=cd_off)
j = tail.rfind(b'PK\x06\x07')
if j >= 0:
    _, _, z64_off, _ = struct.unpack('<IIQI', tail[j:j+20])
    z64, _ = rng(z64_off, z64_off + 56 - 1)
    (sig, rec_size, vm, vn, d, dcd, n_d, n_t, cds, cdo) = struct.unpack('<IQHHIIQQQQ', z64)
    info.update(zip64=True, n_total=n_t, cd_size=cds, cd_off=cdo)
print(json.dumps(info))
cd, _ = rng(info['cd_off'], info['cd_off'] + min(info['cd_size'], 400000) - 1)
pos, entries = 0, []
while pos + 46 <= len(cd) and len(entries) < N_ENTRIES:
    if cd[pos:pos+4] != b'PK\x01\x02':
        break
    (sig, vmade, vneed, flag, method, mtime, mdate, crc, csize, usize, nlen, xlen, clen, dstart, iattr, eattr,
     lho) = struct.unpack('<IHHHHHHIIIHHHHHII', cd[pos:pos+46])
    name = cd[pos+46:pos+46+nlen].decode('utf-8', 'replace')
    extra = cd[pos+46+nlen:pos+46+nlen+xlen]
    # zip64 extra
    k = 0
    while k + 4 <= len(extra):
        hid, hsz = struct.unpack('<HH', extra[k:k+4])
        if hid == 1:
            vals = extra[k+4:k+4+hsz]
            q = 0
            if usize == 0xFFFFFFFF: usize = struct.unpack('<Q', vals[q:q+8])[0]; q += 8
            if csize == 0xFFFFFFFF: csize = struct.unpack('<Q', vals[q:q+8])[0]; q += 8
            if lho == 0xFFFFFFFF: lho = struct.unpack('<Q', vals[q:q+8])[0]; q += 8
        k += 4 + hsz
    entries.append(dict(name=name, method=method, csize=csize, usize=usize, lho=lho))
    pos += 46 + nlen + xlen + clen
print(json.dumps(entries, indent=0))
