"""Extract stored (method 0) members of a remote zip from one byte range [a, b) via HTTP range; writes into OUT."""
import struct, sys, urllib.request, os, zlib
URL, a, b, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
req = urllib.request.Request(URL, headers={'Range': f'bytes={a}-{b-1}'})
blob = urllib.request.urlopen(req, timeout=120).read()
pos = 0
while pos + 30 <= len(blob) and blob[pos:pos+4] == b'PK\x03\x04':
    (sig, vneed, flag, method, mt, md, crc, csize, usize, nlen, xlen) = struct.unpack('<IHHHHHIIIHH', blob[pos:pos+30])
    name = blob[pos+30:pos+30+nlen].decode()
    extra = blob[pos+30+nlen:pos+30+nlen+xlen]
    k = 0
    while k + 4 <= len(extra):
        hid, hsz = struct.unpack('<HH', extra[k:k+4])
        if hid == 1:
            vals = extra[k+4:k+4+hsz]; q = 0
            if usize == 0xFFFFFFFF: usize = struct.unpack('<Q', vals[q:q+8])[0]; q += 8
            if csize == 0xFFFFFFFF: csize = struct.unpack('<Q', vals[q:q+8])[0]; q += 8
        k += 4 + hsz
    start = pos + 30 + nlen + xlen
    data = blob[start:start+csize]
    if flag & 8:
        raise SystemExit('data descriptor not handled')
    assert method == 0, method
    if not name.endswith('/'):
        assert zlib.crc32(data) & 0xFFFFFFFF == crc, name
        safe = os.path.normpath(name).replace('..', '_')
        path = os.path.join(OUT, safe)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, 'wb').write(data)
        print('wrote', path, len(data))
    pos = start + csize
