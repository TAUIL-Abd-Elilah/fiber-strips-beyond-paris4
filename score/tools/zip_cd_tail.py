"""List the central-directory entries in the last N bytes of a remote zip64's central directory (names that are not
fiber_* strip members), via HTTP range requests."""
import struct, sys, urllib.request, json
URL, TAIL = sys.argv[1], int(sys.argv[2])
CD_OFF, CD_SIZE = int(sys.argv[3]), int(sys.argv[4])
a = CD_OFF + CD_SIZE - TAIL
req = urllib.request.Request(URL, headers={'Range': f'bytes={a}-{CD_OFF+CD_SIZE-1}'})
cd = urllib.request.urlopen(req, timeout=120).read()
pos = cd.find(b'PK\x01\x02')
out = []
while pos >= 0 and pos + 46 <= len(cd):
    (sig, vmade, vneed, flag, method, mtime, mdate, crc, csize, usize, nlen, xlen, clen, dstart, iattr, eattr,
     lho) = struct.unpack('<IHHHHHHIIIHHHHHII', cd[pos:pos+46])
    name = cd[pos+46:pos+46+nlen].decode('utf-8', 'replace')
    extra = cd[pos+46+nlen:pos+46+nlen+xlen]
    k = 0
    while k + 4 <= len(extra):
        hid, hsz = struct.unpack('<HH', extra[k:k+4])
        if hid == 1:
            vals = extra[k+4:k+4+hsz]; q = 0
            if usize == 0xFFFFFFFF: usize = struct.unpack('<Q', vals[q:q+8])[0]; q += 8
            if csize == 0xFFFFFFFF: csize = struct.unpack('<Q', vals[q:q+8])[0]; q += 8
            if lho == 0xFFFFFFFF: lho = struct.unpack('<Q', vals[q:q+8])[0]; q += 8
        k += 4 + hsz
    out.append((name, method, csize, usize, lho))
    pos = cd.find(b'PK\x01\x02', pos + 46 + nlen + xlen + clen)
others = [o for o in out if '/fiber_' not in o[0]]
print(len(out), 'entries in tail; last strip entry:', [o for o in out if '/fiber_' in o[0]][-1:])
for o in others: print(o)
