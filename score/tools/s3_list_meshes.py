"""For each PHerc0139 segment dir, list the keys under <seg>/mesh/ (size key)."""
import sys, urllib.request, urllib.parse, re
from concurrent.futures import ThreadPoolExecutor
B = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/'
root = sys.argv[1]
def ls(prefix, delim=None):
    out, token = [], None
    while True:
        q = {'list-type': '2', 'prefix': prefix, 'max-keys': '1000'}
        if delim: q['delimiter'] = delim
        if token: q['continuation-token'] = token
        xml = urllib.request.urlopen(B + '?' + urllib.parse.urlencode(q), timeout=120).read().decode()
        out += [('P', p) for p in re.findall(r'<CommonPrefixes><Prefix>([^<]*)</Prefix>', xml)]
        out += [(s, k) for k, s in re.findall(r'<Key>([^<]*)</Key>.*?<Size>(\d+)</Size>', xml, re.S)]
        m = re.search(r'<NextContinuationToken>([^<]*)</NextContinuationToken>', xml)
        if not m: return out
        token = m.group(1)
segs = [p for t, p in ls(root, '/') if t == 'P']
with ThreadPoolExecutor(8) as ex:
    for seg, rows in zip(segs, ex.map(lambda s: ls(s + 'mesh/'), segs)):
        for s, k in rows:
            print(s, k)
