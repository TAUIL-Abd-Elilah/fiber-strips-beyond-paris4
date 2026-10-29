"""List keys (size, key) under an S3 prefix of the public open-data bucket, with pagination; optional substring filter."""
import sys, urllib.request, urllib.parse, re
B = 'https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/'
prefix, sub = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else '')
token = None
while True:
    q = {'list-type': '2', 'prefix': prefix, 'max-keys': '1000'}
    if token: q['continuation-token'] = token
    xml = urllib.request.urlopen(B + '?' + urllib.parse.urlencode(q), timeout=120).read().decode()
    for key, size in re.findall(r'<Key>([^<]*)</Key>.*?<Size>(\d+)</Size>', xml, re.S):
        if sub in key:
            print(size, key)
    m = re.search(r'<NextContinuationToken>([^<]*)</NextContinuationToken>', xml)
    if not m: break
    token = m.group(1)
