"""Scale an open-data umbilicus JSON from level-0 voxels to the strip model's patch voxels (CT level 2 = L0 / 4).

villa's sheet_strip/data.load_umbilicus reads x, y, z of 'control_points' as patch voxels (the PHerc Paris 4
umbilicus.json on dl.ash2txt.org is in level-2 voxels already). The PHerc0139 / PHerc0332 files under
SAMPLE/representations/umbilicus/ are in level-0 voxels (their metadata: volume_width/height/slices = the L0 shape).

    python umbilicus_l0_to_l2.py IN.json OUT.json [--factor 4]
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--factor', type=float, default=4.)
    a = ap.parse_args()
    o = json.load(open(a.src))
    md = o.get('metadata', {})
    if 'volume_slices' in md:
        zmax = max(p['z'] for p in o['control_points'])
        assert zmax <= md['volume_slices'], 'control points beyond the L0 slice count: not level 0?'
    o['control_points'] = [dict(p, x=p['x']/a.factor, y=p['y']/a.factor, z=p['z']/a.factor) for p in o['control_points']]
    o['metadata'] = dict(md, coordinate_level=2, scaled_from_level0_by=a.factor, scaled_by='umbilicus_l0_to_l2.py')
    json.dump(o, open(a.dst, 'w'), indent=1)


if __name__ == '__main__':
    main()
