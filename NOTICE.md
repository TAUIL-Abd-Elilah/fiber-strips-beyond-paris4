# Third-party code

## ScrollPrize/villa (MIT)

Some files here contain or modify code from ScrollPrize/villa (https://github.com/ScrollPrize/villa):

- `patches/pr2163_full.diff`, `patches/applied_pr2163_on_6f2024569.diff`,
  `patches/applied_pr2163_on_6f2024569_with_tests.diff`: villa PR #2163, the native Windows port of the fiber
  follower and the strip model, by Sebastien Chicoineau (slchicoineau-creator) and Chris Thomas (Baz248). The PR
  was open and unmerged when it was applied (head bfad589dc84240fb80e11d3c3fdeb341fcc1a62c). It was applied to
  villa 6f2024569 with no changes (`patches/APPLIED.txt`). The PR body discloses help from Claude Code.
- `patches/strip_bind_normalization_other_volume.diff`: our patch to villa's `sheet_strip/infer.py` (20 lines), so
  that the strip model accepts a CT volume other than PHerc Paris 4. It is a no-op on Paris 4.
- `patches/worktree_full_after_g0.diff`: both of the above as one diff.
- `score/score_strips.py` (and its v1 and v2 versions in `score/superseded_*`): `orient_decisions`, `stitch` and
  `stitch_values` are copied from villa's `sheet_strip/infer.py` at 6f2024569.

villa's license, which applies to those parts:

```
MIT License

Copyright (c) 2024 Vesuvius Challenge

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Not included

- The strip model weights (`scrollprize/fiber_strip_small_v1` on Hugging Face) declare no license. They are
  downloaded at run time (`fetch_inputs.py checkpoint`), pinned to revision 7bd163a22c839ff312879ae1916195d0c9729c0e.
- No code from modaribrahim/fiber-tracer-heldout-eval or from ChaseHendrick's villa PRs is copied; they are cited
  in the README.
