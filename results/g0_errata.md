# G0 record: errata (28 Oct 2026)

`results/g0.json` and the other files of the setup run are left unchanged. This file corrects figures that the G0
write-up stated wrongly. An independent verifier found these points; each one is checked against the file named.
None of them changes the G0 verdict (10 of 10 Paris 4 strips reproduced, mean of means 0.012 voxels).

| # | Stated | Correct | Source |
|---|---|---|---|
| 1 | Step-6 CT for the whole scroll: 157 GiB (PHerc0139), 32 GiB (PHerc0332) | Those are the figures for **r = 120**. At the **r = 170** footprint that every later budget uses: **PHerc0139 183.0 GiB** (93,716 level-2 chunks, about **90 h** of download at 0.58 MiB/s, i.e. 89.7 h); **PHerc0332 36.4 GiB** (18,639 chunks, about **18 h**, i.e. 17.85 h) | `results/estimate_0139_r170.json`, `estimate_0332_r170.json` (r = 120: `estimate_0139_r120.json` 156.78 GiB / 80,272 chunks; `estimate_0332_r120.json` 31.98 GiB / 16,376 chunks) |
| 2 | The 10-fiber G0 AFV is 0.36 MB | **147,456 bytes** (0.14 MiB) | `data/g0/g0_fibers.afv` |
| 3 | `patches/applied_pr2163_on_6f2024569.diff` is #2163 as applied | It **omits #2163's two new test files**, `fibers/tests/test_chunk_lock.py` and `fibers/tests/test_service_transport.py`. They were applied (they are in the worktree, untracked, and their content is identical to the PR's), but `git diff` does not list untracked files. The complete diff is now **`patches/applied_pr2163_on_6f2024569_with_tests.diff`** (564 lines, the same length as `pr2163_full.diff`; `git apply --check -R` passes on the worktree). The old file is kept as it was. | `patches/pr2163_full.diff`, `villa_6f2` worktree |
| 4 | "4 threads: 1.47 decisions/s" | That run used **16 torch threads with `--crop-threads 4`** (1.465 decisions/s wall, 1.473 per strip). The 8-thread run (batch 16) gave 1.334 / 1.341 decisions/s | `results/g0.json` `throughput.one_process_16_threads_4_crop_threads`, `one_process_8_threads_batch16` |
| 5 | RAM peak | **About 4.1 GB**: the G0 run's peak RSS is 4.073 GB (peak working set 4.392 GB); the cached 16-thread timing run peaked at 3.81 GB RSS (4.38 GB working set) | `results/g0.json` `run.peak_rss_gb`, `run.peak_wset_gb`, `timing_run` |

Also note: the throughput figures above were measured while another session used part of the CPU (`g0.json`
throughput note). An independent rerun on an idle machine measured 3.27 to 3.56 decisions/s, so PREREG v3 (4.7)
gives CPU time as a load-dependent range rather than a single rate.
