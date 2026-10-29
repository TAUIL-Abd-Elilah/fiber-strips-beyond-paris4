"""The frozen runs/v3/infer_cpu.py wrapper, unchanged, with at most 4 torch threads (PREREG v3.1 erratum checks 5 and
6 on Paris 4 run beside other work on this machine). Same checks: CUDA_VISIBLE_DEVICES must be -1, torch must see no
GPU, villa must come from the pinned worktree, BelowNormal priority.

    python -I infer_cpu4.py RECORD.json -- <infer.py arguments>
"""
import sys

sys.path.insert(0, 'C:/VesuviusTemp/strips/score/runs/v3')
import infer_cpu  # noqa: E402

infer_cpu.THREADS = 4

if __name__ == '__main__':
    assert sys.argv[2] == '--'
    infer_cpu.run(sys.argv[1], sys.argv[3:])
