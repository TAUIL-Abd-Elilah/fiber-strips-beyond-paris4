"""pytest hook for tools/test_fibers_json_to_afv.py (added for the public repo; the test file itself is unchanged).

That test module needs two things this repo does not ship:
  - the pinned villa (6f2024569 + patches) in villa_6f2/ at the repo root, or on PYTHONPATH: the converter imports
    its top-level ``vc3d_fiber_format`` package and ``vesuvius``'s AFV code (README, How to reproduce, step 3);
  - the 10 sample fiber JSONs in data/samples/0139 and data/samples/0332. They are raw human fiber annotations,
    which are not redistributed (python fetch_inputs.py samples).
Without them the module is reported as skipped, with the reason, instead of failing to import.
"""
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SAMPLES = REPO/'data'/'samples'
VILLA_SRC = REPO/'villa_6f2'/'vesuvius'/'src'
_REASONS = {}


def _missing():
    have_villa = (VILLA_SRC/'vc3d_fiber_format').is_dir() or importlib.util.find_spec('vc3d_fiber_format') is not None
    if not have_villa:
        return ('needs the pinned villa (villa_6f2/ at the repo root or on PYTHONPATH; '
                'it provides vc3d_fiber_format and the AFV code)')
    if not all(len(list((SAMPLES/s).glob('*.json'))) == 5 for s in ('0139', '0332')):
        return ('needs data/samples/{0139,0332} (raw human fiber JSONs, not redistributed; '
                'python fetch_inputs.py samples)')
    return None


class _SkippedModule(pytest.Module):
    def collect(self):
        pytest.skip(_REASONS[self.path], allow_module_level=True)


def pytest_pycollect_makemodule(module_path, parent):
    if Path(module_path).name != 'test_fibers_json_to_afv.py':
        return None
    reason = _missing()
    if reason is None:
        return None
    _REASONS[Path(module_path)] = reason
    return _SkippedModule.from_parent(parent, path=Path(module_path))
