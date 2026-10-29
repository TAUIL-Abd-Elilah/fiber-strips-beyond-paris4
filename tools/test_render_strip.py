"""Tests of tools/render_strip.py against a stub villa tree (no torch, no CT, no GPU): the CUDA guard, the missing-input
check, and that the flags make_figures.render_block prints (--villa, --out, --checkpoint, --ct, --cache-dir, --device)
parse, import villa from --villa, and bind the normalization before the CT volume is opened, as StripPredictor does
for inference.

    CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 python -m pytest -q tools/test_render_strip.py
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parent/'render_strip.py'

INFER_STUB = '''
CALLS = []
def add_checkpoint_arguments(ap):  # infer.add_checkpoint_arguments + runtime.add_volume_arguments (flags only)
    ap.add_argument('--checkpoint', default='scrollprize/fiber_strip_small_v1')
    for flag in ('--ct', '--ct-level', '--ct-grid-scale', '--grid-scale', '--cache-dir', '--frame-checkpoint',
                 '--umbilicus'):
        ap.add_argument(flag)
    ap.add_argument('--device', default='cuda')
def resolve_checkpoint(c):
    CALLS.append(('resolve_checkpoint', c))
    return 'ckpt.pt'
def configure_volume(args, spec, cfg, path):
    CALLS.append(('configure_volume', args.ct, args.cache_dir, path))
def bind_normalization(spec):
    CALLS.append(('bind_normalization',))
def render_saved(out, vol):
    CALLS.append(('render_saved', str(out), vol.opened_after_bind))
'''
VOLUME_STUB = '''
class CTVolumeSpec:
    @classmethod
    def from_dict(cls, d):
        return cls()
class CTVolume:
    def __init__(self, spec):
        from vesuvius.neural_tracing.tracers.fibers.sheet_strip import infer
        infer.CALLS.append(('CTVolume',))
        self.opened_after_bind = ('bind_normalization',) in infer.CALLS
'''


def fake_villa(root):
    src = root/'vesuvius'/'src'
    for sub, name, body in ((['fibers', 'sheet_strip'], 'infer.py', INFER_STUB),
                            (['shared', 'crop_sampling'], 'volume.py', VOLUME_STUB)):
        d = src
        for p in ['vesuvius', 'neural_tracing', 'tracers', *sub]:
            d = d/p
            d.mkdir(parents=True, exist_ok=True)
            (d/'__init__.py').touch()
        (d/name).write_text(body)
    return root


def piece(root):
    out = root/'P0001'
    (out/'strip').mkdir(parents=True)
    for name in ('summary.json', 'strip.npz', 'strip/x.tif', 'strip/y.tif', 'strip/z.tif'):
        (out/name).write_bytes(b'')
    return out


@pytest.fixture
def tool(monkeypatch):
    """render_strip.py with a stub torch; any real vesuvius modules are set aside and restored afterwards."""
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace(load=lambda *a, **k: {'vol_spec': {}},
                                                                    set_num_threads=lambda n: None))
    monkeypatch.setattr(sys, 'path', list(sys.path))
    is_villa = lambda k: k == 'vesuvius' or k.startswith('vesuvius.')
    saved = {k: sys.modules.pop(k) for k in [k for k in sys.modules if is_villa(k)]}
    spec = importlib.util.spec_from_file_location('render_strip_under_test', TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    yield mod
    for k in [k for k in sys.modules if is_villa(k)]:
        del sys.modules[k]
    sys.modules.update(saved)


def test_refuses_unless_cuda_is_hidden(tool, tmp_path, monkeypatch):
    for value in (None, '', '0'):
        if value is None:
            monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
        else:
            monkeypatch.setenv('CUDA_VISIBLE_DEVICES', value)
        with pytest.raises(SystemExit, match='CUDA_VISIBLE_DEVICES must be -1'):
            tool.main(['--villa', str(fake_villa(tmp_path/'v')), '--out', str(piece(tmp_path/str(value)))])


def test_the_printed_flags_bind_the_normalization_before_the_ct_is_opened(tool, tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    villa, out = fake_villa(tmp_path/'v'), piece(tmp_path)
    tool.main(['--villa', str(villa), '--out', str(out), '--checkpoint', 'scrollprize/x@abc',
               '--ct', 's3://bucket/vol.zarr', '--cache-dir', str(tmp_path/'cache'), '--device', 'cpu'])
    infer = sys.modules['vesuvius.neural_tracing.tracers.fibers.sheet_strip.infer']
    assert Path(infer.__file__).resolve().is_relative_to((villa/'vesuvius'/'src').resolve())
    assert [c[0] for c in infer.CALLS] == ['resolve_checkpoint', 'configure_volume', 'bind_normalization',
                                          'CTVolume', 'render_saved']
    assert infer.CALLS[1][1:] == ('s3://bucket/vol.zarr', str(tmp_path/'cache'), 'ckpt.pt')
    assert infer.CALLS[-1] == ('render_saved', str(out), True)


def test_a_missing_input_stops_before_villa_is_called(tool, tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    out = piece(tmp_path)
    (out/'strip.npz').unlink()
    with pytest.raises(SystemExit, match='strip.npz is missing'):
        tool.main(['--villa', str(fake_villa(tmp_path/'v')), '--out', str(out)])
    infer = sys.modules['vesuvius.neural_tracing.tracers.fibers.sheet_strip.infer']
    assert infer.CALLS == []
