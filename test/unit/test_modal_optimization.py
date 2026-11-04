import json

import numpy as np
import pytest
import torch
import torchaudio

from drumblender.baseline_analysis import BaselineModalAnalysis, fast_modal_tracking
from test.unit.modal_analysis_reference import ReferenceModalAnalysis


@pytest.mark.parametrize("seed", range(8))
def test_tracking_matches_frozen_reference_exactly(seed):
    rng = np.random.default_rng(seed)
    spec = rng.normal(size=(48, 51, 2)).astype(np.float32)
    spec[:, 10:15] = 0
    spec[:, 29] = 0
    spec[12, 29, 0] = 1
    reference = object.__new__(ReferenceModalAnalysis)
    reference.n_bins = 48
    reference.diff_threshold = 0.05
    want = reference.modal_tracking(spec, -60)
    got = fast_modal_tracking(spec, -60, 0.05)
    for a, b in zip(got, want):
        np.testing.assert_array_equal(a, np.asarray(b))
    actual = object.__new__(BaselineModalAnalysis)
    for a, b in zip(
        actual.create_modal_tensors(*got, 51, 3),
        reference.create_modal_tensors(*want, 51, 3),
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_parallel_cpu_cache_matches_serial(tmp_path):
    from scripts.extract_baseline_modal import extract

    source = tmp_path / "audio"
    source.mkdir()
    splits = {}
    for i, n in enumerate([800, 1025, 3001, 4099]):
        x = torch.sin(torch.arange(n) * 0.19)[None] * 0.2
        torchaudio.save(str(source / f"{i}.wav"), x, 48000)
        splits[f"{i}.wav"] = "train"
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps(splits))
    args = dict(
        processed_root=source,
        split_manifest=manifest,
        n_bins=48,
        fmin=100,
        bins_per_octave=12,
        num_modes=4,
        min_length=1,
    )
    extract(out_dir=tmp_path / "serial", workers=1, devices=["cpu"], **args)
    extract(out_dir=tmp_path / "parallel", workers=2, devices=["cpu"], **args)
    for i in range(4):
        a = torch.load(tmp_path / f"serial/features/{i}.pt", weights_only=True)
        b = torch.load(tmp_path / f"parallel/features/{i}.pt", weights_only=True)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert json.loads(
        (tmp_path / "serial/split_manifest.json").read_text()
    ) == json.loads((tmp_path / "parallel/split_manifest.json").read_text())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_cqt_agrees_with_cpu(monkeypatch):
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", True)
    options = dict(
        sample_rate=48000, n_bins=48, fmin=100, bins_per_octave=12, verbose=False
    )
    cpu = BaselineModalAnalysis(**options)
    gpu = BaselineModalAnalysis(**options)
    gpu.cqt.load_state_dict(cpu.cqt.state_dict())
    gpu.cqt.to("cuda:0")
    x = torch.sin(torch.arange(48000) * 0.19)[None]
    with torch.inference_mode():
        a = cpu.spectrogram(x)
        b = gpu.spectrogram(x.cuda()).cpu()
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices required")
def test_two_cuda_workers_match_cpu_cache(tmp_path):
    from scripts.extract_baseline_modal import extract

    source = tmp_path / "audio"
    source.mkdir()
    manifest = tmp_path / "splits.json"
    membership = {}
    for i, n in enumerate([800, 1025, 3001, 4099]):
        x = (torch.sin(torch.arange(n) * 0.19) * torch.exp(-torch.arange(n) / 2000))[
            None
        ] * 0.2
        torchaudio.save(str(source / f"{i}.wav"), x, 48000)
        membership[f"{i}.wav"] = "train"
    manifest.write_text(json.dumps(membership))
    options = dict(
        processed_root=source,
        split_manifest=manifest,
        n_bins=48,
        fmin=100,
        bins_per_octave=12,
        num_modes=4,
        min_length=1,
    )
    extract(out_dir=tmp_path / "cpu", devices=["cpu"], workers=1, **options)
    extract(
        out_dir=tmp_path / "gpu", devices=["cuda:0", "cuda:1"], workers=2, **options
    )
    for i in range(4):
        a = torch.load(tmp_path / f"cpu/features/{i}.pt", weights_only=True)
        b = torch.load(tmp_path / f"gpu/features/{i}.pt", weights_only=True)
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


def test_empty_tracking_retains_full_frame_axis():
    output = fast_modal_tracking(np.zeros((48, 17, 2), dtype=np.float32))
    assert all(v.shape == (0, 17) for v in output)
    analyzer = object.__new__(BaselineModalAnalysis)
    assert all(
        v.shape == (0, 17) for v in analyzer.create_modal_tensors(*output, 17, 10)
    )


@pytest.mark.parametrize(
    "devices,workers", [(["cpu"], 0), (["cpu", "cpu"], 1), (["cuda:99999"], 1)]
)
def test_invalid_execution_settings_do_not_create_cache(tmp_path, devices, workers):
    from scripts.extract_baseline_modal import extract

    source = tmp_path / "audio"
    source.mkdir()
    torchaudio.save(str(source / "x.wav"), torch.zeros(1, 480), 48000)
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"x.wav": "train"}))
    with pytest.raises(ValueError):
        extract(
            source,
            tmp_path / "cache",
            split_manifest=manifest,
            devices=devices,
            workers=workers,
        )
    assert not (tmp_path / "cache").exists()
