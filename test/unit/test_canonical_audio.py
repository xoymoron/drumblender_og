import hashlib
import json

import pytest
import soundfile as sf
import torch
import torchaudio

from drumblender.baseline import BaselineDataModule, BaselineDataset
from drumblender.cli import _infer_cached_baseline
from scripts import extract_baseline_modal


def test_canonical_cache_preserves_split_and_rejects_audio_drift(tmp_path, monkeypatch):
    audio = tmp_path / "audio"
    audio.mkdir()
    wave = audio / "hit.wav"
    samples = torch.tensor([0.001234567, -0.00008765]).repeat(240)
    sf.write(wave, samples.numpy(), 48000, subtype="FLOAT")
    reference = tmp_path / "reference.json"
    row = {
        "orig_relpath": "hit.wav",
        "sample_pack_key": "test pack",
        "split": "test",
        "num_samples": samples.numel(),
        "audio_sha256": hashlib.sha256(wave.read_bytes()).hexdigest(),
    }
    reference.write_text(json.dumps({"hit.wav": row}), encoding="utf-8")

    class Analyzer:
        def __init__(self, **kwargs):
            self.cqt = self

        def to(self, device):
            return self

        def __call__(self, waveform):
            torch.testing.assert_close(waveform[0], samples, rtol=0, atol=0)
            return tuple(torch.ones(1, 1, 2) for _ in range(3))

    calls = []
    original_load = torchaudio.load

    def load(path, *, backend=None):
        calls.append(backend)
        return original_load(path, backend="soundfile")

    monkeypatch.setattr(torchaudio, "load", load)
    monkeypatch.setattr(extract_baseline_modal, "CQTModalAnalysis", Analyzer)
    cache = tmp_path / "cache"
    extract_baseline_modal.extract(
        audio, cache, canonical_audio_metadata=reference, analysis_method="original"
    )
    assert calls == ["sox"]
    ds = BaselineDataset(cache, split="test")
    x, _, key = ds[0]
    assert calls == ["sox", "sox"]
    assert key == "hit.wav"
    torch.testing.assert_close(x[0], samples, rtol=0, atol=0)
    assert ds.metadata[key]["sample_pack_key"] == "test pack"

    class Identity(torch.nn.Module):
        device = torch.device("cpu")

        def forward(self, audio, params):
            assert not self.training and torch.is_inference_mode_enabled()
            return audio

    output = tmp_path / "recon.wav"
    _infer_cached_baseline(
        Identity(),
        {"data_dir": cache, "sample_rate": 48000, "expected_num_modes": 64},
        wave,
        output,
    )
    assert sf.info(output).subtype == "FLOAT"
    reconstructed, _ = sf.read(output, dtype="float32")
    torch.testing.assert_close(torch.from_numpy(reconstructed), samples, rtol=0, atol=0)

    dm = BaselineDataModule(str(cache))
    dm.prepare_data()
    sf.write(wave, samples.numpy(), 48000, subtype="PCM_16")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        dm.prepare_data()
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        extract_baseline_modal.extract(audio, tmp_path / "other", canonical_audio_metadata=reference)
    assert not (tmp_path / "other").exists()
