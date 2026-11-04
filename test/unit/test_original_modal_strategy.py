import json
import torch
import torchaudio


def test_original_strategy_invokes_upstream_analyzer_and_records_provenance(
    tmp_path, monkeypatch
):
    import scripts.extract_baseline_modal as extraction

    calls = []

    class Original:
        def __init__(self, **kwargs):
            self.cqt = self
            calls.append("original")

        def to(self, device):
            return self

        def __call__(self, waveform):
            return tuple(torch.ones(1, 1, 2) for _ in range(3))

    class Corrected(Original):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            calls.append("corrected")

    monkeypatch.setattr(extraction, "CQTModalAnalysis", Original, raising=False)
    monkeypatch.setattr(extraction, "BaselineModalAnalysis", Corrected)
    source = tmp_path / "audio"
    source.mkdir()
    torchaudio.save(str(source / "hit.wav"), torch.ones(1, 480), 48000)
    manifest = tmp_path / "split.json"
    manifest.write_text(json.dumps({"hit.wav": "train"}))
    out = tmp_path / "cache"
    extraction.extract(
        source, out, split_manifest=manifest, num_modes=2, analysis_method="original"
    )
    assert calls == ["original"]
    assert (
        json.loads((out / "modal_config.json").read_text())["analyzer"]
        == "original_upstream_cqt"
    )


def test_original_extraction_matches_direct_upstream_analysis(tmp_path):
    import scripts.extract_baseline_modal as extraction
    from drumblender.utils.modal_analysis import CQTModalAnalysis

    source = tmp_path / "source"
    source.mkdir()
    t = torch.arange(8193) / 48000
    waveform = (
        0.4 * torch.sin(2 * torch.pi * 440 * t)
        + 0.2 * torch.sin(2 * torch.pi * 880 * t)
    )[None]
    torchaudio.save(
        str(source / "hit.wav"), waveform, 48000, encoding="PCM_F", bits_per_sample=32
    )
    manifest = tmp_path / "splits.json"
    manifest.write_text(json.dumps({"hit.wav": "train"}))
    settings = dict(
        sample_rate=48000,
        hop_length=256,
        fmin=100,
        n_bins=48,
        bins_per_octave=12,
        min_length=1,
        num_modes=4,
        threshold=-80.0,
        diff_threshold=5.0,
    )
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    analyzer = CQTModalAnalysis(**settings, verbose=False)
    with torch.no_grad():
        f, a, phase = analyzer(waveform)
    expected = torch.stack([f[0] * (2 * torch.pi / 48000), a[0], phase[0]])
    expected = torch.nn.functional.pad(expected, (0, 0, 0, 4 - expected.shape[1]))
    out = tmp_path / "cache"
    extraction.extract(
        source, out, split_manifest=manifest, analysis_method="original", **settings
    )
    got = torch.load(out / "features/hit.pt", weights_only=True)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    torch.set_num_threads(previous_threads)


def test_extraction_needs_no_reference_or_split(tmp_path, monkeypatch):
    import scripts.extract_baseline_modal as extraction

    class Original:
        def __init__(self, **kwargs):
            self.cqt = self

        def to(self, device):
            return self

        def __call__(self, waveform):
            return tuple(torch.ones(1, 1, 2) for _ in range(3))

    monkeypatch.setattr(extraction, "CQTModalAnalysis", Original)
    source = tmp_path / "source"
    source.mkdir()
    torchaudio.save(str(source / "hit.wav"), torch.ones(1, 480), 48000)
    out = tmp_path / "cache"
    extraction.extract(source, out, num_modes=2, analysis_method="original")
    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["hit.wav"]["num_samples"] == 480
    assert "split" not in metadata["hit.wav"]
    assert not (out / "split_manifest.json").exists()
    assert (out / "features/hit.pt").is_file()
