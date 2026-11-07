"""The OG exporter writes an additional raw-modal waveform at the valid length."""

import csv
import hashlib
import json

import numpy as np
import soundfile as sf
import torch

from drumblender.baseline import VariableLengthDrumBlender
from drumblender.synths import ModalSynth
from scripts import export_reconstructions


def test_og_export_preserves_raw_modal_amplitude_and_common_metric_config(
    tmp_path, monkeypatch
):
    root = tmp_path / "cache"
    root.mkdir()
    wave = torch.full((1, 4097), 0.02)
    params = torch.tensor([[[0.1, 0.12]], [[0.003, 0.006]], [[0.7, 0.7]]])
    sf.write(root / "hit.wav", wave[0].numpy(), 48000, subtype="FLOAT")
    torch.save(params, root / "hit.pt")
    metadata = {"Pack/hit.wav": {
        "filename": "hit.wav",
        "orig_relpath": "Pack/hit.wav",
        "feature_file": "hit.pt",
        "sample_pack_key": "Pack",
        "num_samples": 4097,
        "split": "test",
        "audio_sha256": hashlib.sha256((root / "hit.wav").read_bytes()).hexdigest(),
    }}
    (root / "metadata.json").write_text(json.dumps(metadata))
    (root / "modal_config.json").write_text(
        json.dumps({"sample_rate": 48000, "num_modes": 1})
    )

    class Model(VariableLengthDrumBlender):
        def forward(self, original, params):
            return torch.full_like(original, 0.125)

    model = Model(modal_synth=ModalSynth(), loss_fn=torch.nn.L1Loss())
    monkeypatch.setattr(export_reconstructions, "load_model", lambda *args: (model, None))
    monkeypatch.setattr(export_reconstructions, "load_config_yaml", lambda *args: {
        "seed_everything": 20261106,
        "data": {"init_args": {
            "data_dir": str(root), "sample_rate": 48000, "expected_num_modes": 1
        }},
    })
    metrics = tmp_path / "metrics.yaml"
    metrics.write_text("class_path: torch.nn.ModuleDict\n")
    output = tmp_path / "export"
    export_reconstructions.main([
        "--config", "config.yaml", "--checkpoint", "model.ckpt",
        "--metrics-config", str(metrics), "--device", "cpu", "--output", str(output),
    ])
    expected = ModalSynth()(params[None], 4097).flatten().numpy()
    actual, rate = sf.read(output / "sinusoidal/Pack/hit.wav", dtype="float32")
    np.testing.assert_array_equal(actual, expected)
    assert sf.info(output / "sinusoidal/Pack/hit.wav").subtype == "FLOAT"
    full, _ = sf.read(output / "recon/Pack/hit.wav", dtype="float32")
    np.testing.assert_array_equal(full, np.full(4097, 0.125, dtype=np.float32))
    assert rate == 48000
    assert (output / "configs/evaluation_metrics.yaml").read_bytes() == metrics.read_bytes()
    with (output / "evaluation/per_file_loss.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["length"] == "4097"
    assert rows[0]["source_filename"] == "Pack/hit.wav"
    summary = json.loads((output / "evaluation/summary.json").read_text())
    assert summary["items"] == 1 and summary["split"] == "test"
