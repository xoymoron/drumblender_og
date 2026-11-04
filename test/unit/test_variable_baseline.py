"""Regression coverage for the variable-duration all-parallel baseline."""

import json
from pathlib import Path

import auraloss
import pytest
import torch
import torchaudio
import yaml


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def api():
    from drumblender import baseline

    return baseline


def model():
    from drumblender.models.soundstream import (
        SoundStreamAttentionEncoder,
        SoundStreamEncoder,
    )
    from drumblender.synths import ModalSynth, NoiseGenerator, TransientTCN

    return (
        api()
        .VariableLengthDrumBlender(
            modal_synth=ModalSynth(),
            loss_fn=api().PerItemMRSTFTLoss(),
            noise_autoencoder=SoundStreamEncoder(
                1, 2, 8, strides=(2, 2, 4, 8), transpose_output=True
            ),
            noise_autoencoder_accepts_audio=True,
            noise_synth=NoiseGenerator(window_size=256),
            transient_autoencoder=SoundStreamAttentionEncoder(
                1, 2, 8, strides=(2, 2, 4, 8)
            ),
            transient_autoencoder_accepts_audio=True,
            transient_synth=TransientTCN(
                hidden_channels=2,
                num_layers=2,
                kernel_size=3,
                film_conditioning=True,
                film_embedding_size=8,
            ),
            transient_parallel=True,
        )
        .eval()
    )


def item(samples, frames, key):
    p = torch.zeros(3, 2, frames)
    p[0] = 0.13
    p[1] = torch.exp(-torch.linspace(0, 7, frames))
    p[1, :, -2:] = 0
    return torch.sin(torch.arange(samples).float() * 0.13)[None], p, key


@pytest.mark.parametrize("samples", [129, 1001, 48000, 960000])
def test_real_all_parallel_returns_full_requested_length(samples):
    net = model()
    x, p, _ = item(samples, max(2, samples // 256 + 1), "a")
    with torch.no_grad():
        y = net(x[None], p[None])
    assert y.shape == (1, 1, samples)
    assert torch.isfinite(y).all()


def test_padding_does_not_change_eval_reconstruction_for_same_noise_draws():
    items = [item(1001, 5, "a"), item(4099, 17, "b")]
    net = model()
    with torch.no_grad():
        batch = api().collate_baseline(items)
        torch.manual_seed(7)
        y = net(batch[0], batch[1], batch[2], batch[3])
        torch.manual_seed(7)
        for i, (x, p, _) in enumerate(items):
            single = net(x[None], p[None])
            torch.testing.assert_close(y[i : i + 1, :, : x.shape[-1]], single)


def test_full_length_training_has_finite_gradients():
    net = model().train()
    batch = api().collate_baseline([item(1001, 5, "a"), item(4099, 17, "b")])
    loss, _ = net._do_step(batch)
    loss.backward()
    gradients = [p.grad for p in net.parameters() if p.requires_grad]
    assert gradients and all(
        g is not None and torch.isfinite(g).all() for g in gradients
    )


def test_spectral_loss_is_padding_and_batch_grouping_invariant():
    native = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[64, 128], hop_sizes=[16, 32], win_lengths=[64, 128]
    )
    objective = api().PerItemMRSTFTLoss(native)
    torch.manual_seed(7)
    target = torch.randn(2, 1, 513)
    pred = (target * 0.7).requires_grad_()
    want = (
        native(pred[:1, :, :191], target[:1, :, :191]) + native(pred[1:], target[1:])
    ) / 2
    got = objective(pred, target, [191, 513])
    torch.testing.assert_close(got, want)
    torch.testing.assert_close(
        objective(
            torch.nn.functional.pad(pred, (0, 1024)),
            torch.nn.functional.pad(target, (0, 1024)),
            [191, 513],
        ),
        want,
    )
    got.backward()
    assert torch.count_nonzero(pred.grad[0, :, 191:]) == 0


def test_bucketing_keeps_all_items_and_shuffles_each_epoch():
    sampler = api().BucketingBatchSampler([100 + i * 31 for i in range(29)], 4, seed=8)
    first = list(sampler)
    assert sorted(i for b in first for i in b) == list(range(29))
    assert all(1 <= len(b) <= 4 for b in first)
    sampler.set_epoch(1)
    assert first != list(sampler)
    sampler.set_epoch(0)
    assert first == list(sampler)


def test_dataset_uses_frozen_splits_and_unmodified_waveforms(tmp_path, monkeypatch):
    for index, n in enumerate([1001, 2001, 3001]):
        torchaudio.save(
            str(tmp_path / f"{index}.wav"),
            torch.ones(1, n) * 0.25,
            48000,
            encoding="PCM_F",
            bits_per_sample=32,
        )
        torch.save(torch.zeros(3, 1, n // 256 + 1), tmp_path / f"{index}.pt")
    metadata = {
        str(i): {
            "filename": f"{i}.wav",
            "feature_file": f"{i}.pt",
            "num_samples": n,
            "split": split,
            "sample_pack_key": "p",
        }
        for i, (n, split) in enumerate(
            zip([1001, 2001, 3001], ["train", "val", "test"])
        )
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "modal_config.json").write_text(
        json.dumps({"sample_rate": 48000, "num_modes": 2})
    )
    ds = api().BaselineDataset(tmp_path, split="train", expected_num_modes=2)
    original_load = torchaudio.load

    def require_string_path(path, *args, **kwargs):
        assert isinstance(path, str)
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(torchaudio, "load", require_string_path)
    x, p, key = ds[0]
    assert x.shape == (1, 1001) and key == "0"
    torch.testing.assert_close(x, torch.full_like(x, 0.25))
    assert p.shape == (3, 2, 4)
    with pytest.raises(ValueError, match="sample rate"):
        api().BaselineDataset(tmp_path, sample_rate=44100)


def test_corrected_tracking_keeps_silent_frames_and_separate_peaks():
    from drumblender.baseline_analysis import BaselineModalAnalysis

    analyzer = object.__new__(BaselineModalAnalysis)
    analyzer.n_bins = 9
    analyzer.diff_threshold = 0.05
    spec = torch.zeros(9, 5, 2).numpy()
    spec[2, [0, 2, 4], 0] = 1
    spec[6, [0, 2, 4], 0] = 0.5
    f, a, p = analyzer.modal_tracking(spec, threshold=-60)
    f, a, p = analyzer.create_modal_tensors(f, a, p, num_hops=5, min_length=1)
    assert a.shape == (2, 5)
    assert torch.count_nonzero(a[:, 1]) == 0 and torch.count_nonzero(a[:, 3]) == 0
    torch.testing.assert_close(
        a[:, [0, 2, 4]], torch.tensor([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]])
    )


def test_extraction_uses_preprocessed_files_without_name_collisions(tmp_path):
    from scripts.extract_baseline_modal import extract

    source = tmp_path / "processed"
    for pack in ["a", "b"]:
        (source / pack).mkdir(parents=True)
        torchaudio.save(str(source / pack / "hit.wav"), torch.zeros(1, 480), 48000)
    out = tmp_path / "cache"
    manifest = tmp_path / "splits.json"
    manifest.write_text(json.dumps({"a/hit.wav": "train", "b/hit.wav": "test"}))
    extract(
        source,
        out,
        num_modes=2,
        seed=7,
        n_bins=48,
        fmin=100,
        bins_per_octave=12,
        split_manifest=manifest,
    )
    metadata = json.loads((out / "metadata.json").read_text())
    assert len(metadata) == 2
    assert len({m["feature_file"] for m in metadata.values()}) == 2
    assert all(m["num_samples"] == 480 for m in metadata.values())
    for m in metadata.values():
        assert torch.load(out / m["feature_file"], weights_only=True).shape == (3, 2, 2)
    with pytest.raises(FileExistsError):
        extract(source, out)

    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "z": {
                    "orig_relpath": "a/hit.wav",
                    "sample_pack_key": "source/nested/a",
                },
                "a": {
                    "orig_relpath": "b/hit.wav",
                    "sample_pack_key": "source/nested/b",
                },
            }
        )
    )
    reference_cache = tmp_path / "reference_cache"
    extract(
        source,
        reference_cache,
        num_modes=2,
        n_bins=48,
        fmin=100,
        bins_per_octave=12,
        reference_metadata=reference,
    )
    rows = json.loads((reference_cache / "metadata.json").read_text())
    assert rows["a/hit.wav"]["sample_pack_key"] == "source/nested/a"
    assert rows["b/hit.wav"]["sample_pack_key"] == "source/nested/b"


def test_adaptation_keeps_original_all_parallel_parameterization():
    from jsonargparse import ArgumentParser

    from drumblender.tasks import DrumBlender

    root = Path(__file__).resolve().parents[2]
    parser = ArgumentParser()
    parser.add_argument("--model", type=DrumBlender, enable_path=True)
    original = yaml.safe_load((root / "cfg/05_all_parallel.yaml").read_text())["model"]
    # Resolve config fragments exactly as the experiment does.
    for key, value in list(original["init_args"].items()):
        path = root / "cfg" / str(value)
        if isinstance(value, str) and path.is_file():
            original["init_args"][key] = yaml.safe_load(path.read_text())
    native = parser.instantiate_classes(parser.parse_object({"model": original})).model
    original["class_path"] = "drumblender.baseline.VariableLengthDrumBlender"
    adapted = parser.instantiate_classes(parser.parse_object({"model": original})).model
    assert {k: tuple(v.shape) for k, v in native.state_dict().items()} == {
        k: tuple(v.shape) for k, v in adapted.state_dict().items()
    }


def test_split_uses_reference_ids_and_pack_order(tmp_path):
    from scripts.extract_baseline_modal import reference_splits

    # Reference keys deliberately sort differently from waveform filenames.
    rows = {
        str(i): {"orig_relpath": f"pack/{19-i}.wav", "sample_pack_key": "pack"}
        for i in range(20)
    }
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(rows))
    membership = reference_splits(path, seed=20261106)
    generator = torch.Generator().manual_seed(20261106)
    keys = sorted(rows)
    order = torch.randperm(len(keys), generator=generator).tolist()
    for rank, index in enumerate(order):
        split = "train" if rank < 16 else "val" if rank < 18 else "test"
        assert membership[rows[keys[index]]["orig_relpath"]] == split


def test_baseline_config_matches_reference_training_protocol():
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "cfg/05_all_parallel_variable.yaml").read_text())
    assert config["seed_everything"] == 20261106
    assert config["optimizer"]["class_path"] == "torch.optim.Adam"
    assert config["optimizer"]["init_args"]["lr"] == 0.0001
    scheduler = config["lr_scheduler"]["init_args"]
    assert scheduler == {
        "monitor": "validation/loss",
        "factor": 0.5,
        "patience": 10,
        "min_lr": 0.00000625,
    }
    assert config["trainer"]["accumulate_grad_batches"] == 8
    assert config["trainer"]["max_epochs"] == 150
    early = config["trainer"]["callbacks"][0]["init_args"]
    assert early["patience"] == 20 and early["check_on_train_epoch_end"]
    assert config["data"]["init_args"]["batch_size"] == 2
    assert config["data"]["init_args"]["seed"] == 20261106


def test_plain_loader_epoch_hook_does_not_require_custom_sampler():
    from types import SimpleNamespace

    from torch.utils.data import BatchSampler, SequentialSampler

    net = model()
    net._trainer = SimpleNamespace(
        current_epoch=0,
        train_dataloader=SimpleNamespace(
            batch_sampler=BatchSampler(SequentialSampler(range(3)), 2, False)
        ),
    )
    net.on_train_epoch_start()


def test_equal_length_training_preserves_upstream_batch_normalization():
    import copy

    from drumblender.tasks import DrumBlender

    native = model().train()
    adapted = copy.deepcopy(native)
    items = [item(1001, 5, "a"), item(1001, 5, "b")]
    items[1] = (items[1][0] * 3, items[1][1] * 0.5, "b")
    x, p, lengths, frames = api().collate_baseline(items)
    torch.manual_seed(19)
    want = DrumBlender.forward(native, x, p)
    torch.manual_seed(19)
    got = adapted(x, p, lengths, frames)
    torch.testing.assert_close(got, want)


def test_twenty_second_modal_phase_does_not_lose_float32_precision():
    from drumblender.synths.modal import modal_synth

    count = 960000
    frequency = torch.tensor([[[2 * torch.pi * 12300 / 48000]]])
    amplitude = torch.ones_like(frequency)
    with torch.no_grad():
        got = modal_synth(frequency, amplitude, count)
    phase = torch.arange(1, count + 1, dtype=torch.float64) * frequency.double().item()
    want = phase.sin().float()[None]
    relative_error = torch.linalg.vector_norm(got - want) / torch.linalg.vector_norm(
        want
    )
    assert relative_error < 1e-6


def test_training_config_runs_with_standard_lightning_optimizer_and_accumulation(
    tmp_path,
):
    from pytorch_lightning.cli import LightningCLI

    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "cfg/05_all_parallel_variable.yaml").read_text())
    for key, value in list(config["model"]["init_args"].items()):
        if isinstance(value, str) and (root / "cfg" / value).is_file():
            config["model"]["init_args"][key] = yaml.safe_load(
                (root / "cfg" / value).read_text()
            )
    metadata = {}
    for index in range(7):
        size = 1025 + index * 128
        x = torch.sin(torch.arange(size).float() * 0.13)[None] * 0.1
        torchaudio.save(str(tmp_path / f"{index}.wav"), x, 48000)
        _, params, _ = item(size, size // 256 + 1, str(index))
        params = torch.nn.functional.pad(params, (0, 0, 0, 64 - params.shape[1]))
        torch.save(params, tmp_path / f"{index}.pt")
        metadata[str(index)] = {
            "filename": f"{index}.wav",
            "feature_file": f"{index}.pt",
            "num_samples": size,
            "split": "train" if index < 5 else "val" if index == 5 else "test",
        }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    (tmp_path / "modal_config.json").write_text(
        json.dumps({"sample_rate": 48000, "num_modes": 64})
    )
    (tmp_path / "split_manifest.json").write_text(
        json.dumps({k: v["split"] for k, v in metadata.items()})
    )
    config["data"]["init_args"].update(
        data_dir=str(tmp_path), batch_size=2, num_workers=0, use_bucketing=False
    )
    config["trainer"].update(
        accelerator="cpu",
        devices=1,
        strategy="auto",
        max_epochs=2,
        accumulate_grad_batches=2,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        logger={
            "class_path": "pytorch_lightning.loggers.CSVLogger",
            "init_args": {"save_dir": str(tmp_path / "logs")},
        },
        default_root_dir=str(tmp_path),
    )
    config["trainer"]["callbacks"][1]["init_args"]["dirpath"] = str(
        tmp_path / "checkpoints"
    )
    cli = LightningCLI(args=config, run=False, save_config_callback=None)
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    assert cli.trainer.global_step == 4
    assert torch.isfinite(cli.trainer.callback_metrics["validation/loss"])
    assert isinstance(cli.trainer.optimizers[0], torch.optim.Adam)
    scheduler = cli.trainer.lr_scheduler_configs[0].scheduler
    assert scheduler.patience == 10 and scheduler.min_lrs == [0.00000625]
    assert list((tmp_path / "checkpoints").glob("*.ckpt"))
