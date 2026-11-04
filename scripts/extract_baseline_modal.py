"""Extract full-clip modal features from existing preprocessed mono WAVs."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

# Direct script execution uses this checkout, even if another fork is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drumblender.baseline_analysis import BaselineModalAnalysis


def reference_splits(metadata_file, seed=20261106, train_ratio=0.8, val_ratio=0.1):
    """Reuse the reference dataset's sorted IDs and within-pack RNG sequence."""
    rows = json.loads(Path(metadata_file).read_text(encoding="utf-8"))
    if train_ratio < 0 or val_ratio < 0 or train_ratio + val_ratio > 1:
        raise ValueError("Invalid split ratios")
    packs = {}
    for key in sorted(rows):
        pack = (
            str(rows[key].get("sample_pack_key") or "__root__")
            .replace("\\", "/")
            .strip("/")
        )
        packs.setdefault(pack, []).append(key)
    generator = torch.Generator().manual_seed(seed)
    membership = {}
    for pack in sorted(packs):
        keys = packs[pack]
        order = torch.randperm(len(keys), generator=generator).tolist()
        train = max(1, int(len(keys) * train_ratio)) if train_ratio > 0 else 0
        val = min(int(len(keys) * val_ratio), len(keys) - train)
        for rank, index in enumerate(order):
            relative = rows[keys[index]]["orig_relpath"].replace("\\", "/")
            if relative in membership:
                raise ValueError("Reference metadata contains duplicate waveform paths")
            membership[relative] = (
                "train" if rank < train else "val" if rank < train + val else "test"
            )
    return membership


def extract(
    processed_root,
    out_dir,
    sample_rate=48000,
    num_modes=64,
    seed=20261106,
    hop_length=256,
    fmin=20,
    n_bins=240,
    bins_per_octave=24,
    min_length=10,
    threshold=-80.0,
    diff_threshold=5.0,
    split_manifest=None,
    reference_metadata=None,
    train_ratio=0.8,
    val_ratio=0.1,
):
    root, out = Path(processed_root).resolve(), Path(out_dir).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite cache: {out}")
    if sample_rate < 1 or num_modes < 1:
        raise ValueError("Sample rate and mode count must be positive")
    if root == out or root in out.parents or out in root.parents:
        raise ValueError("Audio and cache directories must be separate")
    wavs = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".wav"
    )
    if not wavs:
        raise ValueError("No preprocessed WAVs found")
    if (split_manifest is None) == (reference_metadata is None):
        raise ValueError(
            "Provide exactly one reference metadata or frozen split manifest"
        )
    manifest = (
        json.loads(Path(split_manifest).read_text(encoding="utf-8"))
        if split_manifest
        else reference_splits(reference_metadata, seed, train_ratio, val_ratio)
    )
    if set(manifest) != {p.relative_to(root).as_posix() for p in wavs}:
        raise ValueError("Split manifest membership must exactly match input WAVs")
    if any(v not in ("train", "val", "test") for v in manifest.values()):
        raise ValueError("Unknown split assignment")
    reference_rows = (
        json.loads(Path(reference_metadata).read_text(encoding="utf-8"))
        if reference_metadata
        else {}
    )
    pack_by_path = {
        row["orig_relpath"].replace("\\", "/"): row.get("sample_pack_key") or "__root__"
        for row in reference_rows.values()
    }
    analyzer = BaselineModalAnalysis(
        sample_rate,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        min_length=min_length,
        num_modes=num_modes,
        threshold=threshold,
        diff_threshold=diff_threshold,
        verbose=False,
    )
    metadata = {}
    out.mkdir(parents=True)
    for path in wavs:
        waveform, rate = torchaudio.load(path)
        if (
            rate != sample_rate
            or waveform.shape[0] != 1
            or waveform.shape[-1] < 1
            or not torch.isfinite(waveform).all()
        ):
            raise ValueError(
                f"Expected preprocessed finite mono {sample_rate} Hz: {path}"
            )
        relative = path.relative_to(root).as_posix()
        with torch.no_grad():
            try:
                frequency, amplitude, phase = analyzer(waveform)
            except RuntimeError as exc:
                if "padding" not in str(exc).lower():
                    raise
                previous = analyzer.cqt.pad_mode
                try:
                    analyzer.cqt.pad_mode = "constant"
                    frequency, amplitude, phase = analyzer(waveform)
                finally:
                    analyzer.cqt.pad_mode = previous
        parameters = torch.stack(
            [frequency[0] * (2 * torch.pi / sample_rate), amplitude[0], phase[0]]
        )
        parameters = F.pad(parameters, (0, 0, 0, num_modes - parameters.shape[1]))
        feature = Path("features") / Path(relative).with_suffix(".pt")
        (out / feature).parent.mkdir(parents=True, exist_ok=True)
        torch.save(parameters, out / feature)
        metadata[relative] = {
            "filename": str(path),
            "orig_relpath": relative,
            "feature_file": feature.as_posix(),
            "num_samples": waveform.shape[-1],
            "sample_pack_key": pack_by_path.get(relative, "__root__"),
            "split": manifest[relative],
            "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    config = dict(
        sample_rate=sample_rate,
        num_modes=num_modes,
        seed=seed,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        min_length=min_length,
        threshold=threshold,
        diff_threshold=diff_threshold,
        pad_mode="reflect; constant for short inputs",
        analyzer="corrected_upstream_cqt",
        frequency_units="radians/sample",
        processed_root=str(root),
        waveform_processing="none",
    )
    for name, content in [
        ("metadata.json", metadata),
        ("modal_config.json", config),
        ("split_manifest.json", manifest),
    ]:
        (out / name).write_text(json.dumps(content, indent=2), encoding="utf-8")
    print(f"Extracted {len(metadata)} complete waveforms into {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--out-dir", required=True)
    splits = parser.add_mutually_exclusive_group(required=True)
    splits.add_argument("--split-manifest")
    splits.add_argument("--reference-metadata")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    for name, default in [
        ("sample_rate", 48000),
        ("num_modes", 64),
        ("seed", 20261106),
        ("hop_length", 256),
        ("fmin", 20),
        ("n_bins", 240),
        ("bins_per_octave", 24),
        ("min_length", 10),
    ]:
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    parser.add_argument("--threshold", type=float, default=-80.0)
    parser.add_argument("--diff-threshold", type=float, default=5.0)
    extract(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
