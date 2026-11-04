"""Extract full-clip modal features from existing preprocessed mono WAVs."""

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

# Direct script execution uses this checkout, even if another fork is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drumblender.baseline_analysis import BaselineModalAnalysis
from drumblender.utils.modal_analysis import CQTModalAnalysis


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


_ANALYZER = None
_DEVICE = "cpu"
_OPTIONS = {}


def _initialize_worker(options, devices, threads):
    global _ANALYZER, _DEVICE, _OPTIONS
    _OPTIONS = options
    _DEVICE = devices.get() if devices is not None else options["device"]
    torch.set_num_threads(threads)
    if _DEVICE.startswith("cuda"):
        torch.cuda.set_device(_DEVICE)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    analyzer_class = (
        CQTModalAnalysis
        if options["analysis_method"] == "original"
        else BaselineModalAnalysis
    )
    _ANALYZER = analyzer_class(**options["analysis"], verbose=False)
    _ANALYZER.cqt.to(_DEVICE)


def _extract_file(path):
    analyzer = _ANALYZER
    root, out = Path(_OPTIONS["root"]), Path(_OPTIONS["out"])
    sample_rate, num_modes = _OPTIONS["sample_rate"], _OPTIONS["num_modes"]
    manifest, pack_by_path = _OPTIONS["manifest"], _OPTIONS["pack_by_path"]
    path = Path(path)
    item_started = time.perf_counter()
    waveform, rate = torchaudio.load(str(path))
    if (
        rate != sample_rate
        or waveform.shape[0] != 1
        or waveform.shape[-1] < 1
        or not torch.isfinite(waveform).all()
    ):
        raise ValueError(f"Expected preprocessed finite mono {sample_rate} Hz: {path}")
    relative = path.relative_to(root).as_posix()
    with torch.no_grad():
        try:
            frequency, amplitude, phase = analyzer(waveform.to(_DEVICE))
        except RuntimeError as exc:
            if "padding" not in str(exc).lower():
                raise
            previous = analyzer.cqt.pad_mode
            try:
                analyzer.cqt.pad_mode = "constant"
                frequency, amplitude, phase = analyzer(waveform.to(_DEVICE))
            finally:
                analyzer.cqt.pad_mode = previous
    track_count = int(frequency.shape[1])
    active_track_count = int(torch.count_nonzero(amplitude[0].abs().sum(dim=-1)))
    parameters = torch.stack(
        [frequency[0] * (2 * torch.pi / sample_rate), amplitude[0], phase[0]]
    )
    parameters = F.pad(parameters, (0, 0, 0, num_modes - parameters.shape[1]))
    feature = Path("features") / Path(relative).with_suffix(".pt")
    (out / feature).parent.mkdir(parents=True, exist_ok=True)
    torch.save(parameters, out / feature)
    item_seconds = time.perf_counter() - item_started
    item_stats = {
        "path": relative,
        "seconds": item_seconds,
        "samples": int(waveform.shape[-1]),
        "tracks": track_count,
        "active_tracks": active_track_count,
    }
    row = {
        "num_tracks": track_count,
        "num_active_tracks": active_track_count,
        "extraction_seconds": item_seconds,
        "filename": str(path),
        "orig_relpath": relative,
        "feature_file": feature.as_posix(),
        "num_samples": waveform.shape[-1],
        "sample_pack_key": pack_by_path.get(relative, "__root__"),
        "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if manifest is not None:
        row["split"] = manifest[relative]
    return relative, row, item_stats


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
    workers=1,
    devices=None,
    threads_per_worker=1,
    analysis_method="corrected",
):
    if analysis_method not in ("original", "corrected"):
        raise ValueError("Expected original or corrected analysis method")
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
    if split_manifest is not None and reference_metadata is not None:
        raise ValueError("Provide at most one reference metadata or split manifest")
    manifest = (
        json.loads(Path(split_manifest).read_text(encoding="utf-8"))
        if split_manifest
        else (
            reference_splits(reference_metadata, seed, train_ratio, val_ratio)
            if reference_metadata
            else None
        )
    )
    if manifest is not None and set(manifest) != {
        p.relative_to(root).as_posix() for p in wavs
    }:
        raise ValueError("Split manifest membership must exactly match input WAVs")
    if manifest is not None and any(
        v not in ("train", "val", "test") for v in manifest.values()
    ):
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
    devices = list(devices or ["cpu"])
    if workers < 1 or threads_per_worker < 1 or workers < len(devices):
        raise ValueError(
            "Need positive workers/threads and at least one worker per device"
        )
    for device in devices:
        parsed = torch.device(device)
        if parsed.type not in ("cpu", "cuda"):
            raise ValueError("Only CPU and CUDA extraction are supported")
        if parsed.type == "cuda" and (
            parsed.index is None
            or not torch.cuda.is_available()
            or parsed.index >= torch.cuda.device_count()
        ):
            raise ValueError(f"CUDA device is unavailable: {device}")
    features = [p.relative_to(root).with_suffix(".pt").as_posix() for p in wavs]
    if len(features) != len(set(features)):
        raise ValueError(
            "Input filenames collide after replacing their extension with .pt"
        )
    options = dict(
        root=str(root),
        out=str(out),
        sample_rate=sample_rate,
        num_modes=num_modes,
        manifest=manifest,
        pack_by_path=pack_by_path,
        device=devices[0],
        analysis_method=analysis_method,
        analysis=dict(
            sample_rate=sample_rate,
            hop_length=hop_length,
            fmin=fmin,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            min_length=min_length,
            num_modes=num_modes,
            threshold=threshold,
            diff_threshold=diff_threshold,
        ),
    )
    metadata, stats = {}, []
    out.mkdir(parents=True)
    started = time.perf_counter()
    # Long files first reduce the final straggler without changing any analysis.
    jobs = sorted(wavs, key=lambda p: (-p.stat().st_size, p.as_posix()))
    previous_threads = torch.get_num_threads()
    try:
        if workers == 1:
            _initialize_worker(options, None, threads_per_worker)
            results = map(_extract_file, jobs)
            for relative, row, item_stats in tqdm(
                results, total=len(jobs), desc="Modal extraction", unit="file"
            ):
                metadata[relative] = row
                stats.append(item_stats)
        else:
            context = mp.get_context("spawn")
            device_queue = context.Queue()
            for index in range(workers):
                device_queue.put(devices[index % len(devices)])
            try:
                with ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=context,
                    initializer=_initialize_worker,
                    initargs=(options, device_queue, threads_per_worker),
                ) as pool:
                    results = pool.map(_extract_file, jobs, chunksize=1)
                    for relative, row, item_stats in tqdm(
                        results, total=len(jobs), desc="Modal extraction", unit="file"
                    ):
                        metadata[relative] = row
                        stats.append(item_stats)
            finally:
                device_queue.close()
                device_queue.join_thread()
    finally:
        torch.set_num_threads(previous_threads)
    metadata = dict(sorted(metadata.items()))
    stats.sort(key=lambda row: row["path"])
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
        analyzer=f"{analysis_method}_upstream_cqt",
        frequency_units="radians/sample",
        processed_root=str(root),
        waveform_processing="none",
        execution=dict(
            devices=devices,
            workers=workers,
            threads_per_worker=threads_per_worker,
            precision="float32; TF32 disabled on CUDA",
        ),
    )
    total_seconds = time.perf_counter() - started
    summary = {
        "files": len(stats),
        "total_seconds": total_seconds,
        "mean_seconds_per_file": total_seconds / len(stats),
        "total_tracks": sum(row["tracks"] for row in stats),
        "total_active_tracks": sum(row["active_tracks"] for row in stats),
        "zero_mode_files": [row["path"] for row in stats if row["active_tracks"] == 0],
        "items": stats,
    }
    for name, content in [
        ("extraction_stats.json", summary),
        ("metadata.json", metadata),
        ("modal_config.json", config),
    ]:
        (out / name).write_text(json.dumps(content, indent=2), encoding="utf-8")
    if manifest is not None:
        (out / "split_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    print(
        f"Extracted {len(metadata)} waveforms in {total_seconds:.1f}s; {summary['total_tracks']} tracks ({summary['total_active_tracks']} active) -> {out}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-method", choices=("original", "corrected"), default="corrected"
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--devices", nargs="+", default=["cpu"])
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--out-dir", required=True)
    splits = parser.add_mutually_exclusive_group()
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
