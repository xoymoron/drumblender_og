"""Export full and direct sinusoidal reconstruction WAVs from an OG checkpoint."""

import argparse
import csv
import json
import shutil
from pathlib import Path

import pytorch_lightning as pl
import soundfile as sf
import torch
from tqdm import tqdm

from drumblender.baseline import BaselineDataset, VariableLengthDrumBlender
from drumblender.utils.model import load_config_yaml, load_model


@torch.inference_mode()
def render_pair(model, audio, params):
    """Keep analyzed modal amplitudes and initial phases in the original units."""
    prediction = model(audio[None], params[None])[0]
    sinusoidal = model.modal_synth(params[None], audio.shape[-1])[0]
    return prediction, sinusoidal


@torch.inference_mode()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Resolved training config")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--metrics-config", type=Path, required=True,
                        help="The common evaluator's drumblender_metrics.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.metrics_config.is_file():
        parser.error(f"Missing common metric configuration: {args.metrics_config}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Select --device cpu.")
    config = load_config_yaml(str(args.config))
    seed = config.get("seed_everything", 20261106)
    pl.seed_everything(seed, workers=True)
    model, _ = load_model(str(args.config), str(args.checkpoint))
    if not isinstance(model, VariableLengthDrumBlender):
        parser.error("Expected a VariableLengthDrumBlender checkpoint.")
    model = model.eval().to(args.device)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    settings = config["data"]["init_args"]
    dataset = BaselineDataset(
        args.data_dir or settings["data_dir"],
        split=args.split,
        sample_rate=settings["sample_rate"],
        expected_num_modes=settings["expected_num_modes"],
        audio_dir=args.audio_dir or settings.get("audio_dir"),
    )
    if settings.get("require_canonical_audio") and not dataset.canonical_audio:
        parser.error("Extract the cache with --canonical-audio-metadata.")
    if not len(dataset):
        parser.error(f"Empty {args.split} split.")
    args.output.mkdir(parents=True, exist_ok=False)
    for folder in ("target", "recon", "sinusoidal", "evaluation", "configs"):
        (args.output / folder).mkdir()
    shutil.copyfile(
        args.metrics_config, args.output / "configs" / "evaluation_metrics.yaml"
    )
    rows = []
    for index in tqdm(range(len(dataset)), unit="clip"):
        key = dataset.keys[index]
        if dataset.canonical_audio:
            dataset.verify_audio(key)
        audio, params, _ = dataset[index]
        source = dataset.metadata[key]
        relative = Path(source["orig_relpath"].replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe relative waveform path: {relative}")
        prediction, sinusoidal = render_pair(
            model, audio.to(args.device), params.to(args.device)
        )
        for folder, wave in (
            ("target", audio), ("recon", prediction), ("sinusoidal", sinusoidal)
        ):
            path = args.output / folder / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(
                path, wave.flatten().cpu().numpy(), dataset.sample_rate, subtype="FLOAT"
            )
        rows.append({
            "meta_key": key,
            "source_filename": relative.as_posix(),
            "orig_relpath": relative.as_posix(),
            "sample_pack_key": source.get("sample_pack_key", "__root__"),
            "length": audio.shape[-1],
        })
    with (args.output / "evaluation" / "per_file_loss.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "items": len(rows),
        "split": args.split,
        "seed": seed,
        "sinusoidal": "Direct cached modal synthesis before learned corrections",
    }
    (args.output / "evaluation" / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
