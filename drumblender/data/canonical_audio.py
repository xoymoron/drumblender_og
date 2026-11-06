"""Verify shared WAVs before loading them with the reference decoder."""

import hashlib

import soundfile as sf
import torchaudio


def verify_canonical_waveform(path, row, sample_rate):
    """Check the frozen file identity and the training waveform format."""
    digest = row.get("audio_sha256")
    if not digest or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Canonical WAV SHA-256 mismatch: {path}")
    info = sf.info(path)
    if (
        info.format not in ("WAV", "WAVEX")
        or info.subtype != "FLOAT"
        or info.channels != 1
        or info.samplerate != sample_rate
        or info.frames != row["num_samples"]
    ):
        raise ValueError(f"Expected canonical mono float32 WAV at {sample_rate} Hz: {path}")


def load_canonical_waveform(path):
    """Use the DSP training decoder even when FFmpeg is installed."""
    return torchaudio.load(str(path), backend="sox")
