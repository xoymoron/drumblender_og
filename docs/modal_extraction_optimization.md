# Parallel CQT extraction

The analysis settings, original full waveform, greedy matching order, strict thresholds, tie breaking, mode ranking, split membership, and output feature layout are unchanged. No resampling, waveform batching/padding, chunking, mode reduction, mixed precision, TF32, or tracking approximation is used.

## Implementation

- Vectorize magnitude/phase/peak preparation over frames. Compute pairwise distances once per frame and mark consumed columns; preserve the original sequential greedy assignment order.
- Store complete histories in contiguous arrays with geometric capacity growth, then convert selected tracks directly to tensors. Silent frame positions are retained.
- Use spawned processes with one cached CQT analyzer each. Workers handle independent complete files, write unique feature paths, and return small metadata records rather than transferring waveforms/features between processes.
- Distribute workers across the requested CUDA devices; several workers per GPU allow one file's CPU tracking/I/O to overlap other files' GPU CQT work. CPU tracking remains on CPU, not DDP.
- Limit torch CPU threads per worker, schedule larger files first, and use one job per scheduling unit. Reject unavailable CUDA devices and feature-path collisions before creating the cache.
- Record device/worker settings in modal_config.json. Cache format remains compatible with the training loader. Failed extraction never writes a completed dataset manifest/config.

## Server command

From the baseline repository root in its activated conda environment:

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/extract_baseline_modal.py \
  --processed-root ../dataset/processed \
  --reference-metadata ../dataset/modal_features/processed_modal_flat/metadata.json \
  --out-dir ../dataset/modal_features/drumblender_og_cqt64_parallel \
  --devices cuda:0 cuda:1 \
  --workers 8 \
  --threads-per-worker 1
```

This requests four processes per GPU. Set workers to 2 or 4 for a smaller CPU/RAM budget. Device indices are relative to CUDA_VISIBLE_DEVICES. The output directory must be new. An existing cache is never overwritten or mixed with partial new results. Existing old CPU caches remain usable.

For a CPU-only run preserving the reference CPU computation, use `--devices cpu --workers 8`. Single-worker CPU remains the default for backward compatibility.

## Verification and precision boundary

A frozen copy of the previous tracker is retained only as a test oracle. Random complex spectra, silent frames, observed peaks and multi-process cache generation are compared exactly, not only with an approximate tolerance.

On one synthetic 48 kHz, 2-second percussion signal (240 CQT bins, hop 256), the CPU CQT stage took 1.504 s, previous tracking 0.2233 s and optimized tracking 0.05161 s: **4.33x tracking speedup**, with identical track values. This is one CPU stage measurement, not an end-to-end dataset or two-GPU speed claim; CQT dominated this example.

FP32 CUDA CQT is not guaranteed bitwise identical to CPU convolution. Small floating-point differences can affect threshold-sensitive peak/track decisions. Therefore CPU exact-equivalence tests do not establish unchanged reconstruction quality for CUDA caches. No real RTX 3090 benchmark or CUDA equivalence result was obtained on the CPU-only development machine. GPU tests check both CQT values and final cache features on short synthetic files and should be run on the server before selecting CUDA caches:

```bash
python -m pytest -q -p no:cacheprovider -o addopts="" \
  test/unit/test_modal_optimization.py
```

CUDA-specific tests skip when the required devices are absent. Passing those small tests is a regression check, not a guarantee for every dataset item. If strict reproduction of CPU-derived features is mandatory, use the parallel CPU path. No checkpoint, optimizer, or learning strategy changes are part of this optimization.
