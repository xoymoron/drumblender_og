# Variable-length all-parallel baseline

This is the upstream all-parallel DrumBlender architecture retrained under the reference project's experiment protocol. It is an adapted baseline, not a reproduction of the original paper's training schedule or published numbers. No pretrained weights are used.

## Fixed experiment settings

The settings were taken from the reference project's `cfg/05_all_parallel.yaml` and `cfg/data/custom_all.yaml` on 2026-11-04:

| Setting | Value |
| --- | --- |
| Input | Existing mono 48,000 Hz WAVs, complete duration |
| Model | Original all-parallel modal + transient + noise routing and parameter shapes |
| Optimizer | Adam, learning rate 0.0001 |
| Scheduler | ReduceLROnPlateau, `validation/loss`, factor 0.5, patience 10, minimum LR 0.00000625 |
| Early stopping | `validation/loss`, patience 20, finite check, epoch-end check |
| Maximum epochs | 150 |
| Training batch / accumulation | 8 / 8, standard Lightning automatic optimization |
| Device count | 1 |
| Seed | 20261106 |
| Workers | 12 |
| Bucketing | Same quantile-bucket sampler as the reference project; 8 buckets, retain incomplete batches, shuffle each epoch |
| Validation batches | Length sorted; keep every item |
| Objective | Mean per-item auraloss MR-STFT on valid waveform extents, same FFT defaults as the reference |
| Modal extraction | Original CQT/SMS approach with demonstrated indexing fixes; 64 modes, hop 256, fmin 20, 240 bins, 24 bins/octave, threshold -80 dB, matching threshold 5 percent, minimum 10 observed frames |
| Checkpoints | Monitor validation loss, keep best 5 and last |

Eight times eight is a nominal effective batch of 64. Incomplete bucket batches and the final accumulation group follow normal Lightning behavior, exactly as requested. No custom gradient rescaling, oversampling, adaptive sample budgets, or per-item validation RNG is added. Noise synthesis remains stochastic, as in the reference training path.

## Data and split contract

Extraction reads already-preprocessed WAVs. It does not download, trim, normalize, resample, truncate, or append samples to the saved training targets. It rejects non-mono audio, mismatched rates, missing split members, and attempts to overwrite a cache. Modal features retain the complete CQT frame axis, including silent frames. Short inputs use constant CQT boundary padding only when reflection padding cannot run; the waveform itself is unchanged.

Supply either the reference cache's complete `metadata.json`, or a frozen JSON map from preprocessed-root-relative WAV paths to `train`, `val`, or `test`. Reference metadata splits use the same sorted reference IDs, full pack names, random generator, seed, and 80/10/10 within-pack allocation as the reference dataset. Re-splitting by new feature filenames would change membership even with the same seed, so it is not used. A within-pack split does not establish held-out-pack generalization. If a different split is selected for the paper, supply that same explicit manifest to both systems.

Train and compare each model on exactly these waveform members. Missing reference files are an error rather than an excuse to re-split. Extract into a fresh directory; legacy, Hybrid, and corrected CQT caches must not share a directory.

## Minimal changes and audit results

| Finding | Verdict and independent evidence | Correction / regression |
| --- | --- | --- |
| Fixed 2-second dataset cannot train complete variable clips | CONFIRMED: original loader asserts the configured sample count | Cache-only dataset loads full WAVs; tests cover unchanged sample amplitudes and lengths |
| Modal/encoder padding changes short-item reconstruction | CONFIRMED by forward-path inspection: modal interpolation consumes the whole padded frame axis, and attention/FiLM receive padded audio | Carry sample/frame lengths and crop before upstream forward; compare against unpadded reconstruction with identical noise draws |
| Naively forwarding every item separately changes equal-length training BatchNorm | CONFIRMED: regression differed in 1990/2002 output elements before correction | Use the reference project's grouping by identical sample/frame extents; equal-length training matches native upstream forward |
| Padded MR-STFT averages introduce batch-dependent objectives | CONFIRMED by the original unmasked auraloss call and per-item/padding regression | Crop each item and average its MR-STFT; FFT settings unchanged. This replaces original batch-coupled spectral convergence, matching the reference objective |
| Single-frame noise tensors fail complex conversion | CONFIRMED: 129-sample real all-parallel forward raised the stride-divisibility RuntimeError | Standard `torch.complex(real, zeros)` conversion; full-length forward and gradients tested |
| CQT tracking drops time frames / merges first peaks / mismatches phases | CONFIRMED by original tracking loop and synthetic multi-peak/silent-frame regression | Preserve all frame positions, initialize each peak independently, delete matched phase/frequency/amplitude entries together; empty analysis returns zero-mode tensors |
| Zero-dB peaks are incorrectly removed | CONFIRMED: synthetic two-peak example yielded only one track | Boolean threshold/local-maximum detection, independent of the numerical dB value |
| Float32 cumulative phase loses accuracy over 20 seconds | CONFIRMED: relative waveform error 0.0233 for a constant 12.3 kHz oscillator at 48 kHz | Float64 phase accumulation and modulo reduction before casting the carrier back; regression requires error below 0.000001 |
| Same seed alone guarantees the same train/test members | REFUTED: random permutations depend on the ordered IDs and pack membership | Reuse reference IDs before deriving a path-based manifest; a deliberately reversed filename/ID regression checks membership |
| Original environment supports current reference dependencies | PARTIAL: removed torchaudio archive import and NumPy 2 / Lightning 1.9 conflict reproduced | Standard-library archive helper, reference package versions and Python range; verify in an isolated supported environment |

No original tests are removed. The underlying trainable encoders, TCN, FiLM layers and all-parallel routing are retained. BatchNorm uses valid-extent groups, matching the reference adaptation; this is not a claim that ragged training reproduces padded fixed-duration paper training. No learned normalization replacement or custom masked layer is introduced.

## Server commands

Use a separate virtual environment for this checkout, since both repositories install a package named `drumblender`. Python 3.10-3.12 is supported; NumPy is pinned to 1.26.4 for Lightning 1.9.5. Install the appropriate CUDA build of the pinned torch/torchaudio versions on the training server. The historical `train-packages.txt` is an upstream environment snapshot; the current dependency declaration is `pyproject.toml`.

From this repository root, with environment variables pointing to existing processed audio and the reference cache:

```bash
python -m pip install -e ".[modal]"

python scripts/extract_baseline_modal.py \
  --processed-root "$PROCESSED_ROOT" \
  --out-dir "$BASELINE_CACHE" \
  --reference-metadata "$REFERENCE_CACHE/metadata.json"

drumblender fit --config cfg/05_all_parallel_variable.yaml \
  --data.init_args.data_dir "$BASELINE_CACHE" \
  --data.init_args.audio_dir "$PROCESSED_ROOT"
```

`BASELINE_CACHE` must be a fresh path. If using an already-frozen path manifest, substitute `--split-manifest "$SPLIT_MANIFEST"` for `--reference-metadata`. The original numbered configs remain available as upstream references; use the new variable-length config for this experiment. Save the resolved config, cache configuration, manifest, environment versions, and source revision with each run.

## Verification boundary

Local regressions exercise the actual synthesis/encoder classes at reduced widths through 960,000 samples, loss/backward finiteness, padding independence, parameter-shape parity, extraction of silent/short audio, exact reference splits, and a tiny end-to-end run using the full configured model with standard optimizer/scheduler/accumulation/checkpoint callbacks. This is implementation evidence. It does not measure GPU peak memory for eight 20-second clips, convergence, reconstruction quality, or the server dataset. Do not alter batch/accumulation for only one model if a shared hardware constraint requires revising the experiment protocol.

Final verification on 2026-11-04: **132 tests passed**, including all original tests, under Python 3.10 / NumPy 1.26.4 / torch 2.7.1 CPU / Lightning 1.9.5. Focused Ruff checks passed. The existing `.venv-modal` Python 3.13 / NumPy 2 environment gave 124 passes and 7 original/new config failures before the end-to-end test was added; the cause was Lightning's removed `np.Inf` dependency. No third-party runtime monkeypatch was applied.

A separate synthetic inference check used the **full configured 2,692,833-parameter model**, 64 modes, and 960,000 samples. It returned finite output of shape `(1, 1, 960000)` in 9.193 seconds on this CPU with two torch threads. This single forward checks the full model's length handling; it is not a GPU training-memory or reconstruction benchmark.

Source and regression pointers:

| Area | Source | Regression |
| --- | --- | --- |
| Full waveforms, frame lengths and valid normalization groups | `drumblender/baseline.py:69`, `:142`, `:194` | `test/unit/test_variable_baseline.py:70`, `:79`, `:328` |
| Per-item spectral loss | `drumblender/baseline.py:41` | `test/unit/test_variable_baseline.py:103` |
| Reference bucketing and epoch shuffle | `drumblender/data/bucketing.py:11`, `drumblender/baseline.py:110` | `test/unit/test_variable_baseline.py:128`, `:313` |
| CQT time/peak/phase indexing | `drumblender/baseline_analysis.py:12`, `:51` | `test/unit/test_variable_baseline.py:174`, `:192` |
| Noise tensor strides / modal phase precision | `drumblender/synths/noise.py:52`, `drumblender/synths/modal.py:60` | `test/unit/test_variable_baseline.py:70`, `:345` |
| Exact split and full pack provenance | `scripts/extract_baseline_modal.py:18`, `:48` | `test/unit/test_variable_baseline.py:192`, `:273` |
| Config parity and actual optimizer/checkpoint path | `cfg/05_all_parallel_variable.yaml:1` | `test/unit/test_variable_baseline.py:292`, `:361` |

The original, unmodified transient-conditioning option remains off in this all-parallel config; this audit does not certify that unused optional path. No existing checkpoints or published scores are migrated into the new experiment. Train the adapted baseline from scratch.
