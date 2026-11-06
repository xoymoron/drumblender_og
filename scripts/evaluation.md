# Reconstruction Metrics

Run the following commands from the project root to compute reconstruction metrics that were reported in the paper.

Requires:
- Dataset (unfortunately, it is private)
- Pre-trained models, which can downloaded by running the following command in the project root:
    ```bash
    git submodule update --init --recursive
    ```

## Commands to run test metrics
### Cross intrument metrics:

```bash
./scripts/reconstruction_metrics.sh models/forum-acusticum-2023 all_parallel && ./scripts/reconstruction_metrics.sh models/forum-acusticum-2023 noise_parallel_transient_params && ./scripts/reconstruction_metrics.sh models/forum-acusticum-2023 noise_params && ./scripts/reconstruction_metrics.sh models/forum-acusticum-2023 noise_transient_params && ./scripts/reconstruction_metrics.sh models/forum-acusticum-2023 transient_params
```

### Instrument dependent metrics:
```bash
./scripts/reconstruction_metrics_inst.sh models/forum-acusticum-2023 all_parallel && ./scripts/reconstruction_metrics_inst.sh models/forum-acusticum-2023 noise_parallel_transient_params && ./scripts/reconstruction_metrics_inst.sh models/forum-acusticum-2023 noise_params && ./scripts/reconstruction_metrics_inst.sh models/forum-acusticum-2023 noise_transient_params && ./scripts/reconstruction_metrics_inst.sh models/forum-acusticum-2023 transient_params
```

### Modal-only metrics
```bash
./scripts/reconstruction_modal.sh
```

## Compile Results
After running all the reconstruction evaluation scripts, results will be saved in a series of csv files in a logs directory. To compile these results into Latex tables used in the paper, these scripts can be used:

```bash
python scripts/compile_results.py logs all
python scripts/compile_results.py logs instrument
```

## Variable-length baseline on shared audio

Use a fresh run checkout and the reference run's processed WAVs. Copy the WAVs
as files and keep their relative paths. Use the frozen baseline metadata
containing the reference SHA-256, sample count and split for each WAV.

On RTX 5090 use the PyTorch 2.7.1 and torchaudio 2.7.1 CUDA 12.8 builds.
Confirm that `torchaudio.list_audio_backends()` includes `sox`. Match the
reference Python 3.10.21 and package versions in a new environment. The
CUDA/cuDNN builds depend on the GPU architecture.

```bash
python -m pip install 'scipy==1.15.3' 'nnAudio==0.3.4' 'numpy==1.26.4' \
  'soundfile==0.14.0' 'jsonargparse[signatures]==4.52.0' 'wandb==0.30.0'
```

```bash
python scripts/extract_baseline_modal.py \
  --processed-root "$SHARED_WAVS" \
  --out-dir "$NEW_OG_CACHE" \
  --canonical-audio-metadata "$FROZEN_OG_METADATA" \
  --analysis-method original \
  --devices cuda:0 cuda:1 --workers 8

drumblender fit --config cfg/05_all_parallel_variable.yaml \
  --data.init_args.data_dir "$NEW_OG_CACHE"
```

The extractor verifies every WAV and carries the frozen split into the new CQT
cache. Lightning rechecks the reference files at startup. The loader uses SoX
with no resampling or quantization during training and test. The recipe uses
FP32 and disables TF32 for matmul and cuDNN. Keep the original training config
with any historical checkpoint.

For one-off inference with the variable-length recipe, use the training run's
resolved config and a WAV from its cache. `drumblender-synth` loads the cached
modal features and writes a float32 WAV. For a relocated WAV root, set
`data.init_args.audio_dir` in the resolved config.
