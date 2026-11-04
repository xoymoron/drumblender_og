# Local and two-GPU training environments

Use Python 3.10-3.12 and an isolated environment for this checkout.
The local .venv has Python 3.10.11, NumPy 1.26.4, Lightning 1.9.5,
and torch/torchaudio 2.7.1+cpu. Do not install another drumblender checkout
into this same environment.

Local setup, from this repository root:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/local.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider -o addopts=""
```

On the Linux RTX 3090 server:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/server.txt
python -m pip check
nvidia-smi
CUDA_VISIBLE_DEVICES=0,1 drumblender fit \
  --config cfg/05_all_parallel_variable.yaml \
  --data.init_args.data_dir "$BASELINE_CACHE" \
  --data.init_args.audio_dir "$PROCESSED_ROOT"
```

The server driver must support CUDA 12.6. The training configuration uses
standard DDP with two devices. Each rank gets a sharded sampler, and Lightning
must not replace it (replace_sampler_ddp=false). Validation losses synchronize
across ranks before scheduler/checkpoint/early-stopping decisions.
Per-GPU batch 2 x 2 devices x accumulation 8 gives nominal effective batch 32.
The last partial batch uses standard Lightning accumulation.
For a local CLI smoke test, explicitly override trainer.accelerator=cpu,
trainer.devices=1, and trainer.strategy=auto.

The real two-process CPU regression is test/integration/test_baseline_ddp_regression.py.
It uses small models and short synthetic inputs. It does not establish CUDA/NCCL
compatibility or peak GPU memory. DDP does not pool GPU memory: each 24 GiB card
holds a model replica. Peak memory for full 20-second training on these cards
must still be measured on the server. If necessary, per-GPU batch 1 with
accumulation 16 preserves the same nominal effective batch of 32.
