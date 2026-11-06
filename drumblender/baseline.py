"""Full-clip training adapter around the unchanged upstream DrumBlender model."""

import json
from pathlib import Path
from typing import Optional

import auraloss
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from drumblender.data.bucketing import BucketingBatchSampler
from drumblender.tasks.drumblender import DrumBlender


def _counts(values, count, maximum):
    result = torch.as_tensor(values)
    if (
        result.ndim != 1
        or result.numel() != count
        or result.dtype == torch.bool
        or result.is_floating_point()
        or torch.any(result < 1)
        or torch.any(result > maximum)
    ):
        raise ValueError("Invalid valid-length counts")
    return result.tolist()


class PerItemMRSTFTLoss(torch.nn.Module):
    """Preserve auraloss settings, reducing valid item losses with equal weight."""

    def __init__(self, mrstft: Optional[torch.nn.Module] = None):
        super().__init__()
        self.mrstft = (
            mrstft if mrstft is not None else auraloss.freq.MultiResolutionSTFTLoss()
        )

    def forward(self, pred, target, lengths=None):
        if pred.shape != target.shape or pred.ndim != 3 or min(pred.shape) < 1:
            raise ValueError("Expected matching nonempty audio [B,C,T]")
        sizes = (
            [pred.shape[-1]] * len(pred)
            if lengths is None
            else _counts(lengths, len(pred), pred.shape[-1])
        )
        minimum = max((s.fft_size // 2 + 1 for s in self.mrstft.stft_losses), default=1)
        losses = []
        for i, size in enumerate(sizes):
            x, y = pred[i : i + 1, :, :size], target[i : i + 1, :, :size]
            pad = max(0, minimum - size)
            if pad:
                x = F.pad(x, (0, pad))
                y = F.pad(y, (0, pad))
            losses.append(self.mrstft(x.contiguous(), y.contiguous()))
        return torch.stack(losses).mean()


class VariableLengthDrumBlender(DrumBlender):
    """Use the original encoders, synths, weights and all-parallel signal routing.

    Equal valid extents share a forward, matching the reference project's
    variable-length policy. Encoders and FiLM never observe collate padding.
    """

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        audio, params, lengths, frame_lengths = batch
        return (
            audio.to(device, non_blocking=True),
            params.to(device, non_blocking=True),
            lengths,
            frame_lengths,
        )

    def forward(self, original, params, lengths=None, frame_lengths=None):
        if original.ndim != 3 or original.shape[1] != 1 or min(original.shape) < 1:
            raise ValueError("Expected nonempty mono audio [B,1,T]")
        if params.ndim != 4 or params.shape[0] != len(original):
            raise ValueError("Expected modal parameters [B,P,M,F]")
        if lengths is None and frame_lengths is None:
            return super().forward(original, params)
        batch, _, maximum = original.shape
        sizes = (
            [maximum] * batch if lengths is None else _counts(lengths, batch, maximum)
        )
        frames = (
            [params.shape[-1]] * batch
            if frame_lengths is None
            else _counts(frame_lengths, batch, params.shape[-1])
        )
        groups = {}
        for index, extent in enumerate(zip(sizes, frames)):
            groups.setdefault(extent, []).append(index)
        results = [None] * batch
        for (size, frame_count), indices in groups.items():
            selection = indices
            if len(indices) == 1:
                selection = slice(indices[0], indices[0] + 1)
            x = original[selection, :, :size]
            p = params[selection, ..., :frame_count]
            y = super().forward(x, p)
            if y.shape != x.shape:
                raise ValueError(
                    f"Synthesized shape {y.shape} differs from target {x.shape}"
                )
            if size < maximum:
                y = F.pad(y, (0, maximum - size))
            for local, index in enumerate(indices):
                results[index] = y[local : local + 1]
        return results[0] if batch == 1 else torch.cat(results)

    def _do_step(self, batch):
        x, p, lengths, frames = batch
        y = self(x, p, lengths, frames)
        return self.loss_fn(y, x, lengths=lengths), y

    def training_step(self, batch, batch_idx):
        loss, _ = self._do_step(batch)
        batch_size = len(batch[0])
        self.log(
            "train/loss_step", loss, on_step=True, on_epoch=False, batch_size=batch_size
        )
        self.log(
            "train/loss_epoch",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self._do_step(batch)
        self.log(
            "validation/loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch[0]),
            sync_dist=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, _ = self._do_step(batch)
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=len(batch[0]),
            sync_dist=True,
        )
        return loss


def collate_baseline(items):
    """Keep original waveform and modal extents, including silent modal frames."""
    if not items:
        raise ValueError("Cannot collate an empty batch")
    audio, params, _ = zip(*items)
    sizes = [x.shape[-1] for x in audio]
    frames = [p.shape[-1] for p in params]
    if any(x.ndim != 2 or x.shape[0] != 1 or x.shape[-1] < 1 for x in audio):
        raise ValueError("Expected mono nonempty input audio")
    if any(
        p.ndim != 3 or p.shape[:2] != params[0].shape[:2] or p.shape[-1] < 1
        for p in params
    ):
        raise ValueError("Modal parameter and mode axes must match")
    max_size, max_frames = max(sizes), max(frames)
    x = torch.stack(
        [
            F.pad(a, (0, max_size - n)) if n < max_size else a
            for a, n in zip(audio, sizes)
        ]
    )
    p = torch.stack(
        [
            F.pad(v, (0, max_frames - n)) if n < max_frames else v
            for v, n in zip(params, frames)
        ]
    )
    return x, p, torch.tensor(sizes), torch.tensor(frames)


class BaselineDataset(Dataset):
    """Read an extracted cache and its frozen split without preprocessing audio."""

    def __init__(
        self,
        data_dir,
        split=None,
        sample_rate=48000,
        expected_num_modes=64,
        audio_dir=None,
    ):
        self.data_dir = Path(data_dir)
        self.config = json.loads((self.data_dir / "modal_config.json").read_text())
        if self.config["sample_rate"] != sample_rate:
            raise ValueError("Cache sample rate differs from training sample rate")
        if self.config["num_modes"] != expected_num_modes:
            raise ValueError("Cache mode count differs from the selected baseline")
        self.metadata = json.loads((self.data_dir / "metadata.json").read_text())
        for row in self.metadata.values():
            if row.get("split") not in ("train", "val", "test"):
                raise ValueError("Every item needs a frozen train/val/test split")
        self.keys = [
            k
            for k in sorted(self.metadata)
            if split is None or self.metadata[k]["split"] == split
        ]
        self.lengths = [int(self.metadata[k]["num_samples"]) for k in self.keys]
        self.sample_rate, self.expected_num_modes = sample_rate, expected_num_modes
        self.audio_dir = Path(audio_dir) if audio_dir is not None else None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        row = self.metadata[key]
        path = (
            self.audio_dir / row["orig_relpath"]
            if self.audio_dir is not None
            else self.data_dir / row["filename"]
        )
        x, sr = torchaudio.load(str(path))
        if (
            sr != self.sample_rate
            or x.shape != (1, self.lengths[index])
            or not torch.isfinite(x).all()
        ):
            raise ValueError(
                f"Audio does not match preprocessed cache provenance: {key}"
            )
        p = torch.load(
            self.data_dir / row["feature_file"], map_location="cpu", weights_only=True
        )
        if (
            p.ndim != 3
            or p.shape[0] != 3
            or p.shape[-1] < 1
            or p.shape[1] > self.expected_num_modes
            or not torch.isfinite(p).all()
        ):
            raise ValueError(f"Invalid modal features: {key}")
        if p.shape[1] < self.expected_num_modes:
            p = F.pad(p, (0, 0, 0, self.expected_num_modes - p.shape[1]))
        return x, p.float(), key


class BaselineDataModule(pl.LightningDataModule):
    """Local-cache-only, single-device loaders with exact validation membership."""

    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 48000,
        expected_num_modes: int = 64,
        audio_dir: Optional[str] = None,
        batch_size: int = 8,
        num_workers: int = 12,
        seed: int = 20261106,
        use_bucketing: bool = True,
    ):
        super().__init__()
        self.data_dir, self.sample_rate = data_dir, sample_rate
        self.expected_num_modes, self.audio_dir = expected_num_modes, audio_dir
        self.batch_size, self.num_workers = batch_size, num_workers
        self.seed, self.use_bucketing = seed, use_bucketing

    def prepare_data(self):
        for name in ["metadata.json", "modal_config.json", "split_manifest.json"]:
            if not (Path(self.data_dir) / name).is_file():
                raise FileNotFoundError(f"Extract modal features first: missing {name}")

    def setup(self, stage=None):
        splits = (
            ("train", "val", "test")
            if stage is None
            else {"fit": ("train", "val"), "validate": ("val",), "test": ("test",)}[
                stage
            ]
        )
        for split in splits:
            dataset = BaselineDataset(
                self.data_dir,
                split,
                self.sample_rate,
                self.expected_num_modes,
                self.audio_dir,
            )
            if not len(dataset):
                raise ValueError(f"The frozen {split} split is empty")
            setattr(self, f"{split}_dataset", dataset)

    def _distributed_state(self):
        """Keep all sampler sharding here instead of Lightning's replacement."""
        world_size = 1
        global_rank = 0
        current_epoch = 0
        if getattr(self, "trainer", None) is not None:
            world_size = int(getattr(self.trainer, "world_size", 1) or 1)
            global_rank = int(getattr(self.trainer, "global_rank", 0) or 0)
            current_epoch = int(getattr(self.trainer, "current_epoch", 0) or 0)
            connector = getattr(self.trainer, "_accelerator_connector", None)
            if world_size > 1 and getattr(connector, "replace_sampler_ddp", False):
                raise ValueError(
                    "AudioDataModule shards distributed loaders itself. Set "
                    "Trainer(replace_sampler_ddp=False) to prevent Lightning "
                    "from replacing its duration-aware batch sampler."
                )
        return world_size, global_rank, current_epoch

    def _loader(self, dataset, training):
        world_size, rank, epoch = self._distributed_state()
        if self.use_bucketing:
            sampler = BucketingBatchSampler(
                dataset.lengths,
                self.batch_size,
                seed=self.seed,
                shuffle=training,
                boundaries=None if training else [],
                num_replicas=world_size,
                rank=rank,
            )
            sampler.set_epoch(epoch)
            return DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=self.num_workers,
                collate_fn=collate_baseline,
                pin_memory=True,
                persistent_workers=training and self.num_workers > 0,
            )
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=training,
                seed=self.seed,
                drop_last=False,
            )
            if world_size > 1
            else None
        )
        if sampler is not None:
            sampler.set_epoch(epoch)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=training and sampler is None,
            num_workers=self.num_workers,
            collate_fn=collate_baseline,
            pin_memory=True,
            persistent_workers=training and self.num_workers > 0,
            drop_last=False,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, False)
