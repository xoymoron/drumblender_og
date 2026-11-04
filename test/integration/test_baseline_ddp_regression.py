"""Exercise real DDP gradient synchronization using two CPU worker processes."""

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from drumblender.baseline import (
    BaselineDataModule,
    VariableLengthDrumBlender,
    PerItemMRSTFTLoss,
)
from drumblender.models.soundstream import (
    SoundStreamEncoder,
    SoundStreamAttentionEncoder,
)
from drumblender.synths import ModalSynth, NoiseGenerator, TransientTCN


class SyntheticPercussion(Dataset):
    lengths = [513 + i * 8 for i in range(8)]

    def __len__(self):
        return 8

    def __getitem__(self, index):
        length = 513 + index * 8
        clock = torch.arange(length).float()
        audio = (clock * 0.1).sin()[None] * 0.1
        tracks = torch.zeros(3, 2, 5)
        tracks[0] = 0.1
        tracks[1] = torch.linspace(0.1, 0.01, 5)
        return audio, tracks, length


class CheckSynchronizedWeights(pl.Callback):
    def on_fit_start(self, trainer, pl_module):
        torch.set_num_threads(1)
        self.initial = torch.cat(
            [p.detach().flatten() for p in pl_module.parameters() if p.requires_grad]
        ).clone()

    def on_train_end(self, trainer, pl_module):
        assert trainer.world_size == 2
        parameters = torch.cat(
            [p.detach().flatten() for p in pl_module.parameters() if p.requires_grad]
        )
        assert torch.isfinite(parameters).all()
        assert not torch.equal(parameters, self.initial)
        copies = [torch.empty_like(parameters) for _ in range(trainer.world_size)]
        dist.all_gather(copies, parameters)
        torch.testing.assert_close(copies[0], copies[1], rtol=0, atol=0)


class TrainableBaseline(VariableLengthDrumBlender):
    def __init__(self):
        super().__init__(
            modal_synth=ModalSynth(),
            loss_fn=PerItemMRSTFTLoss(),
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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


class SyntheticData(BaselineDataModule):
    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataset = SyntheticPercussion()
        self.val_dataset = SyntheticPercussion()


def test_two_process_baseline_training_synchronizes_weights(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    pl.seed_everything(20261106, workers=True)
    model = TrainableBaseline()
    data = SyntheticData("unused", batch_size=2, num_workers=0)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=2,
        strategy="ddp_spawn",
        replace_sampler_ddp=False,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
        callbacks=[CheckSynchronizedWeights()],
    )
    trainer.fit(model, datamodule=data)
