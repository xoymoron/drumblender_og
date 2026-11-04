"""Distributed baseline loaders must shard once and keep equal step counts."""

from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

from drumblender.baseline import BaselineDataModule


class Items(Dataset):
    lengths = [32] * 13

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, index):
        return torch.full((1, 32), float(index)), torch.zeros(3, 2, 2), str(index)


def test_loaders_shard_each_rank_once():
    for bucketing in (True, False):
        counts = []
        for rank in range(2):
            data = BaselineDataModule(
                "unused", batch_size=3, num_workers=0, use_bucketing=bucketing
            )
            data.trainer = SimpleNamespace(
                world_size=2,
                global_rank=rank,
                current_epoch=0,
                _accelerator_connector=SimpleNamespace(replace_sampler_ddp=False),
            )
            data.train_dataset = Items()
            loader = data.train_dataloader()
            indices = [int(i) for batch in loader for i in batch[0][:, 0, 0]]
            assert len(set(indices)) < len(Items())
            counts.append(len(loader))
        assert counts[0] == counts[1]
