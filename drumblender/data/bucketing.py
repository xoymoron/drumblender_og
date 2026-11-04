"""Duration-aware batch sampling without dataset-specific time boundaries."""

from bisect import bisect_left
from numbers import Integral
from typing import Iterable, List, Optional, Sequence

import torch
from torch.utils.data import Sampler


class BucketingBatchSampler(Sampler[List[int]]):
    """Shuffle within quantile buckets, or emit length-sorted evaluation batches.

    Explicit boundaries are sample counts and override quantile estimation.
    With ``drop_last=False`` every item is retained. Distributed training repeats
    complete batches when needed to give all ranks the same number of steps;
    those repeats are not extra independent observations.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        boundaries: Optional[Sequence[int]] = None,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 20261106,
        num_replicas: int = 1,
        rank: int = 0,
        num_buckets: int = 8,
    ):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        for name, value in (
            ("batch_size", batch_size),
            ("num_buckets", num_buckets),
            ("num_replicas", num_replicas),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(rank, Integral) or not 0 <= rank < num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        if any(
            isinstance(length, bool) or not isinstance(length, Integral) or length < 1
            for length in self.lengths
        ):
            raise ValueError("lengths must contain positive integer sample counts")

        if boundaries is None:
            ordered = sorted(self.lengths)
            # Lower empirical quantiles keep boundaries at observed sample counts.
            self.boundaries = (
                sorted(
                    {
                        ordered[(len(ordered) - 1) * index // num_buckets]
                        for index in range(1, num_buckets)
                    }
                    - {ordered[-1]}
                )
                if ordered
                else []
            )
        else:
            self.boundaries = list(boundaries)
            if any(
                isinstance(value, bool) or not isinstance(value, Integral) or value < 1
                for value in self.boundaries
            ) or any(
                left >= right
                for left, right in zip(self.boundaries, self.boundaries[1:])
            ):
                raise ValueError("boundaries must be strictly increasing sample counts")
        self.buckets = [[] for _ in range(len(self.boundaries) + 1)]
        for index, length in enumerate(self.lengths):
            self.buckets[bisect_left(self.boundaries, length)].append(index)

    def set_epoch(self, epoch: int) -> None:
        """Use a reproducible, different ordering for each training epoch."""
        self.epoch = epoch

    def __iter__(self) -> Iterable[List[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        batches = []
        for bucket in self.buckets:
            indices = bucket.copy()
            if self.shuffle:
                permutation = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[index] for index in permutation]
            else:
                # Leave the shortest remainder for the final incomplete batch.
                indices.sort(key=lambda index: self.lengths[index], reverse=True)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            permutation = torch.randperm(len(batches), generator=generator).tolist()
            batches = [batches[index] for index in permutation]

        # In distributed training, shard batches across ranks while keeping
        # equal number of steps per rank to avoid DDP synchronization issues.
        if self.num_replicas > 1:
            total = len(batches)
            if self.drop_last:
                total_size = (total // self.num_replicas) * self.num_replicas
                batches = batches[:total_size]
            elif total:
                per_rank = (total + self.num_replicas - 1) // self.num_replicas
                total_size = per_rank * self.num_replicas
                batches += [
                    batches[index % total] for index in range(total_size - total)
                ]
            batches = batches[self.rank :: self.num_replicas]

        yield from batches

    def __len__(self) -> int:
        total = sum(
            (
                len(bucket) // self.batch_size
                if self.drop_last
                else (len(bucket) + self.batch_size - 1) // self.batch_size
            )
            for bucket in self.buckets
        )
        if self.drop_last:
            return total // self.num_replicas
        return (total + self.num_replicas - 1) // self.num_replicas
