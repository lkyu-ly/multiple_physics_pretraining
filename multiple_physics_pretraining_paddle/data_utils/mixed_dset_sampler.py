from typing import Iterator

import numpy as np
import paddle

__all__ = ["MultisetSampler"]


class MultisetSampler(paddle.io.Sampler):
    """Sampler that restricts data loading to a subset of the dataset."""

    def __init__(
        self,
        dataset: paddle.io.Dataset,
        base_sampler: paddle.io.Sampler,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        max_samples=10,
        rank=0,
        distributed=True,
    ) -> None:
        self.batch_size = batch_size
        self.sub_dsets = dataset.sub_dsets
        if distributed:
            self.sub_samplers = [
                base_sampler(dataset, drop_last=drop_last) for dataset in self.sub_dsets
            ]
        else:
            self.sub_samplers = [base_sampler(dataset) for dataset in self.sub_dsets]
        self.dataset = dataset
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.max_samples = max_samples
        self.rank = rank

    def __iter__(self) -> Iterator[int]:
        samplers = [iter(sampler) for sampler in self.sub_samplers]
        sampler_choices = list(range(len(samplers)))
        generator = paddle.Generator()
        generator.manual_seed(100 * self.epoch + 10 * self.seed + self.rank)
        count = 0
        while len(sampler_choices) > 0:
            count += 1
            index_sampled = paddle.randint(
                low=0, high=len(sampler_choices), shape=(1,)
            ).item()
            dset_sampled = sampler_choices[index_sampled]
            offset = max(0, self.dataset.offsets[dset_sampled])
            try:
                queue = []
                for i in range(self.batch_size):
                    queue.append(next(samplers[dset_sampled]) + offset)
                if len(queue) == self.batch_size:
                    for d in queue:
                        yield d
            except Exception as err:
                print("ERRRR", err)
                sampler_choices.pop(index_sampled)
                print(
                    f"Note: dset {dset_sampled} fully used. Dsets remaining: {len(sampler_choices)}"
                )
                continue
            if count >= self.max_samples:
                break

    def __len__(self) -> int:
        return len(self.dataset)

    def set_epoch(self, epoch: int) -> None:
        """
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        for sampler in self.sub_samplers:
            sampler.set_epoch(epoch)
        self.epoch = epoch
