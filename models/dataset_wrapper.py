from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence, Tuple, Union

from datasets import Dataset

PathLike = Union[str, Path]
PathsInput = Union[PathLike, Sequence[PathLike]]
DatasetFactory = Callable[[Path], Dataset]


@dataclass
class DatasetWrapper:
    """
    A single node in a linked-list of datasets.

    Each node stores:
      - path: the dataset path on disk
      - next: link to the next node (or None)
      - dataset_factory: a callable that loads a `datasets.Dataset` from the path
    """
    path: Path
    next: Optional["DatasetWrapper"] = None
    dataset_factory: Optional[DatasetFactory] = None

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def dataset(self) -> Dataset:
        if self.dataset_factory is None:
            raise ValueError(
                "DatasetWrapper has no dataset_factory. Attach one via MultiDatasetsWrapper(...) or node.with_factory(...).")
        return self.dataset_factory(self.path)

    def name_path(self) -> Tuple[str, Path]:
        return self.name, self.path

    def name_dataset(self) -> Tuple[str, Dataset]:
        return self.name, self.dataset

    def with_factory(self, dataset_factory: DatasetFactory) -> "DatasetWrapper":
        # Keep links; just attach/replace factory
        return DatasetWrapper(path=self.path, next=self.next, dataset_factory=dataset_factory)

    def __len__(self) -> int:
        count = 0
        cur: Optional[DatasetWrapper] = self
        while cur is not None:
            count += 1
            cur = cur.next
        return count

    def __str__(self) -> str:
        return self.name


class MultiDatasetsWrapper:
    """
    A lightweight linked-list wrapper around one or more dataset paths.

    Provides:
      - .first: first DatasetWrapper node (or raises if empty)
      - iteration over DatasetWrapper nodes
      - iteration over (name, path) or (name, dataset)
    """

    def __init__(self, paths: PathsInput, dataset_factory: Optional[DatasetFactory] = None):
        norm_paths = self._normalize_paths(paths)

        head: Optional[DatasetWrapper] = None
        for p in reversed(norm_paths):
            head = DatasetWrapper(path=p, next=head, dataset_factory=dataset_factory)
        self._head = head
        self._len = len(norm_paths)

    @staticmethod
    def _normalize_paths(paths: PathsInput) -> Tuple[Path, ...]:
        if isinstance(paths, (str, Path)):
            seq = [paths]
        else:
            seq = list(paths)

        norm: list[Path] = []
        for p in seq:
            pp = Path(p).expanduser().resolve()
            norm.append(pp)
        return tuple(norm)

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(node.name for node in self)

    @property
    def first(self) -> DatasetWrapper:
        if self._head is None:
            raise ValueError("MultiDatasetsWrapper is empty.")
        return self._head

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[DatasetWrapper]:
        cur = self._head
        while cur is not None:
            yield cur
            cur = cur.next

    def iter_name_path(self) -> Iterator[Tuple[str, Path]]:
        for node in self:
            yield node.name, node.path

    def iter_name_dataset(self) -> Iterator[Tuple[str, Dataset]]:
        for node in self:
            yield node.name, node.dataset

    def with_factory(self, dataset_factory: DatasetFactory) -> "MultiDatasetsWrapper":
        return MultiDatasetsWrapper([n.path for n in self], dataset_factory=dataset_factory)

    def __str__(self) -> str:
        return ", ".join(str(node) for node in self)
