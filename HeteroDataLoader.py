# Modified implementation from the library, solving a bug described below
# src: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/loader/dataloader.html#DataLoader
# Bug: if there were no nodes for one modality (e.g. no text nodes), that sample would have no connections involving that modality, e.g. ('text', ..., 'audio')
#      this is expected, but the collater was looking at the first element in each batch and using its edge indices for the whole batch.  
# Fix: kind of hacky, but I find the element in the batch with the most edge index types, and move it to the first position.  That way the collator takes the max edge types from that batch.

from typing import Union, List, Optional

from collections.abc import Mapping, Sequence
import numpy as np

import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Data, HeteroData, Dataset, Batch

class Collater(object):
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def collate(self, batch):
        elem = batch[0]

        lens = [len(list(elt.edge_index_dict.keys())) for elt in batch]
        max_idx = np.argmax(lens)
        tmp = batch[0]
        batch[0] = batch[max_idx]
        batch[max_idx] = tmp

        if isinstance(elem, Data) or isinstance(elem, HeteroData):
            ret = Batch.from_data_list(batch, self.follow_batch, self.exclude_keys)
        elif isinstance(elem, torch.Tensor):
            ret = default_collate(batch)
        elif isinstance(elem, float):
            ret = torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            ret = torch.tensor(batch)
        elif isinstance(elem, str):
            ret = batch
        elif isinstance(elem, Mapping):
            ret = {key: self.collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            ret = type(elem)(*(self.collate(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            ret = [self.collate(s) for s in zip(*batch)]
        else:
            raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

        return ret

    def __call__(self, batch):
        return self.collate(batch)


class DataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        dataset: Union[Dataset, List[Data], List[HeteroData]],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):

        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # Save for PyTorch Lightning...
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(dataset, batch_size, shuffle,
                         collate_fn=Collater(follow_batch,
                                             exclude_keys), **kwargs)
