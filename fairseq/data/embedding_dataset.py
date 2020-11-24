from fairseq.data import data_utils

from . import BaseWrapperDataset

import torch.nn.functional as F
from torch import nn


class EmbeddingDataset(BaseWrapperDataset):

    def __init__(self, dataset, pad_idx, left_pad, cls_format='max_pool'):
        # left and right pad
        for i, data in enumerate(dataset):
            dataset[i] = F.pad(data, (0, 0, 1, 1), mode='constant', value=0)
        
        if cls_format=='max_pool':
            for i, data in enumerate(dataset):
                maxpool = nn.MaxPool1d(data.size(0))
                dataset[i][0] = maxpool(data.transpose(0,1).unsqueeze(0))[0].transpose(1,0)
        else:
            print('Unknown pooling format !!!')

        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad

    def collater(self, samples):
        return data_utils.collate_2d_tokens(samples, self.pad_idx, left_pad=self.left_pad)