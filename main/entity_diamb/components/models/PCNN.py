import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import BasicModule
from module import Embedding, CNN
from typing import List, Tuple, Dict, Union
import numpy as np

def seq_len_to_mask(seq_len: Union[List, np.ndarray, torch.Tensor], max_len=None, mask_pos_to_true=True):
    """
    将一个表示sequence length的一维数组转换为二维的mask，默认pad的位置为1。
    转变 1-d seq_len到2-d mask。

    Args :
        seq_len (list, np.ndarray, torch.LongTensor) : shape将是(B,)
        max_len (int): 将长度pad到这个长度。默认(None)使用的是seq_len中最长的长度。但在nn.DataParallel的场景下可能不同卡的seq_len会有区别，所以需要传入一个max_len使得mask的长度是pad到该长度。
    Return: 
        mask (np.ndarray, torch.Tensor) : shape将是(B, max_length)， 元素类似为bool或torch.uint8
    """
    if isinstance(seq_len, list):
        seq_len = np.array(seq_len)

    if isinstance(seq_len, np.ndarray):
        seq_len = torch.from_numpy(seq_len)

    if isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, logger.error(f"seq_len can only have one dimension, got {seq_len.dim()} != 1.")
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len.device)
        if mask_pos_to_true:
            mask = broad_cast_seq_len.ge(seq_len.unsqueeze(1))
        else:
            mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise logger.error("Only support 1-d list or 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask

class PCNN(BasicModule):
    def __init__(self, cfg):
        super(PCNN, self).__init__()

        self.use_pcnn = cfg.use_pcnn
        if cfg.dim_strategy == 'cat':
            cfg.in_channels = cfg.word_dim + 2 * cfg.pos_dim
        else:
            cfg.in_channels = cfg.word_dim

        self.embedding = Embedding(cfg)
        self.cnn = CNN(cfg)
        self.fc1 = nn.Linear(len(cfg.kernel_sizes) * cfg.out_channels, cfg.intermediate)
        self.fc2 = nn.Linear(cfg.intermediate, cfg.num_attributes)
        self.dropout = nn.Dropout(cfg.dropout)

        if self.use_pcnn:
            self.fc_pcnn = nn.Linear(3 * len(cfg.kernel_sizes) * cfg.out_channels,
                                     len(cfg.kernel_sizes) * cfg.out_channels)
            self.pcnn_mask_embedding = nn.Embedding(4, 3)
            masks = torch.tensor([[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]])
            self.pcnn_mask_embedding.weight.data.copy_(masks)
            self.pcnn_mask_embedding.weight.requires_grad = False


    def forward(self, x):
        word, lens, entity_pos, attribute_value_pos = x['word'], x['lens'], x['entity_pos'], x['attribute_value_pos']
        mask = seq_len_to_mask(lens)

        inputs = self.embedding(word, entity_pos, attribute_value_pos)
        out, out_pool = self.cnn(inputs, mask=mask)

        if self.use_pcnn:
            out = out.unsqueeze(-1)  # [B, L, Hs, 1]
            pcnn_mask = x['pcnn_mask']
            pcnn_mask = self.pcnn_mask_embedding(pcnn_mask).unsqueeze(-2)  # [B, L, 1, 3]
            out = out + pcnn_mask  # [B, L, Hs, 3]
            out = out.max(dim=1)[0] - 100  # [B, Hs, 3]
            out_pool = out.view(out.size(0), -1)  # [B, 3 * Hs]
            out_pool = F.leaky_relu(self.fc_pcnn(out_pool))  # [B, Hs]
            out_pool = self.dropout(out_pool)

        output = self.fc1(out_pool)
        output = F.leaky_relu(output)
        output = self.dropout(output)
        output = self.fc2(output)

        return output
