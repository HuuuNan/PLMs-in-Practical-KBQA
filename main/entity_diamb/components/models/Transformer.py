import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import torch.nn as nn
from . import BasicModule
from module import Embedding
from module import Transformer as TransformerBlock
import torch
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


class Transformer(BasicModule):
    def __init__(self, cfg):
        super(Transformer, self).__init__()

        if cfg.dim_strategy == 'cat':
            cfg.hidden_size = cfg.word_dim + 2 * cfg.pos_dim
        else:
            cfg.hidden_size = cfg.word_dim

        self.embedding = Embedding(cfg)
        self.transformer = TransformerBlock(cfg)
        self.fc = nn.Linear(cfg.hidden_size, cfg.num_attributes)

    def forward(self, x):
        word, lens, entity_pos, attribute_value_pos = x['word'], x['lens'], x['entity_pos'], x['attribute_value_pos']
        mask = seq_len_to_mask(lens)
        inputs = self.embedding(word, entity_pos, attribute_value_pos)
        last_layer_hidden_state, all_hidden_states, all_attentions = self.transformer(inputs, key_padding_mask=mask)
        out_pool = last_layer_hidden_state.max(dim=1)[0]
        output = self.fc(out_pool)

        return output
