import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import torch
from . import BasicModule
from module import Embedding, CNN
from module import Capsule as CapsuleLayer
import numpy as np
from typing import List, Tuple, Dict, Union

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


def to_one_hot(x: torch.Tensor, length: int) -> torch.Tensor:
    """
    Args:
        x (torch.Tensor):[B] , 一般是 target 的值
        length (int) : L ,一般是关系种类树
    Return:
        x_one_hot.to(device=x.device) (torch.Tensor) : [B, L]  每一行，只有对应位置为1，其余为0
    """
    B = x.size(0)
    x_one_hot = torch.zeros(B, length)
    for i in range(B):
        x_one_hot[i, x[i]] = 1.0

    return x_one_hot.to(device=x.device)


class Capsule(BasicModule):
    def __init__(self, cfg):
        super(Capsule, self).__init__()

        if cfg.dim_strategy == 'cat':
            cfg.in_channels = cfg.word_dim + 2 * cfg.pos_dim
        else:
            cfg.in_channels = cfg.word_dim

        # capsule config
        cfg.input_dim_capsule = cfg.out_channels
        cfg.num_capsule = cfg.num_attributes

        self.num_attributes = cfg.num_attributes
        self.embedding = Embedding(cfg)
        self.cnn = CNN(cfg)
        self.capsule = CapsuleLayer(cfg)

    def forward(self, x):
        word, lens, entity_pos, attribute_value_pos = x['word'], x['lens'], x['entity_pos'], x['attribute_value_pos']
        mask = seq_len_to_mask(lens)
        inputs = self.embedding(word, entity_pos, attribute_value_pos)

        primary, _ = self.cnn(inputs)  # 由于长度改变，无法定向mask，不mask可可以，毕竟primary capsule 就是粗粒度的信息
        output = self.capsule(primary)
        output = output.norm(p=2, dim=-1)  # 求得模长再返回值

        return output  # [B, N]

    def loss(self, predict, target, reduction='mean'):
        m_plus, m_minus, loss_lambda = 0.9, 0.1, 0.5

        target = to_one_hot(target, self.num_attributes)
        max_l = (torch.relu(m_plus - predict))**2
        max_r = (torch.relu(predict - m_minus))**2
        loss = target * max_l + loss_lambda * (1 - target) * max_r
        loss = torch.sum(loss, dim=-1)

        if reduction == 'sum':
            return loss.sum()
        else:
            # 默认情况为求平均
            return loss.mean()
