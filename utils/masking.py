"""
    设置掩码
"""
import torch

class TriangularCausalMask():
    def __init__(self , B , L , device="cpu"):
        mask_shape = [B , 1 , L , L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape , dtype=torch.bool) , diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    """
        这段代码是根据输入的索引和分数来生成一个相应的掩码
    """
    def __init__(self, B, H, L, index, scores, device="cpu"):
        # 这里的scores.shape[-1]是指获得scores的最后一个维度的大小，然后保留一个上三角掩码矩阵
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        # 这里在_mask的前面增加了两个维度，变成了(1,1,L,D),再扩展就变成了(B.H,L,D)，然后每个批次和每个头上面都是同样的上三角掩码矩阵
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])

        # torch.arange(B)[:, None, None] 生成一个形状为 (B, 1, 1) 的张量，表示批次的索引
        # torch.arange(H)[None, :, None] 生成一个形状为 (1, H, 1) 的张量，表示头的索引。
        # index这个索引表示第三个维度，这样可以遍历每一个批次，每一个头，给定index的值，来组成一个新的tensor
        indicator = _mask_ex[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
        ].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask
