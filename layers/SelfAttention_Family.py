import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask

class FullAttention(nn.Module):
    def __init__(self , mask_flag = True , factor = 5 , scale = None , attention_dropout=0.1 , output_attention=False):
        """
            @param mask_flag: 表示是否要进行masking操作
            @param factor:
            @param scale: Scaled Dot-Product Attention需要进行缩放，防止点积爆炸，梯度消失
        """
        super(FullAttention , self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self , queries , keys , values , attn_mask , tau = None , delta = None):
        """
            @param queries , keys , values: 对应地查询向量和键值向量
            @param attn_mask: 注意力masking矩阵
            @param tau:
        """

        """
        Dimension Explanation:
            B:
            L:
            H:
            E:
            S:
            D:
        """
        B , L , H , E = queries.shape
        _ , S , _ , D = values.shape
        scale = self.scale or 1. / sqrt(E)
        # 仍然理解不了这里的维度变化的逻辑
        """
            这里的直观理解是B&H表示某个样本的某个注意力头，某一个时间步的嵌入[L , E]与另一个时间步的嵌入[S , E]求点积 ， 由于公式中需要转置后者
            所以[L , E] * [E , S] ==> [L , S] , 表示query的第L个时间步和key的第S个时间步的相似度 ， 最后放到每个样本和每个注意力头上
            就得到了[B , H , L , S]
        """
        scores = torch.einsum("blhe,bshe->bhls" , queries , keys)

        # Masking
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B , L , device=queries.device)
            
            # 这里对应的mask矩阵为True的地方都用负无穷代替，这就是Mask Attention的操作 ， 这样softmax的注意力分数就接近0了
            scores.masked_fill_(attn_mask.mask , -np.inf)
        
        # 这两步就分别是计算点积注意力分数，然后加权求和
        A = self.dropout(torch.softmax(scale * scores , dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A , values)

        if self.output_attention:
            return (V.contiguous() , A) # 要在内存上让他们连续
        else:
            return (V.contiguous() , None)


class AttentionLayer(nn.Module):
    def __init__(self , attention , d_model , n_heads , d_keys = None , d_values = None):
        super(AttentionLayer , self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention

        # 这就是计算投影的WQ，WK，WV，用来将输入数据进行投影到q,k,v的
        self.query_projection = nn.Linear(d_model , d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self , queries , keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau,
            delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn