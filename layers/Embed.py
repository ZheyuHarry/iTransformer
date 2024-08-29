import torch 
import torch.nn as nn
import math

class DataEmbedding_inverted(nn.Module):
    """
        进行inverted的数据嵌入
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        """
            @param d_model: 表示嵌入空间的维度，通常会是512
        """
        super(DataEmbedding_inverted , self).__init__()
        self.value_embedding = nn.Linear(c_in , d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self , x , x_mark):
        """
            这里需要进行维度转换，主要是第二维和第三维进行交换
        """
        x = x.permute(0 , 2 , 1)
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x , x_mark.permute(0 , 2 , 1)] , dim=1)) # 这里的话是把时间特征标记也编码进去提供timestamp的信息
        return self.dropout(x)