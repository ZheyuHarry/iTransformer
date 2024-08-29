import torch.nn as nn
import torch.nn.functional as F

class EncoderLayer(nn.Module):
    def __init__(self , attention , d_model , d_ff = None , dropout = 0.1 , activation="relu"):
        super(EncoderLayer , self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        # FFN
        self.conv1 = nn.Conv1d(d_model , d_ff , kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff , d_model , kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self , x , attn_mask = None , tau = None , delta = None):
        # queries , keys , values都输入原来的x
        # 这个时候的形状是(B , N , E) , 通过多头注意力汇聚之后得到的形状仍然是(B , N , E)
        new_x , attn = self.attention(
            x , x , x,
            attn_mask = attn_mask,
            tau=tau , delta=delta
        )

        # 第一层的Add&Norm
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)

        # FFN
        y = self.dropout(self.activation(self.conv1(y.transpose(-1 , 1))))
        y = self.dropout(self.conv2(y).transpose(-1 , 1))

        # 第二层Add&Norm
        return self.norm2(x + y) , attn


class Encoder(nn.Module):
    def __init__(self , attn_layers , conv_layers = None , norm_layer=None):
        super(Encoder , self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        """
            这里就是把每一个EncoderLayer都执行一遍，然后得到最后的正则化输出
        """
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(
                zip(self.attn_layers, self.conv_layers)
            ):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
