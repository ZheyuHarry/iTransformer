import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding_inverted
from layers.SelfAttention_Family import FullAttention , AttentionLayer
from layers.Transformer_EncDec import EncoderLayer , Encoder
import numpy as np

class Model(nn.Module):
    def __init__(self , configs):
        super(Model , self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len , configs.d_model , configs.embed , configs.freq , configs.dropout)
        #
        self.class_strategy = configs.class_strategy

        # Encoder-only architecture
        # ---------------------------------------------------------------#
        #            创建一个Encoder，用列表生成式生成，N层layer           #
        # ---------------------------------------------------------------#
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False , configs.factor , attention_dropout=configs.dropout ,
                                       output_attention=configs.output_attention) ,
                        configs.d_model , configs.n_heads) , configs.d_model , configs.d_ff,
                        configs.dropout , activation=configs.activation
                    ) for l in range(configs.e_layers)
            ] ,
        norm_layer=nn.LayerNorm(configs.d_model)
        )
        # 输出层投影
        self.projector = nn.Linear(configs.d_model , configs.pred_len , bias=True)

    def forecast(self , x_enc , x_mark_enc , x_dec , x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            # -----------------------------------------------------#
            #       计算均值和标准差然后移除统计特征并存储起来        #
            # -----------------------------------------------------#
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _ , _ , N = x_enc.shape # B , L , N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E (这个是iTransformer中把每一个变量独立的当作一个token，所以是N个token) (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc , x_mark_enc) # 这里把时间特征编码也一起嵌入

        # B N E -> B N E
        # 这里将每个变量的token通过嵌入进行注意力汇聚，然后得到新的表征
    
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N
        # 这里直接进行预测 ， 是Encoder-only结构，直接输出最后结果
        dec_out = self.projector(enc_out).permute(0 , 2 , 1)[: , : , :N] # 这里是为了筛选掉我们嵌入的时候加进去的时间特征

        if self.use_norm:
            # 把当时存储起来的统计特征还回去
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self , x_enc , x_mark_enc , x_dec , x_mark_dec , mask=None):
        dec_out = self.forecast(x_enc , x_mark_enc , x_dec , x_mark_dec)
        return dec_out[: , -self.pred_len: , :] # 输出最后的预测长度，因为我们一开始是包含了一些前缀信息的
