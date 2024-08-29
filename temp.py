import torch

# 定义张量形状
B = 2  # Batch size
L = 3  # Length of queries
S = 4  # Length of keys
H = 2  # Number of heads
E = 5  # Embedding size per head

# 随机生成 queries 和 keys 张量
queries = torch.randn(B, L, H, E)  # 形状为 [2, 3, 2, 5]
keys = torch.randn(B, S, H, E)  # 形状为 [2, 4, 2, 5]

# 计算注意力分数
scores = torch.einsum("blhe,bshe->bhls", queries, keys)

# 打印结果形状
print(scores.shape)  # 结果: torch.Size([2, 2, 3, 4])
