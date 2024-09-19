import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =======================
# 多头注意力模块定义
# =======================
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        初始化多头注意力模块。

        :param embed_dim: 嵌入向量的维度。
        :param num_heads: 注意力头的数量。
        :param dropout: Dropout概率。
        """
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "嵌入维度必须能被头数整除。"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 定义线性层，将Q、K、V映射到不同的子空间
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        # 输出的线性层
        self.out_linear = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, query, key, value, mask=None):
        """
        前向传播函数。

        :param query: 查询张量，形状：(batch_size, seq_len, embed_dim)
        :param key: 键张量，形状：(batch_size, seq_len, embed_dim)
        :param value: 值张量，形状：(batch_size, seq_len, embed_dim)
        :param mask: 掩码张量，形状：(batch_size, 1, 1, seq_len) 或其他适用形状
        :return: 注意力输出，形状：(batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = query.size()

        # 线性变换并分头
        Q = self.q_linear(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = self.k_linear(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)    # (batch_size, num_heads, seq_len, head_dim)
        V = self.v_linear(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        # 计算注意力得分
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch_size, num_heads, seq_len, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        attn_output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len, head_dim)

        # 拼接头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)  # (batch_size, seq_len, embed_dim)

        # 最终线性变换
        output = self.out_linear(attn_output)  # (batch_size, seq_len, embed_dim)

        return output, attn_weights

# =======================
# 示例：计算随机矩阵的注意力权重
# =======================
def main():
    # 设置随机种子以便复现
    torch.manual_seed(0)

    # 定义参数
    batch_size = 2
    seq_len = 4
    embed_dim = 8
    num_heads = 2

    # 生成随机的查询、键、值矩阵
    Q = torch.randn(batch_size, seq_len, embed_dim)
    K = torch.randn(batch_size, seq_len, embed_dim)
    V = torch.randn(batch_size, seq_len, embed_dim)

    print("查询 (Q):\n", Q)
    print("\n键 (K):\n", K)
    print("\n值 (V):\n", V)

    # 实例化多头注意力模块
    multi_head_attn = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)

    # 计算注意力输出和权重
    output, attn_weights = multi_head_attn(Q, K, V)

    print("\n注意力输出 (Output):\n", output)
    print("\n注意力权重 (Attention Weights):\n", attn_weights)

    # 分析注意力权重
    for b in range(batch_size):
        for h in range(num_heads):
            print(f"\nBatch {b+1}, Head {h+1} Attention Weights:")
            print(attn_weights[b, h])

if __name__ == '__main__':
    main()
