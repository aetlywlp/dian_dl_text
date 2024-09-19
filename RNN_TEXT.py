import torch
import torch.nn as nn

class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # 定义输入到隐藏层的线性变换
        self.input2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        # 定义隐藏层到输出的线性变换（可选）
        self.hidden2output = nn.Linear(hidden_size, hidden_size)

    def forward(self, inputs, hidden=None):
        # inputs: (seq_len, batch_size, input_size)
        seq_len, batch_size, _ = inputs.size()
        if hidden is None:
            # 初始化隐藏状态
            hidden = torch.zeros(batch_size, self.hidden_size).to(inputs.device)
        outputs = []

        for t in range(seq_len):
            # 取出当前时间步的输入
            input_t = inputs[t]
            # 将输入和隐藏状态拼接
            combined = torch.cat((input_t, hidden), dim=1)
            # 计算新的隐藏状态
            hidden = torch.tanh(self.input2hidden(combined))
            # 计算输出（如果需要）
            output = self.hidden2output(hidden)
            outputs.append(output)

        # 将所有时间步的输出堆叠起来
        outputs = torch.stack(outputs)
        return outputs, hidden

# 使用示例
input_size = 10    # 输入特征的大小
hidden_size = 20   # 隐藏层大小
seq_len = 5        # 序列长度
batch_size = 3     # 批量大小

# 创建模型实例
model = CustomRNN(input_size, hidden_size)
# 随机生成输入数据
inputs = torch.randn(seq_len, batch_size, input_size)
# 前向传播
outputs, hidden = model(inputs)
print(outputs.shape)  # 输出形状: (seq_len, batch_size, hidden_size)
print(hidden)
