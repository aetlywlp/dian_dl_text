import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import math


# =======================
# 位置编码模块
# =======================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        初始化位置编码模块。

        :param d_model: 嵌入向量的维度。
        :param max_len: 支持的最大序列长度。
        """
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播，将位置编码添加到输入张量中。

        :param x: 输入张量，形状：(batch_size, seq_len, d_model)
        :return: 加入位置编码后的张量，形状：(batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return x


# =======================
# Transformer分类模型
# =======================
class TransformerClassifier(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, dim_feedforward, num_classes, max_len=5000):
        super(TransformerClassifier, self).__init__()

        self.d_model = d_model
        self.input_size = input_size

        # 输入线性层，将输入特征映射到d_model维度
        self.input_linear = nn.Linear(input_size, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)

        # 分类头
        self.fc = nn.Linear(d_model, num_classes)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.input_linear.bias.data.zero_()
        self.input_linear.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        """
        :param src: 输入张量，形状：(batch_size, seq_len, input_size)
        :return: 输出张量，形状：(batch_size, num_classes)
        """
        # 输入线性变换
        src = self.input_linear(src)  # (batch_size, seq_len, d_model)

        # 添加位置编码
        src = self.pos_encoder(src)  # (batch_size, seq_len, d_model)

        # Transformer需要的输入形状是(seq_len, batch_size, d_model)
        src = src.permute(1, 0, 2)  # (seq_len, batch_size, d_model)

        # Transformer编码器
        output = self.transformer_encoder(src)  # (seq_len, batch_size, d_model)

        # 聚合序列信息（取最后一个时间步的输出）
        output = output[-1, :, :]  # (batch_size, d_model)

        # 分类头
        output = self.fc(output)  # (batch_size, num_classes)

        return output


# =======================
# 检查是否可以使用 GPU
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备：", device)

# =======================
# 数据加载与预处理
# =======================

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载训练集和测试集（使用本地数据集，设置 download=False）
train_dataset = datasets.FashionMNIST(
    root='./data', train=True, download=False, transform=transform)
test_dataset = datasets.FashionMNIST(
    root='./data', train=False, download=False, transform=transform)

# 定义数据加载器
batch_size = 64

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)

# =======================
# 可视化部分数据
# =======================

# 获取一批数据
dataiter = iter(train_loader)
images, labels = next(dataiter)


# 定义展示函数
def imshow(img):
    img = img / 2 + 0.5  # 反归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.show()


# 显示部分训练图像
imshow(torchvision.utils.make_grid(images[:4]))
print('真实标签:', ' '.join('%5s' % labels[j].item() for j in range(4)))

# =======================
# 模型定义
# =======================

# 定义Transformer分类模型
# 参数选择示例，可以根据需要调整
input_size = 28  # 每个时间步的输入特征大小（图片的列数）
d_model = 128  # 嵌入向量的维度
nhead = 8  # 多头注意力机制的头数
num_encoder_layers = 3  # Transformer编码器层数
dim_feedforward = 512  # 前馈网络的隐藏层维度
num_classes = 10  # 分类数（Fashion-MNIST 有 10 个类别）
num_epochs = 5  # 训练轮数
learning_rate = 0.001  # 学习率

model = TransformerClassifier(input_size, d_model, nhead, num_encoder_layers, dim_feedforward, num_classes).to(device)

# =======================
# 训练模型
# =======================

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 记录训练过程的数据
train_losses = []
train_accuracies = []

for epoch in range(num_epochs):
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.squeeze().to(device)  # 形状：(batch_size, 28, 28)
        labels = labels.to(device)

        # 将图片的维度调整为：(batch_size, seq_len, input_size)
        images = images.permute(0, 2, 1)  # (batch_size, 28, 28)

        # 前向传播
        outputs = model(images)  # (batch_size, num_classes)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计数据
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

# =======================
# 模型评估
# =======================

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.squeeze().to(device)
        labels = labels.to(device)
        images = images.permute(0, 2, 1)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f'Test Accuracy: {test_accuracy:.2f}%')

# =======================
# 结果可视化
# =======================

# 绘制训练损失曲线
plt.figure(figsize=(10, 5))
plt.title("Training Loss")
plt.plot(train_losses, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# 绘制训练准确率曲线
plt.figure(figsize=(10, 5))
plt.title("Training Accuracy")
plt.plot(train_accuracies, label="Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.show()

# 可视化部分预测结果
dataiter = iter(test_loader)
images, labels = next(dataiter)
images_display = images[:4]
images = images.squeeze().to(device)
images = images.permute(0, 2, 1)
labels = labels.to(device)

# 预测
outputs = model(images)
_, predicted = torch.max(outputs.data, 1)

# 显示图片及预测结果
imshow(torchvision.utils.make_grid(images_display))
print('真实标签:', ' '.join('%5s' % labels[j].item() for j in range(4)))
print('预测标签:', ' '.join('%5s' % predicted[j].item() for j in range(4)))
