import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np  # 确保导入 numpy

# 检查是否可以使用 GPU
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
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 显示部分训练图像
imshow(torchvision.utils.make_grid(images[:4]))
print(' '.join('%5s' % labels[j].item() for j in range(4)))


# =======================
# 模型定义
# =======================

class CustomRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CustomRNN, self).__init__()
        self.hidden_size = hidden_size
        # 定义输入到隐藏状态的线性层
        self.input2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        # 定义隐藏状态到输出的线性层
        self.hidden2output = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs):
        # inputs: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = inputs.size()
        hidden = torch.zeros(batch_size, self.hidden_size).to(inputs.device)

        for t in range(seq_len):
            input_t = inputs[:, t, :]
            combined = torch.cat((input_t, hidden), dim=1)
            hidden = torch.tanh(self.input2hidden(combined))

        output = self.hidden2output(hidden)
        return output


# =======================
# 训练模型
# =======================

# 定义超参数和模型实例
input_size = 28  # 每个时间步的输入特征大小（图片的列数）
hidden_size = 128  # 隐藏层大小
num_classes = 10  # 分类数（Fashion-MNIST 有 10 个类别）
num_epochs = 5  # 训练轮数
learning_rate = 0.001

model = CustomRNN(input_size, hidden_size, num_classes).to(device)

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
        images = images.permute(0, 2, 1)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss .backward()
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
