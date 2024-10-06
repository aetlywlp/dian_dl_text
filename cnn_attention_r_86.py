# 导入必要的库
# 导入必要的库
import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from scipy.stats import mode
from sklearn.preprocessing import StandardScaler
from collections import Counter
import warnings
from torch.cuda.amp import autocast, GradScaler

# 忽略警告
warnings.filterwarnings("ignore")

# 检查是否有可用的 GPU，如果有则使用 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义全局参数
DATA_DIR = '/media/aetly/Data/BaiduNetdiskDownload/sEMG_DeepLearning-master/datah5'  # 数据目录
WINDOW_SIZE = 200  # 每个窗口包含的样本数
STEP_SIZE = 100  # 窗口滑动的步长
BATCH_SIZE = 256  # 增大批量大小
NUM_EPOCHS = 80  # 训练周期数，减少总训练时间
EARLY_STOPPING_PATIENCE = 40  # 早停的耐心次数，避免过拟合
LEARNING_RATE = 0.001  # 增大学习率，加快收敛速度


def load_and_preprocess_data(data_dir):
    """
    加载并预处理数据，包括窗口化、标准化和标签处理。

    参数:
    - data_dir: 数据文件所在的目录。

    返回:
    - emg_data: 预处理后的 EMG 数据，形状为 (N_samples, window_size, num_channels) 的张量。
    - emg_label: 对应的标签，形状为 (N_samples,) 的张量。
    - num_classes: 类别数量。
    """
    # 获取所有 .h5 文件的列表
    file_list = glob.glob(os.path.join(data_dir, '*.h5'))
    if not file_list:
        raise ValueError("数据目录中没有找到任何 .h5 文件。")

    windowed_data_list = []
    windowed_label_list = []

    # 遍历所有数据文件并读取数据
    for file_path in file_list:
        with h5py.File(file_path, 'r') as f:
            emg = f['emg'][:]  # 读取 'emg' 数据集，形状：(num_samples, num_channels)
            restimulus = f['restimulus'][:].squeeze()  # 读取 'restimulus' 数据集，形状：(num_samples,)

            # 检查 restimulus 是否为一维
            if restimulus.ndim != 1:
                print(f"文件 {file_path} 中的 'restimulus' 数据无法展平为一维，跳过该文件。")
                continue

            num_samples, num_channels = emg.shape
            num_windows = (num_samples - WINDOW_SIZE) // STEP_SIZE + 1  # 计算可生成的窗口数量
            if num_windows <= 0:
                print(f"文件 {file_path} 中的数据样本不足以生成一个窗口，跳过该文件。")
                continue

            # 数据增强：添加高斯噪声
            noise = np.random.normal(0, 0.005, emg.shape)
            emg += noise

            # 使用滑动窗口方法获取数据窗口
            try:
                windows = np.array([emg[i * STEP_SIZE:i * STEP_SIZE + WINDOW_SIZE] for i in range(num_windows)])
                # windows 形状：(num_windows, WINDOW_SIZE, num_channels)
            except Exception as e:
                print(f"窗口化数据时出错，文件: {file_path}, 错误: {e}")
                continue

            # 获取每个窗口的标签（众数）
            window_labels = []
            for i in range(len(windows)):
                labels_in_window = restimulus[i * STEP_SIZE:i * STEP_SIZE + WINDOW_SIZE]
                if len(labels_in_window) == 0:
                    continue
                try:
                    window_label = mode(labels_in_window, axis=None).mode.item()
                except Exception as e:
                    print(f"计算众数时出错，文件: {file_path}, 窗口: {i}, 错误: {e}")
                    window_label = -1  # 设置一个无效标签
                window_labels.append(window_label)

            # 过滤掉无效标签
            window_labels = np.array(window_labels)
            valid_mask = window_labels != -1  # 创建一个布尔掩码，标记有效的标签
            windows = windows[valid_mask]
            window_labels = window_labels[valid_mask]

            if len(window_labels) == 0:
                print(f"文件 {file_path} 中没有有效的窗口，跳过该文件。")
                continue

            windowed_data_list.append(windows)  # 将当前文件的窗口数据添加到列表
            windowed_label_list.append(window_labels)  # 将对应的标签添加到列表

    if not windowed_data_list:
        raise ValueError("没有可用的窗口化数据，请检查数据文件和窗口设置。")

    # 拼接所有窗口化数据和标签
    emg_data = np.concatenate(windowed_data_list, axis=0)  # 形状：(总窗口数, WINDOW_SIZE, num_channels)
    emg_label = np.concatenate(windowed_label_list, axis=0)  # 形状：(总窗口数,)

    print('EMG 数据形状:', emg_data.shape)
    print('标签形状:', emg_label.shape)
    print('唯一的标签:', np.unique(emg_label))

    # 数据标准化，每个通道单独标准化
    num_samples, window_size, num_channels = emg_data.shape
    emg_data = emg_data.reshape(-1, num_channels)  # 重塑为 (总窗口数 * WINDOW_SIZE, num_channels)
    scaler = StandardScaler()
    emg_data = scaler.fit_transform(emg_data)  # 标准化
    emg_data = emg_data.reshape(num_samples, window_size, num_channels)  # 重新调整形状

    # 标签处理，确保标签从0开始
    unique_labels = np.unique(emg_label)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    emg_label = np.array([label_mapping[label] for label in emg_label])

    # 更新类别数量
    num_classes = len(unique_labels)
    print(f'调整后类别数量: {num_classes}')
    print(f'标签范围: {emg_label.min()} - {emg_label.max()}')

    # 转换为 PyTorch 张量
    emg_data = torch.tensor(emg_data, dtype=torch.float32)
    emg_label = torch.tensor(emg_label, dtype=torch.long)

    return emg_data, emg_label, num_classes


class ResidualBlock(nn.Module):
    """残差块，用于构建深层 CNN 模型。"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, downsample=False):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if self.downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.downsample_layer = None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample_layer is not None:
            identity = self.downsample_layer(x)
        out += identity
        out = self.relu(out)
        return out


class CNNModel(nn.Module):
    """基于 CNN 的模型，包含多个残差块。"""

    def __init__(self, num_channels, num_classes):
        super(CNNModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.layer2 = ResidualBlock(64, 128, downsample=True)
        self.layer3 = ResidualBlock(128, 256, downsample=True)
        self.layer4 = ResidualBlock(256, 512, downsample=True)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 自适应平均池化，输出长度为1
        self.fc = nn.Linear(512, num_classes)  # 全连接层，输出为类别数

    def forward(self, x):
        # x 形状: (batch_size, window_size, num_channels)
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, num_channels, window_size)
        x = self.layer1(x)  # 通过第一层卷积
        x = self.layer2(x)  # 通过第二层残差块
        x = self.layer3(x)  # 通过第三层残差块
        x = self.layer4(x)  # 通过第四层残差块
        x = self.avg_pool(x)  # 自适应平均池化，形状: (batch_size, 512, 1)
        x = x.squeeze(-1)  # 形状: (batch_size, 512)
        x = self.fc(x)  # 全连接层输出，形状: (batch_size, num_classes)
        return x


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs):
    """训练模型并使用早停策略。"""
    best_acc = 0.0  # 初始化最佳准确率
    early_stopping_counter = 0  # 早停计数器
    scaler = GradScaler()  # 用于混合精度训练

    # 用于记录损失和准确率
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        # 每个 epoch 有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式，启用 Dropout 和 BatchNorm
            else:
                model.eval()  # 设置模型为评估模式，禁用 Dropout 和 BatchNorm

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # 混合精度训练
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                if phase == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                # 统计损失和准确率
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_acc_value = epoch_acc.item()

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc_value:.4f}")

            # 记录损失和准确率
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc_value)
                scheduler.step()  # 更新学习率
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc_value)

                # 保存最佳模型
                if epoch_acc_value > best_acc:
                    best_acc = epoch_acc_value
                    torch.save(model.state_dict(), 'best_model.pth')
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                        print("早停触发，停止训练")
                        plot_history(history)
                        return history

    print("训练完成！")
    # 绘制损失和准确率曲线
    plot_history(history)
    return history


def evaluate_model(model, dataloader, criterion):
    """评估模型在测试集上的性能。"""
    model.eval()  # 设置模型为评估模式
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():  # 不计算梯度，节省内存和计算
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失

            _, preds = torch.max(outputs, 1)  # 获取预测结果
            running_loss += loss.item() * inputs.size(0)  # 累计损失
            running_corrects += torch.sum(preds == labels.data)  # 累计正确预测数量

    total_loss = running_loss / len(dataloader.dataset)  # 计算平均损失
    total_acc = running_corrects.double() / len(dataloader.dataset)  # 计算准确率

    print(f'测试集损失: {total_loss:.4f} | 测试集准确率: {total_acc:.4f}')

    return total_loss, total_acc.item()


def plot_history(history):
    """绘制训练和验证的损失与准确率曲线。"""
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('损失曲线')
    plt.xlabel('周期')
    plt.ylabel('损失')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.title('准确率曲线')
    plt.xlabel('周期')
    plt.ylabel('准确率')
    plt.legend()

    plt.show()


def main():
    # 加载和预处理数据
    emg_data, emg_label, num_classes = load_and_preprocess_data(DATA_DIR)

    # 创建数据集和数据加载器
    dataset = TensorDataset(emg_data, emg_label)

    # 划分训练集和验证集，比例为 8:2
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 处理数据不平衡，使用 WeightedRandomSampler
    train_labels = [label.item() for _, label in train_dataset]
    class_counts = Counter(train_labels)
    class_weights = [1.0 / class_counts[i] for i in range(num_classes)]
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # 数据加载器
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    }

    # 实例化模型、损失函数、优化器和学习率调度器
    model = CNNModel(num_channels=emg_data.shape[2], num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  # AdamW 优化器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)  # 余弦退火调度器

    # 训练模型
    start_time = time.time()
    history = train_model(model, dataloaders, criterion, optimizer, scheduler, NUM_EPOCHS)
    end_time = time.time()
    print(f'训练时间: {end_time - start_time:.2f} 秒')

    # 加载最佳模型并评估
    model.load_state_dict(torch.load('best_model.pth'))
    test_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    evaluate_model(model, test_loader, criterion)


if __name__ == '__main__':
    main()
