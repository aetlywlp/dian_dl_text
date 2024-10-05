# 导入必要的库
import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from scipy.stats import mode
from sklearn.preprocessing import StandardScaler
from collections import Counter
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# 检查是否有可用的 GPU，如果有则使用 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义全局参数
DATA_DIR = '/media/aetly/Data/BaiduNetdiskDownload/sEMG_DeepLearning-master/datah5/'  # 数据目录
WINDOW_SIZE = 200  # 每个窗口包含的样本数
STEP_SIZE = 100  # 窗口滑动的步长
BATCH_SIZE = 128  # 批量大小
NUM_EPOCHS = 100  # 训练周期数
EARLY_STOPPING_PATIENCE = 10  # 早停的耐心次数
LEARNING_RATE = 0.001  # 学习率


def load_and_preprocess_data(data_dir):
    """
    加载并预处理数据，包括窗口化、标准化和标签处理。
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

            if restimulus.ndim != 1:
                print(f"文件 {file_path} 中的 'restimulus' 数据无法展平为一维，跳过该文件。")
                continue

            num_samples, num_channels = emg.shape
            num_windows = (num_samples - WINDOW_SIZE) // STEP_SIZE + 1
            if num_windows <= 0:
                print(f"文件 {file_path} 中的数据样本不足以生成一个窗口，跳过该文件。")
                continue

            # 使用滑动窗口方法获取数据窗口
            try:
                windows = np.lib.stride_tricks.sliding_window_view(emg, window_shape=(WINDOW_SIZE, num_channels))
                windows = windows[::STEP_SIZE, :, :]  # 步长滑动
                windows = windows.reshape(-1, WINDOW_SIZE, num_channels)  # 形状：(num_windows, window_size, num_channels)
            except Exception as e:
                print(f"窗口化数据时出错，文件: {file_path}, 错误: {e}")
                continue

            # 获取每个窗口的标签（众数）
            window_labels = []
            for i in range(len(windows)):
                labels_in_window = restimulus[i * STEP_SIZE:i * STEP_SIZE + WINDOW_SIZE]
                try:
                    window_label = mode(labels_in_window, axis=None).mode[0]
                except Exception as e:
                    print(f"计算众数时出错，文件: {file_path}, 窗口: {i}, 错误: {e}")
                    window_label = -1  # 设置一个无效标签
                window_labels.append(window_label)

            # 过滤掉无效标签
            window_labels = np.array(window_labels)
            valid_mask = window_labels != -1
            windows = windows[valid_mask]
            window_labels = window_labels[valid_mask]

            windowed_data_list.append(windows)
            windowed_label_list.append(window_labels)

    if not windowed_data_list:
        raise ValueError("没有可用的窗口化数据，请检查数据文件和窗口设置。")

    # 拼接所有窗口化数据和标签
    emg_data = np.concatenate(windowed_data_list, axis=0)
    emg_label = np.concatenate(windowed_label_list, axis=0)

    print('EMG 数据形状:', emg_data.shape)
    print('标签形状:', emg_label.shape)
    print('唯一的标签:', np.unique(emg_label))

    # 数据标准化，每个通道单独标准化
    num_samples, window_size, num_channels = emg_data.shape
    emg_data = emg_data.reshape(-1, num_channels)
    scaler = StandardScaler()
    emg_data = scaler.fit_transform(emg_data)
    emg_data = emg_data.reshape(num_samples, window_size, num_channels)

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

    # 计算类别权重
    label_counts = Counter(emg_label.numpy())
    print("标签计数:", label_counts)

    class_counts = np.array([label_counts.get(i, 0) for i in range(num_classes)])
    if np.any(class_counts == 0):
        print("警告: 某些类别没有样本，无法计算类别权重。")
        class_counts = class_counts + 1e-6  # 避免除以零

    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()  # 归一化
    class_weights = torch.FloatTensor(class_weights).to(device)

    return emg_data, emg_label, num_classes, class_weights


class TransformerModel(nn.Module):
    """基于 Transformer 的模型，用于处理 sEMG 数据。"""

    def __init__(self, num_channels, num_classes, d_model=64, nhead=4, num_layers=3, dim_feedforward=256, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder = nn.Linear(num_channels, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                    dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, num_classes)

    def forward(self, src):
        # src: (batch_size, window_size, num_channels)
        src = self.encoder(src)  # (batch_size, window_size, d_model)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)  # (window_size, batch_size, d_model)
        output = self.transformer_encoder(src)  # (window_size, batch_size, d_model)
        output = output.mean(dim=0)  # (batch_size, d_model)
        output = self.decoder(output)  # (batch_size, num_classes)
        return output


class PositionalEncoding(nn.Module):
    """位置编码，用于 Transformer 模型。"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(device)
        div_term = torch.exp(torch.arange(0, d_model, 2).float().to(device) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs):
    """训练模型并使用早停策略。"""
    best_acc = 0.0
    early_stopping_counter = 0

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
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)  # (batch_size, window_size, num_channels)
                labels = labels.to(device)

                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)  # (batch_size, num_classes)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # 训练阶段反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计损失和准确率
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # 记录损失和准确率
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                # 调整学习率
                scheduler.step(epoch_loss)

                # 深拷贝模型
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
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


def evaluate_model(model, dataloaders, criterion, mode='test'):
    """
    评估模型的函数。
    """
    model.eval()  # 设置模型为评估模式
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():  # 不计算梯度，节省内存和计算
        for inputs, labels in dataloaders[mode]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss / len(dataloaders[mode].dataset)
    total_acc = running_corrects.double() / len(dataloaders[mode].dataset)

    if mode == 'test':
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
    emg_data, emg_label, num_classes, class_weights = load_and_preprocess_data(DATA_DIR)

    # 创建数据集和数据加载器
    dataset = TensorDataset(emg_data, emg_label)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False),
        'test': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)  # 使用验证集作为测试集
    }

    # 实例化模型、损失函数、优化器和学习率调度器
    model = TransformerModel(num_channels=emg_data.shape[2], num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # 训练模型
    start_time = time.time()
    history = train_model(model, dataloaders, criterion, optimizer, scheduler, NUM_EPOCHS)
    end_time = time.time()
    print(f'训练时间: {end_time - start_time:.2f} 秒')

    # 加载最佳模型并评估
    model.load_state_dict(torch.load('best_model.pth'))
    evaluate_model(model, dataloaders, criterion, mode='test')


if __name__ == '__main__':
    main()
