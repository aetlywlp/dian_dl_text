# 导入必要的库
import os  # 文件和路径操作
import glob  # 文件名模式匹配
import h5py  # 处理 HDF5 文件格式
import numpy as np  # 数值计算库
import torch  # PyTorch 深度学习框架
from torch.utils.data import DataLoader, TensorDataset, random_split, WeightedRandomSampler, Subset  # 数据处理和加载工具
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 激活函数等
import matplotlib.pyplot as plt  # 数据可视化
import time  # 时间相关的函数
from scipy.stats import mode  # 统计函数，用于计算众数
from sklearn.preprocessing import StandardScaler  # 数据预处理工具，用于标准化
from sklearn.model_selection import train_test_split  # 用于数据的分层拆分
from collections import Counter  # 计数器，用于统计元素出现的次数
import warnings  # 警告控制
from torch.cuda.amp import autocast, GradScaler  # 混合精度训练

# 忽略警告
warnings.filterwarnings("ignore")

# 检查是否有可用的 GPU，如果有则使用 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义全局参数
DATA_DIR = '/media/aetly/Data/BaiduNetdiskDownload/sEMG_DeepLearning-master/datah5'  # 数据目录，存放 EMG 数据文件
WINDOW_SIZE = 200  # 每个窗口包含的样本数，用于将时间序列数据分割成固定长度的片段
STEP_SIZE = 100  # 窗口滑动的步长，控制窗口之间的重叠程度
BATCH_SIZE = 256  # 增大批量大小，稳定梯度估计
NUM_EPOCHS = 100  # 增加训练周期数，以达到更好的收敛
EARLY_STOPPING_PATIENCE = 25  # 早停的耐心次数，避免过拟合
LEARNING_RATE = 0.001  # 初始学习率
LABEL_SMOOTHING = 0.1  # 标签平滑参数

# 定义 Focal Loss
class FocalLoss(nn.Module):
    """Focal Loss, 平衡类别不平衡问题。"""

    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 类别权重
        self.gamma = gamma  # 调节因子，控制对困难样本的关注度
        self.reduction = reduction  # 损失的计算方式：'none' | 'mean' | 'sum'

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # 计算交叉熵损失
        pt = torch.exp(-ce_loss)  # p_t 是模型对正确类别的预测概率
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss  # 计算焦点损失

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 定义带标签平滑的 CrossEntropyLoss
class LabelSmoothingCrossEntropy(nn.Module):
    """带标签平滑的交叉熵损失函数。"""

    def __init__(self, smoothing=0.0):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        log_probs = F.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# 数据加载和预处理函数
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

    windowed_data_list = []  # 存储所有窗口化的数据
    windowed_label_list = []  # 存储所有窗口对应的标签

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
                    print(f"窗口 {i} 的标签为空，跳过该窗口。")
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
    emg_label = np.concatenate(windowed_label_list, axis=0).astype(int).squeeze()  # 形状：(总窗口数,)

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

# 定义残差块
class ResidualBlock(nn.Module):
    """残差块，用于构建深层 CNN 模型。"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, downsample=False, dropout=0.3):
        super(ResidualBlock, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)  # 添加 Dropout
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=1, bias=False)
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
        out = self.dropout(out)  # 应用 Dropout
        out = self.bn2(self.conv2(out))
        if self.downsample_layer is not None:
            identity = self.downsample_layer(x)
        out += identity
        out = self.relu(out)
        return out

# 定义 CNN 模型
class CNNModel(nn.Module):
    """基于 CNN 的模型，包含多个残差块和 Dropout。"""

    def __init__(self, num_channels, num_classes):
        super(CNNModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.layer2 = ResidualBlock(64, 128, downsample=True, dropout=0.3)
        self.layer3 = ResidualBlock(128, 256, downsample=True, dropout=0.3)
        self.layer4 = ResidualBlock(256, 512, downsample=True, dropout=0.3)
        self.layer5 = ResidualBlock(512, 512, downsample=False, dropout=0.3)  # 增加一个残差块
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 自适应平均池化，输出长度为1
        self.fc = nn.Linear(512, num_classes)  # 全连接层，输出为类别数
        self.dropout = nn.Dropout(0.5)  # 添加更高的 Dropout

    def forward(self, x):
        # x 形状: (batch_size, window_size, num_channels)
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, num_channels, window_size)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)  # 通过第五个残差块
        x = self.avg_pool(x)  # 形状: (batch_size, 512, 1)
        x = x.squeeze(-1)  # 形状: (batch_size, 512)
        x = self.dropout(x)  # 应用 Dropout
        x = self.fc(x)  # 全连接层输出，形状: (batch_size, num_classes)
        return x

# 定义训练函数
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
                model.eval()   # 设置模型为评估模式，禁用 Dropout 和 BatchNorm

            running_loss = 0.0  # 累计损失
            running_corrects = 0  # 累计正确预测的数量

            # 遍历数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()  # 清零梯度

                # 混合精度训练
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                if phase == 'train':
                    scaler.scale(loss).backward()  # 反向传播，计算梯度
                    scaler.step(optimizer)  # 更新参数
                    scaler.update()  # 更新 scaler 状态

                # 统计损失和准确率
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)  # 计算平均损失
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)  # 计算准确率
            epoch_acc_value = epoch_acc.item()

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc_value:.4f}")

            # 记录损失和准确率
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc_value)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc_value)
                # 调整学习率
                scheduler.step(epoch_loss)  # 根据验证集损失调整学习率

                # 早停策略
                if epoch_acc_value > best_acc:
                    best_acc = epoch_acc_value  # 更新最佳准确率
                    torch.save(model.state_dict(), 'best_model.pth')  # 保存最佳模型
                    early_stopping_counter = 0  # 重置早停计数器
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
                        print("早停触发，停止训练")
                        plot_history(history)
                        return history  # 提前结束训练

    print("训练完成！")
    # 绘制损失和准确率曲线
    plot_history(history)
    return history

# 定义评估函数
def evaluate_model(model, dataloader, criterion):
    """评估模型。"""
    model.eval()  # 设置模型为评估模式
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():  # 不计算梯度，节省内存和计算
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss / len(dataloader.dataset)
    total_acc = running_corrects.double() / len(dataloader.dataset)

    print(f'测试集损失: {total_loss:.4f} | 测试集准确率: {total_acc:.4f}')

    return total_loss, total_acc.item()

# 定义可视化函数
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

# 主函数
def main():
    # 加载和预处理数据
    emg_data, emg_label, num_classes = load_and_preprocess_data(DATA_DIR)

    # 创建数据集
    dataset = TensorDataset(emg_data, emg_label)

    # 将数据集转换为 NumPy 数组用于 stratified 分割
    labels = emg_label.numpy()

    # 使用 sklearn 的 train_test_split 进行分层拆分
    train_indices, val_indices = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    # 处理数据不平衡，使用 WeightedRandomSampler
    train_labels = labels[train_indices]
    class_counts = Counter(train_labels)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # 数据加载器
    dataloaders = {
        'train': DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True),
        'val': DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    }

    # 实例化模型、损失函数、优化器和学习率调度器
    model = CNNModel(num_channels=emg_data.shape[2], num_classes=num_classes).to(device)
    # 使用 Focal Loss
    criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')
    # 或者使用带标签平滑的 CrossEntropyLoss
    # criterion = LabelSmoothingCrossEntropy(smoothing=LABEL_SMOOTHING)
    # 使用 AdamW 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # 使用 OneCycleLR 学习率调度器
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE*10,  # OneCycleLR 需要设置一个学习率的上限
        steps_per_epoch=len(dataloaders['train']),
        epochs=NUM_EPOCHS,
        anneal_strategy='linear'
    )

    # 训练模型
    start_time = time.time()
    history = train_model(model, dataloaders, criterion, optimizer, scheduler, NUM_EPOCHS)
    end_time = time.time()
    print(f'训练时间: {end_time - start_time:.2f} 秒')

    # 加载最佳模型并评估
    model.load_state_dict(torch.load('best_model.pth'))
    test_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    evaluate_model(model, test_loader, criterion)

if __name__ == '__main__':
    main()
