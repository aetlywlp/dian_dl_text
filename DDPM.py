import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm
import os

# =======================
# 1. 设置设备
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# =======================
# 2. 数据准备
# =======================
def prepare_data(selected_labels=[1, 3, 5], batch_size=64):
    """
    加载并准备Fashion-MNIST数据集，仅保留指定标签的图像。

    :param selected_labels: 要保留的标签列表。
    :param batch_size: 数据加载器的批次大小。
    :return: 训练和测试数据加载器。
    """
    # 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize((0.5,), (0.5,))  # 归一化到[-1, 1]
    ])

    # 加载Fashion-MNIST数据集
    train_dataset = datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform)

    # 选择特定类别进行训练
    def filter_dataset(dataset, selected_labels):
        indices = [i for i, label in enumerate(dataset.targets) if label in selected_labels]
        dataset.data = dataset.data[indices]
        dataset.targets = dataset.targets[indices]
        return dataset

    train_dataset = filter_dataset(train_dataset, selected_labels)
    test_dataset = filter_dataset(test_dataset, selected_labels)

    # 定义数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# =======================
# 3. 定义DDPM的参数和过程
# =======================
def get_ddpm_parameters(T=100, beta_start=0.0001, beta_end=0.02, device='cuda'):
    """
    计算DDPM所需的参数。

    :param T: 扩散步数。
    :param beta_start: beta的起始值。
    :param beta_end: beta的结束值。
    :param device: 设备（'cuda'或'cpu'）。
    :return: 包含所有预计算参数的字典。
    """
    betas = torch.linspace(beta_start, beta_end, T).to(device)  # (T,)
    alphas = 1.0 - betas  # (T,)
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # (T,)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), alphas_cumprod[:-1]], dim=0)  # (T,)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)  # (T,)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)  # (T,)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)  # (T,)

    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'sqrt_recip_alphas': sqrt_recip_alphas
    }


def q_sample(x_start, t, noise=None, params=None):
    """
    前向扩散过程：给定原始图像x_start，扩散步数t，添加噪声得到x_t。

    :param x_start: 原始图像，形状：(batch_size, 1, 28, 28)
    :param t: 扩散步数，形状：(batch_size,)
    :param noise: 添加的噪声，形状：(batch_size, 1, 28, 28)，如果为None则自动生成
    :param params: 预计算的DDPM参数字典
    :return: 含噪图像x_t，形状：(batch_size, 1, 28, 28)
    """
    if noise is None:
        noise = torch.randn_like(x_start).to(x_start.device)
    sqrt_alpha_cumprod_t = params['sqrt_alphas_cumprod'][t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod_t = params['sqrt_one_minus_alphas_cumprod'][t].view(-1, 1, 1, 1)
    return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise


def p_sample(model, x, t, params):
    """
    反向采样一步：根据模型预测的噪声，计算x_{t-1}。

    :param model: 去噪模型，预测噪声
    :param x: 当前含噪图像x_t，形状：(batch_size, 1, 28, 28)
    :param t: 当前扩散步数t，形状：(batch_size,)
    :param params: 预计算的DDPM参数字典
    :return: 下一步的含噪图像x_{t-1}，形状：(batch_size, 1, 28, 28)
    """
    betas_t = params['betas'][t].view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod_t = params['sqrt_one_minus_alphas_cumprod'][t].view(-1, 1, 1, 1)
    sqrt_recip_alphas_t = params['sqrt_recip_alphas'][t].view(-1, 1, 1, 1)

    # 预测噪声
    epsilon_theta = model(x, t)

    # 计算x0的估计值
    x0_est = (x - betas_t * epsilon_theta) / params['sqrt_alphas_cumprod'][t].view(-1, 1, 1, 1)

    # 计算后验均值
    posterior_mean = (
            sqrt_recip_alphas_t * x
            - (betas_t / sqrt_one_minus_alpha_cumprod_t) * epsilon_theta
    )

    # 采样x_{t-1}
    # 当t == 0时，不添加噪声
    t_max = t.max().item()
    if t_max == 0:
        return posterior_mean
    else:
        noise = torch.randn_like(x).to(x.device)
        posterior_variance = params['betas'][t].view(-1, 1, 1, 1)
        return posterior_mean + torch.sqrt(posterior_variance) * noise


# =======================
# 4. RNN去噪网络的实现
# =======================
class RNNDenoiser(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, embed_dim, num_classes):
        """
        RNN去噪网络，用于预测扩散过程中的噪声。

        :param input_dim: 输入特征维度（每行像素数）。
        :param hidden_dim: RNN隐藏状态维度。
        :param num_layers: RNN层数。
        :param embed_dim: 时间步嵌入维度。
        :param num_classes: 扩散步数（T）。
        """
        super(RNNDenoiser, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # 时间步嵌入
        self.time_embed = nn.Embedding(num_classes, embed_dim)

        # RNN层（LSTM）
        self.lstm = nn.LSTM(input_dim + embed_dim, hidden_dim, num_layers, batch_first=True)

        # 输出层，预测噪声
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x, t):
        """
        前向传播函数。

        :param x: 含噪图像，形状：(batch_size, 1, 28, 28)
        :param t: 扩散步数，形状：(batch_size,)
        :return: 预测的噪声，形状：(batch_size, 1, 28, 28)
        """
        batch_size, channels, height, width = x.size()
        assert channels == 1, "仅支持单通道图像。"

        # 将图像转换为序列（batch_size, seq_len=28, input_dim=28）
        x = x.squeeze(1)  # (batch_size, 28, 28)

        # 时间步嵌入
        t_embed = self.time_embed(t)  # (batch_size, embed_dim)
        t_embed = t_embed.unsqueeze(1).repeat(1, height, 1)  # (batch_size, 28, embed_dim)

        # 拼接图像序列与时间步嵌入
        x = torch.cat([x, t_embed], dim=-1)  # (batch_size, 28, 28 + embed_dim)

        # RNN前向传播
        out, _ = self.lstm(x)  # (batch_size, 28, hidden_dim)

        # 预测噪声
        out = self.output_layer(out)  # (batch_size, 28, 28)

        # 重新转换为图像格式
        out = out.unsqueeze(1)  # (batch_size, 1, 28, 28)
        return out


# =======================
# 5. 训练过程
# =======================
def train_ddpm(model, train_loader, params, device, num_epochs=20, learning_rate=1e-3, save_interval=5):
    """
    训练DDPM模型。

    :param model: 去噪模型
    :param train_loader: 训练数据加载器
    :param params: 预计算的DDPM参数字典
    :param device: 设备（'cuda'或'cpu'）
    :param num_epochs: 训练轮数
    :param learning_rate: 学习率
    :param save_interval: 每隔多少个epoch保存一次模型
    :return: 训练损失列表
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # 创建保存模型的目录
    os.makedirs('saved_models', exist_ok=True)

    train_losses = []
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for batch_idx, (x, _) in enumerate(progress_bar):
            x = x.to(device)  # (batch_size, 1, 28, 28)

            # 随机选择扩散步数t
            t = torch.randint(0, params['betas'].size(0), (x.size(0),)).to(device)  # (batch_size,)

            # 添加噪声，得到x_t
            noise = torch.randn_like(x).to(device)
            x_t = q_sample(x, t, noise, params)  # (batch_size, 1, 28, 28)

            # 预测噪声
            epsilon_theta = model(x_t, t)  # (batch_size, 1, 28, 28)

            # 计算损失
            loss = criterion(epsilon_theta, noise)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": loss.item()})

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.6f}")

        # 每隔save_interval个epoch保存一次模型
        if epoch % save_interval == 0:
            torch.save(model.state_dict(), f"saved_models/rnn_ddpm_epoch_{epoch}.pth")
            print(f"模型已保存至 saved_models/rnn_ddpm_epoch_{epoch}.pth")

    return train_losses


# =======================
# 6. 生成图像
# =======================
def generate_images(model, params, device, num_images=16, save_path='generated_images.png'):
    """
    使用训练好的模型生成图像。

    :param model: 训练好的去噪模型
    :param params: 预计算的DDPM参数字典
    :param device: 设备（'cuda'或'cpu'）
    :param num_images: 生成图像的数量
    :param save_path: 保存生成图像的路径
    """
    model.eval()
    with torch.no_grad():
        # 初始化x_T为标准正态分布
        x = torch.randn(num_images, 1, 28, 28).to(device)

        # 逐步反向采样
        for t in reversed(range(params['betas'].size(0))):
            t_batch = torch.full((num_images,), t, dtype=torch.long).to(device)  # (num_images,)
            x = p_sample(model, x, t_batch, params)

        # 反归一化
        x = x.clamp(-1, 1)
        x = (x + 1) / 2
        x = x.cpu()

        # 可视化生成的图像
        grid = torchvision.utils.make_grid(x[:num_images], nrow=int(math.sqrt(num_images)))
        np_grid = grid.numpy()
        plt.figure(figsize=(8, 8))
        plt.imshow(np.transpose(np_grid, (1, 2, 0)), cmap='gray')
        plt.title('生成的图像')
        plt.axis('off')
        plt.savefig(save_path)
        plt.show()
        print(f"生成的图像已保存至 {save_path}")


# =======================
# 7. 绘制训练损失曲线
# =======================
def plot_losses(train_losses, save_path='training_loss_curve.png'):
    """
    绘制训练损失曲线。

    :param train_losses: 训练损失列表
    :param save_path: 保存损失曲线的路径
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练损失曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()
    print(f"训练损失曲线已保存至 {save_path}")


# =======================
# 8. 主函数
# =======================
def main():
    # 设置随机种子以便复现
    torch.manual_seed(42)
    if device == 'cuda':
        torch.cuda.manual_seed(42)

    # 1. 准备数据
    selected_labels = [1, 3, 5]  # 可以根据需要调整
    batch_size = 64
    train_loader, test_loader = prepare_data(selected_labels=selected_labels, batch_size=batch_size)

    # 2. 可视化部分训练图像
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    grid = torchvision.utils.make_grid(images[:16], nrow=4)
    np_grid = grid.numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(np.transpose(np_grid, (1, 2, 0)), cmap='gray')
    plt.title('真实图像')
    plt.axis('off')
    plt.show()
    print('真实标签:', labels[:16].tolist())

    # 3. 获取DDPM参数
    T = 1000  # 扩散步数
    params = get_ddpm_parameters(T=T, beta_start=0.0001, beta_end=0.02, device=device)

    # 4. 实例化RNN去噪网络
    input_dim = 28  # 每行像素数
    hidden_dim = 128
    num_layers = 2
    embed_dim = 32
    num_classes = params['betas'].size(0)  # 扩散步数

    model = RNNDenoiser(input_dim, hidden_dim, num_layers, embed_dim, num_classes).to(device)
    print(model)

    # 5. 训练模型
    num_epochs = 20  # 根据计算资源调整
    learning_rate = 1e-3
    save_interval = 5  # 每隔多少个epoch保存一次模型
    train_losses = train_ddpm(model, train_loader, params, device, num_epochs=num_epochs, learning_rate=learning_rate,
                              save_interval=save_interval)

    # 6. 绘制训练损失曲线
    plot_losses(train_losses)

    # 7. 生成并展示图像
    generate_images(model, params, device, num_images=16, save_path='generated_images.png')


if __name__ == '__main__':
    main()
