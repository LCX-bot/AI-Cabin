# 学习内容：GPU、batch、DataLoader
# 实践任务：把 MLP 训练搬到 GPU

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 让中文输出在 UTF-8 终端中正常显示（Windows 默认用 GBK，否则会乱码）
sys.stdout.reconfigure(encoding='utf-8')


# ============ 1. 模型定义 ============
class MLP(nn.Module):
    """三层 MLP：输入 1 维 → 隐藏层 → 隐藏层 → 输出 1 维。"""

    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


# ============ 2. 数据准备 ============
def make_data(n=5000, noise=0.05, seed=42):
    """生成 y = x² + 噪声，形状 (n, 1)。"""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n).astype(np.float32)
    y = (x ** 2 + rng.normal(0, noise, n).astype(np.float32)).astype(np.float32)
    return x.reshape(-1, 1), y.reshape(-1, 1)


# ============ 3. 单个 epoch 训练 ============
def train_one_epoch(model, loader, criterion, optimizer, device):
    """跑一个 epoch，返回按样本数加权的平均损失。"""
    model.train()
    total_loss, n_samples = 0.0, 0
    for xb, yb in loader:
        # 关键：使用 non_blocking=True，配合 pin_memory 加速 CPU→GPU 拷贝
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad()
        y_pred = model(xb)
        loss = criterion(y_pred, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        n_samples += xb.size(0)
    return total_loss / n_samples


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """验证集评估，返回平均损失。"""
    model.eval()
    total_loss, n_samples = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        y_pred = model(xb)
        loss = criterion(y_pred, yb)
        total_loss += loss.item() * xb.size(0)
        n_samples += xb.size(0)
    return total_loss / n_samples


# ============ 4. 主流程 ============
def main():
    torch.manual_seed(42)

    # ---- 4.1 选择设备 ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'PyTorch 版本: {torch.__version__}')
    print(f'使用设备: {device}')
    if device.type == 'cuda':
        print(f'GPU 型号: {torch.cuda.get_device_name(0)}')
        print(f'CUDA 版本: {torch.version.cuda}')
        print(f'显存总量: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    else:
        print('未检测到 GPU，将使用 CPU 训练。')

    # ---- 4.2 数据：切分 + 封装 DataLoader ----
    x_all, y_all = make_data(n=5000, noise=0.05)
    n_train = 4000
    x_train, y_train = x_all[:n_train], y_all[:n_train]
    x_val, y_val = x_all[n_train:], y_all[n_train:]

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))

    batch_size = 128
    # pin_memory=True 让数据先放在锁页内存中，配合 non_blocking=True 加速传输到 GPU
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        pin_memory=(device.type == 'cuda'), num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        pin_memory=(device.type == 'cuda'), num_workers=0
    )

    # ---- 4.3 模型、损失、优化器 ----
    model = MLP(hidden=64).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # ---- 4.4 训练循环 ----
    epochs = 200
    train_losses, val_losses = [], []
    best_val_loss, best_epoch = float('inf'), -1
    ckpt_path = Path(__file__).parent / 'best_model_gpu.pt'

    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            torch.save(model.state_dict(), ckpt_path)

        if epoch == 1 or epoch % 20 == 0:
            print(f'Epoch [{epoch:3d}/{epochs}]  '
                  f'train_loss={train_loss:.6f}  '
                  f'val_loss={val_loss:.6f}')
    t_end = time.perf_counter()
    print(f'\n训练完成，用时 {t_end - t_start:.2f} 秒，'
          f'最佳 epoch = {best_epoch}，最佳 val_loss = {best_val_loss:.6f}')

    # ---- 4.5 加载最佳模型并预测 ----
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    with torch.no_grad():
        x_grid = torch.linspace(-1.0, 1.0, 200).reshape(-1, 1).to(device)
        y_grid = model(x_grid).cpu().numpy()

    # ---- 4.6 绘图 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.semilogy(train_losses, label='train loss', color='tab:blue')
    ax.semilogy(val_losses, label='val loss', color='tab:orange')
    ax.axvline(best_epoch - 1, color='gray', linestyle='--', alpha=0.7,
               label=f'best epoch = {best_epoch}')
    ax.set_xlabel('epoch')
    ax.set_ylabel('MSE loss (log scale)')
    ax.set_title(f'Training on {device.type.upper()}')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    ax = axes[1]
    ax.scatter(x_train[:500], y_train[:500], s=8, alpha=0.4,
               color='tab:blue', label='train data (subset)')
    ax.scatter(x_val, y_val, s=8, alpha=0.4, color='tab:green', label='val data')
    ax.plot(x_grid.cpu().numpy(), y_grid, 'r-', linewidth=2, label='MLP prediction')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('MLP Fit of y = x²')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    out_path = Path(__file__).parent / 'gpu_training.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'图片已保存为 {out_path}')
    plt.show()


if __name__ == '__main__':
    main()