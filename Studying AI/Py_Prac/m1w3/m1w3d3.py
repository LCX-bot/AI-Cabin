# 学习内容：Optimizer、loss、训练循环
# 实践任务：写完整 train loop，保存 loss 曲线

import sys
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
    """两层 MLP：输入 1 维，隐藏层 hidden 个神经元，输出 1 维。"""

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
def make_data(n=500, noise=0.05, seed=42):
    """生成 y = x² + 噪声，形状 (n, 1)。"""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n).astype(np.float32)
    y = (x ** 2 + rng.normal(0, noise, n).astype(np.float32)).astype(np.float32)
    return x.reshape(-1, 1), y.reshape(-1, 1)


# ============ 3. 单个 epoch 的训练与评估 ============
def train_one_epoch(model, loader, criterion, optimizer, device):
    """跑一个 epoch 的训练，返回按样本数加权的平均损失。"""
    model.train()
    total_loss, n_samples = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)  # 数据搬到设备
        optimizer.zero_grad()                  # 1. 清空上一步的梯度
        y_pred = model(xb)                     # 2. 前向
        loss = criterion(y_pred, yb)           # 3. 计算损失
        loss.backward()                        # 4. 反向传播
        optimizer.step()                       # 5. 更新参数
        total_loss += loss.item() * xb.size(0)
        n_samples += xb.size(0)
    return total_loss / n_samples


@torch.no_grad()  # 评估时禁用梯度，节省内存和计算
def evaluate(model, loader, criterion, device):
    """在验证集上评估，返回平均损失。"""
    model.eval()
    total_loss, n_samples = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        y_pred = model(xb)
        loss = criterion(y_pred, yb)
        total_loss += loss.item() * xb.size(0)
        n_samples += xb.size(0)
    return total_loss / n_samples


# ============ 4. 主流程 ============
def main():
    torch.manual_seed(42)

    # 选择设备：有 GPU 就用 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # ---- 4.1 数据：切分训练/验证，封装 DataLoader ----
    x_all, y_all = make_data(n=500, noise=0.05)
    n_train = 400
    x_train, y_train = x_all[:n_train], y_all[:n_train]
    x_val, y_val = x_all[n_train:], y_all[n_train:]

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))

    batch_size = 32
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ---- 4.2 模型、损失函数、优化器、学习率调度器 ----
    model = MLP(hidden=64).to(device)
    criterion = nn.MSELoss()                                                   # 回归任务用 MSE
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    # ---- 4.3 完整训练循环 ----
    epochs = 500
    patience = 50                    # 早停：验证损失连续 patience 轮不下降就停止
    best_val_loss = float('inf')
    best_epoch = -1
    epochs_no_improve = 0

    train_losses, val_losses = [], []
    ckpt_path = Path(__file__).parent / 'best_model.pt'

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step()             # 每个 epoch 结束后更新学习率

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # 保存表现最好的模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1

        if epoch == 1 or epoch % 50 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch:3d}/{epochs}]  '
                  f'train_loss={train_loss:.6f}  '
                  f'val_loss={val_loss:.6f}  '
                  f'lr={lr_now:.2e}')

        # 早停
        if epochs_no_improve >= patience:
            print(f'早停于 epoch {epoch}，最佳 epoch = {best_epoch}，'
                  f'最佳 val_loss = {best_val_loss:.6f}')
            break

    # ---- 4.4 加载最佳模型并预测网格上的值 ----
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    with torch.no_grad():
        x_grid = torch.linspace(-1.0, 1.0, 200).reshape(-1, 1).to(device)
        y_grid = model(x_grid).cpu().numpy()

    # ---- 4.5 绘图：loss 曲线 + 拟合结果 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) 训练 / 验证 loss 曲线（对数坐标）
    ax = axes[0]
    ax.semilogy(train_losses, label='train loss', color='tab:blue')
    ax.semilogy(val_losses, label='val loss', color='tab:orange')
    ax.axvline(best_epoch - 1, color='gray', linestyle='--', alpha=0.7,
               label=f'best epoch = {best_epoch}')
    ax.set_xlabel('epoch')
    ax.set_ylabel('MSE loss (log scale)')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    # (b) 拟合结果
    ax = axes[1]
    ax.scatter(x_train, y_train, s=8, alpha=0.4, color='tab:blue', label='train data')
    ax.scatter(x_val, y_val, s=8, alpha=0.4, color='tab:green', label='val data')
    ax.plot(x_grid.cpu().numpy(), y_grid, 'r-', linewidth=2, label='MLP prediction')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('MLP Fit of y = x²')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    out_path = Path(__file__).parent / 'train_loop_loss.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'图片已保存为 {out_path}')
    plt.show()


if __name__ == '__main__':
    main()