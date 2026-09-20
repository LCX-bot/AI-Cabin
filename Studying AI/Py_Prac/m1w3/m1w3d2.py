# 学习内容：nn.Module、MLP
# 实践任务：写两层 MLP 拟合 y=x^2

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# 让中文输出在 UTF-8 终端中正常显示（Windows 默认用 GBK，否则会乱码）
sys.stdout.reconfigure(encoding='utf-8')


class MLP(nn.Module):
    """两层 MLP：输入 1 维，隐藏层 32 个神经元，输出 1 维。

    结构：Linear(1, 32) → ReLU → Linear(32, 1)
    """

    def __init__(self, hidden_size=32):
        super().__init__()  # 调用父类 nn.module 的 init 方法
        self.fc1 = nn.Linear(1, hidden_size)      # 第一层：输入到隐藏层
        self.act = nn.ReLU()                       # 激活函数
        self.fc2 = nn.Linear(hidden_size, 1)       # 第二层：隐藏层到输出

    def forward(self, x):
        """前向传播。"""
        h = self.act(self.fc1(x))   # 隐藏层
        y = self.fc2(h)             # 输出层
        return y


def main():
    # 设置随机种子，保证可复现
    torch.manual_seed(42)
    np.random.seed(42)

    # ---- 1. 生成数据：y = x² + 噪声 ----
    n = 200
    x = torch.linspace(-1.0, 1.0, n).reshape(-1, 1)   # 形状 (n, 1)
    y_true = x ** 2
    noise = torch.randn(n, 1) * 0.05                    # 小噪声
    y = y_true + noise

    # 上面完成了拟合目标的创建

    # ---- 2. 构建模型、损失函数和优化器 ----
    model = MLP(hidden_size=32) # 多层感知机类的实例化
    criterion = nn.MSELoss()    # 均方误差损失
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)   # 自适应矩估计优化器

    epochs = 2000
    loss_hist = []

    # ---- 3. 训练循环 ----
    for epoch in range(epochs):
        model.train()                   # 将模型设置为训练模式
        y_pred = model(x)               # 前向（调用模型的 forward 方法，传入输入张量 x，得到预测值 y_pred）
        loss = criterion(y_pred, y)     # 计算损失
        optimizer.zero_grad()           # 将所有可训练参数的梯度清零
        loss.backward()                 # 计算损失对所有可训练参数的梯度
        optimizer.step()                # 更新参数

        loss_hist.append(loss.item())

        if (epoch + 1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    # # 接在你的 main 函数末尾，plt.show() 之前
    # print("\n=== 模型参数 ===")
    # for name, param in model.state_dict().items():
    #     print(f"{name}: shape {param.shape}")
    #     print(param.numpy())
    #     print()

    # ---- 4. 绘制结果 ----
    model.eval()
    with torch.no_grad():
        y_hat = model(x).numpy()        # 模型预测值

    x_np = x.numpy()
    y_np = y.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) 拟合曲线对比
    ax = axes[0]
    ax.scatter(x_np, y_np, s=10, alpha=0.5, color='tab:blue', label='data (with noise)')
    ax.plot(x_np, y_hat, 'r-', linewidth=2, label='MLP prediction')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('MLP Fits y = x²')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    # (b) 损失曲线（对数坐标）
    ax = axes[1]
    ax.semilogy(loss_hist, color='tab:red')
    ax.set_xlabel('epoch')
    ax.set_ylabel('MSE loss')
    ax.set_title('Training Loss')
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    out_path = Path(__file__).parent / 'mlp_fit_square.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'图片已保存为 {out_path}')
    plt.show()


if __name__ == '__main__':
    main()