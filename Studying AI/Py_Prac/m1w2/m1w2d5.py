# 学习内容：梯度、链式法则、梯度下降
# 实践任务：NumPy 实现线性回归梯度下降

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# 让中文输出在 UTF-8 终端中正常显示（Windows 默认用 GBK，否则会乱码）
sys.stdout.reconfigure(encoding='utf-8')


def forward(x, w, b):
    """模型：y = w·x + b。"""
    return w * x + b


def mse(y_pred, y):
    """均方误差损失：L = mean((ŷ - y)²)。"""
    return np.mean((y_pred - y) ** 2)


def gradients(x, y, y_pred):
    """用链式法则求 L 对 w、b 的梯度：

        L   = mean((ŷ - y)²)
        ∂L/∂ŷ = 2(ŷ - y)/n         （外层：平方的导数）
        ∂ŷ/∂w = x， ∂ŷ/∂b = 1      （内层：线性函数的导数）
        ∴ ∂L/∂w = ∂L/∂ŷ · ∂ŷ/∂w = 2·mean((ŷ - y)·x)
          ∂L/∂b = ∂L/∂ŷ · ∂ŷ/∂b = 2·mean(ŷ - y)
    """
    error = y_pred - y
    dw = 2 * np.mean(error * x)
    db = 2 * np.mean(error)
    return dw, db


def main():
    rng = np.random.default_rng(42)

    # ---- 1. 造数据：真实 y = 3x + 2 + 噪声 ----
    w_true, b_true = 3.0, 2.0
    n = 100
    x = rng.uniform(0.0, 1.0, n)
    y = w_true * x + b_true + rng.normal(0.0, 0.2, n)

    # ---- 2. 梯度下降 ----
    w, b = 0.0, 0.0            # 初始参数
    w0, b0 = w, b              # 记录初始值，画图用
    lr = 0.1                   # 学习率
    epochs = 300

    loss_hist, w_hist, b_hist = [], [], []

    for _ in range(epochs):
        y_pred = forward(x, w, b)
        loss = mse(y_pred, y)
        dw, db = gradients(x, y, y_pred)
        w -= lr * dw           # 沿负梯度方向更新
        b -= lr * db
        loss_hist.append(loss)
        w_hist.append(w)
        b_hist.append(b)

    print(f'真实参数：w={w_true}, b={b_true}')
    print(f'学到的参数：w={w:.4f}, b={b:.4f}')
    print(f'最终损失 MSE = {loss_hist[-1]:.6f}（噪声方差约 {0.2**2:.4f}）')

    # ---- 3. 画图 ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) 数据 + 拟合线（灰虚线=初始，红实线=拟合结果）
    ax = axes[0]
    ax.scatter(x, y, s=20, alpha=0.6, color='tab:blue', label='data')
    xs = np.linspace(0, 1, 50)
    ax.plot(xs, forward(xs, w0, b0), color='gray', linestyle='--',
            label=f'initial (w={w0}, b={b0})')
    ax.plot(xs, forward(xs, w, b), color='tab:red', linewidth=2, label='fitted')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Linear Regression Fit')
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)

    # (b) 损失曲线（对数坐标，看指数下降）
    ax = axes[1]
    ax.semilogy(loss_hist, color='tab:red')
    ax.set_xlabel('epoch')
    ax.set_ylabel('MSE loss')
    ax.set_title('Loss Curve')
    ax.grid(True, linestyle='--', alpha=0.5)

    # (c) 参数轨迹（w、b 收敛到真实值）
    ax = axes[2]
    ax.plot(w_hist, label='w', color='tab:blue')
    ax.plot(b_hist, label='b', color='tab:orange')
    ax.axhline(w_true, color='tab:blue', linestyle='--', linewidth=1, alpha=0.6)
    ax.axhline(b_true, color='tab:orange', linestyle='--', linewidth=1, alpha=0.6)
    ax.set_xlabel('epoch')
    ax.set_ylabel('value')
    ax.set_title('Parameter Trajectory')
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    out_path = Path(__file__).parent / 'linear_regression_gd.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'图片已保存为 {out_path}')
    plt.show()


if __name__ == '__main__':
    main()
