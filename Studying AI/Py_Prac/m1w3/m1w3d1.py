# 学习内容：Tensor、autograd
# 实践任务：验证 y=(wx+b)^2 的梯度

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

# 让中文输出在 UTF-8 终端中正常显示（Windows 默认用 GBK，否则会乱码）
sys.stdout.reconfigure(encoding='utf-8')


def forward(x, w, b):
    """前向计算：y = (w·x + b)²"""
    return (w * x + b) ** 2


def manual_gradients(x, w, b):
    """手动推导的梯度（将 x 视为常数，w,b 为变量）：

    令 u = w·x + b，则 y = u²
    ∂y/∂u = 2u
    ∂u/∂w = x, ∂u/∂b = 1
    所以：
        ∂y/∂w = 2u·x = 2(wx+b)·x
        ∂y/∂b = 2u·1 = 2(wx+b)
    """
    u = w * x + b
    dw = 2 * u * x
    db = 2 * u
    return dw, db


def main():
    torch.manual_seed(42)

    # 固定参数（真实值），x 取一系列值，验证不同输入下梯度是否正确
    # 创建张量
    w_val, b_val = 3.0, 4.0
    w = torch.tensor(w_val, dtype=torch.float32, requires_grad=True)
    b = torch.tensor(b_val, dtype=torch.float32, requires_grad=True)

    # 生成测试点 x
    xs = torch.linspace(-3.0, 3.0, 7, requires_grad=False)
    auto_dw = []
    auto_db = []
    manual_dw = []
    manual_db = []

    print(f"固定参数：w = {w_val}, b = {b_val}\n")
    print("x\t自动∂y/∂w\t手动∂y/∂w\t自动∂y/∂b\t手动∂y/∂b")
    for x in xs:
        # 前向计算（每次都要创建新图，因为 backward 后会释放图）
        y = forward(x, w, b)
        # 自动求导
        y.backward()
        # 取出梯度（注意梯度会累加，所以每次循环需要清零）
        g_w = w.grad.item()
        g_b = b.grad.item()
        auto_dw.append(g_w)
        auto_db.append(g_b)
        # 手动计算（使用当前 x 的数值，不需要梯度）
        m_w, m_b = manual_gradients(x.item(), w_val, b_val)
        manual_dw.append(m_w)
        manual_db.append(m_b)

        print(f"{x.item():.1f}\t{g_w:.4f}\t\t{m_w:.4f}\t\t{g_b:.4f}\t\t{m_b:.4f}")

        # 梯度清零，防止下一次累加
        w.grad.zero_()
        b.grad.zero_()

    # 断言所有点的梯度一致
    assert np.allclose(auto_dw, manual_dw, atol=1e-6), "∂y/∂w 不一致！"
    assert np.allclose(auto_db, manual_db, atol=1e-6), "∂y/∂b 不一致！"
    print("\n✅ 所有测试点的自动梯度与手动梯度一致，验证通过！")

    # ---- 可视化对比 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    xs_np = xs.numpy()
    ax = axes[0]
    ax.scatter(xs_np, auto_dw, color='tab:blue', label='autograd ∂y/∂w', zorder=5)
    ax.plot(xs_np, manual_dw, 'r--', label='manual ∂y/∂w')
    ax.set_xlabel('x')
    ax.set_ylabel('gradient')
    ax.set_title('∂y/∂w vs x')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    ax = axes[1]
    ax.scatter(xs_np, auto_db, color='tab:green', label='autograd ∂y/∂b', zorder=5)
    ax.plot(xs_np, manual_db, 'r--', label='manual ∂y/∂b')
    ax.set_xlabel('x')
    ax.set_ylabel('gradient')
    ax.set_title('∂y/∂b vs x')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    out_path = Path(__file__).parent / 'gradient_check.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'图片已保存为 {out_path}')
    plt.show()


if __name__ == '__main__':
    main()