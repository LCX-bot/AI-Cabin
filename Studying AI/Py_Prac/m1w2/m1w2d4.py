# 学习内容：概率分布、均值、方差
# 实践任务：采样高斯分布并画直方图

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# 让中文输出在 UTF-8 终端中正常显示（Windows 默认用 GBK，否则会乱码）
sys.stdout.reconfigure(encoding='utf-8')


def gaussian_pdf(x, mu, sigma):
    """高斯分布 N(μ, σ²) 的概率密度函数。"""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def main():
    rng = np.random.default_rng(42)

    mu, sigma = 0.0, 1.0     # 真实均值、标准差
    n = 10000                # 采样数量

    # ---- 1. 采样 ----
    samples = rng.normal(mu, sigma, n)

    # ---- 2. 样本均值 / 样本方差（无偏估计，除以 n-1）----
    sample_mean = samples.mean()
    sample_var = samples.var(ddof=1)
    sample_std = samples.std(ddof=1)

    print(f'真实均值 μ = {mu}    样本均值 = {sample_mean:.4f}')
    print(f'真实方差 σ² = {sigma ** 2}    样本方差 = {sample_var:.4f}')
    print(f'真实标准差 σ = {sigma}    样本标准差 = {sample_std:.4f}')

    # ---- 3. 经验法则 68-95-99.7 验证 ----
    print('\n经验法则验证（样本落在均值±kσ 内的比例）：')
    for k, theo in zip([1, 2, 3], [0.6827, 0.9545, 0.9973]):
        frac = np.mean(np.abs(samples - sample_mean) <= k * sample_std)
        print(f'  k={k}: {frac:.4f}（理论 {theo:.4f}）')

    # ---- 4. 画直方图 + 理论 PDF + 均值/σ 竖线 ----
    fig, ax = plt.subplots(figsize=(9, 6))

    # density=True 把直方图归一化成"密度"，才能和 PDF 直接比较
    ax.hist(samples, bins=50, density=True, alpha=0.6, color='tab:blue',
            label=f'samples (n={n})')

    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
    ax.plot(x, gaussian_pdf(x, mu, sigma), color='tab:red', linewidth=2,
            label='theoretical PDF')

    # 竖线：黑色 = μ，橙色 = μ±σ，绿色 = μ±2σ
    for k, color in [(0, 'black'), (1, 'tab:orange'), (2, 'tab:green')]:
        ax.axvline(mu + k * sigma, color=color, linestyle='--', linewidth=1)
        ax.axvline(mu - k * sigma, color=color, linestyle='--', linewidth=1)

    ax.set_xlabel('value')
    ax.set_ylabel('density')
    ax.set_title('Gaussian Distribution: Histogram vs PDF')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)

    fig.tight_layout()
    out_path = Path(__file__).parent / 'gaussian_histogram.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'图片已保存为 {out_path}')
    plt.show()


if __name__ == '__main__':
    main()
