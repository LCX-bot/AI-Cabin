# 学习内容：Matplotlib 进阶 —— 多实验对比、平滑窗口、子图布局
# 实践任务：模拟多次训练的 reward 曲线（带置信带）并保存 PNG

import sys

import numpy as np
import matplotlib.pyplot as plt

# 让中文输出在 UTF-8 终端中正常显示（Windows 默认用 GBK，否则会乱码）
sys.stdout.reconfigure(encoding='utf-8')

# 设置随机种子，保证结果可复现
np.random.seed(42)

# ---------- 1. 生成模拟数据 ----------
EPISODES = 100        # episode 数量
N_RUNS = 5            # 独立训练实验次数
WINDOWS = [1, 5, 20]  # 平滑窗口（1 表示原始曲线，不平滑）

episodes = np.arange(1, EPISODES + 1)


def simulate_run(episodes, noise_scale=5.0):
    """模拟一次训练：真实曲线 + 随机噪声。"""
    true = 50.0 * (1.0 - np.exp(-episodes / 30.0))  # 指数上升的真实曲线
    noise = np.random.normal(0, noise_scale, size=episodes.shape)
    return true + noise


def moving_average(x, window):
    """计算滑动平均，返回的数组长度比输入短 window-1。"""
    # 利用convolve卷积函数实现滑动平均值计算
    return np.convolve(x, np.ones(window) / window, mode='valid')


# 多次实验的 reward，形状 (N_RUNS, EPISODES)
runs = np.array([simulate_run(episodes) for _ in range(N_RUNS)])
mean = runs.mean(axis=0)   # 各 episode 的平均值
std = runs.std(axis=0)     # 各 episode 的标准差

# ---------- 2. 绘图 ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图axes0：多次实验对比 + 均值置信带
ax = axes[0]
# enumerate：将runs转化成一个枚举对象，枚举索引返回给i，枚举元素返回给run
for i, run in enumerate(runs):
    ax.plot(episodes, run, alpha=0.3, linewidth=1, label=f'Run {i + 1}')
ax.plot(episodes, mean, color='tab:red', linewidth=2, label='Mean')
ax.fill_between(episodes, mean - std, mean + std,
                color='tab:red', alpha=0.2, label='±1 std')
ax.set_xlabel('Episode')
ax.set_ylabel('Reward')
ax.set_title('Multiple Training Runs')
ax.legend(fontsize=8)
ax.grid(True, linestyle='--', alpha=0.6)

# 右图axes1：散点图（按 episode 着色）+ 不同平滑窗口的折线图
ax = axes[1]
sc = ax.scatter(episodes[::5], mean[::5], c=episodes[::5], cmap='viridis',
                s=20, alpha=0.7, label='Raw mean (sampled)')
for w in WINDOWS:
    smoothed = moving_average(mean, w)
    ax.plot(episodes[w - 1:], smoothed, linewidth=2, label=f'window={w}')
ax.set_xlabel('Episode')
ax.set_ylabel('Reward')
ax.set_title('Smoothing Window Comparison')
ax.legend(fontsize=8)
ax.grid(True, linestyle='--', alpha=0.6)
fig.colorbar(sc, ax=ax, label='Episode')

# 整体标题 + 布局
fig.suptitle('Training Reward Curve Analysis', fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.95])

# ---------- 3. 保存图片 ----------
fig.savefig('reward_curve_advanced.png', dpi=150, bbox_inches='tight')
print('图片已保存为 reward_curve_advanced.png')

# 显示图形（可选）
plt.show()