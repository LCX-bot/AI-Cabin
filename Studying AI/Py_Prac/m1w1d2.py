# 学习内容：Matplotlib 折线图、散点图、保存图片
# 实践任务：模拟 reward 曲线并保存 PNG

import sys

import numpy as np
import matplotlib.pyplot as plt

# 让中文输出在 UTF-8 终端中正常显示（Windows 默认用 GBK，否则会乱码）
sys.stdout.reconfigure(encoding='utf-8')

# 设置随机种子，保证结果可复现
np.random.seed(42)

# 模拟训练过程：100 个 episode 的累计奖励
episodes = np.arange(1, 101)
# print(f"{episodes}")
true_reward = 50.0 * (1.0 - np.exp(-episodes / 30.0))  # 指数上升曲线
noise = np.random.normal(0, 2.0, size=episodes.shape)   # 随机噪声
reward = true_reward + noise                              # 实际观测奖励

# 可选：计算滑动平均（窗口大小为 10），让趋势更明显
window = 10
moving_avg = np.convolve(reward, np.ones(window)/window, mode='valid')
avg_episodes = episodes[window-1:]  # 滑动平均对应的 episode 序号

# 创建图形
plt.figure(figsize=(10, 6))

# 散点图：展示原始数据点
plt.scatter(episodes, reward, color='tab:blue', alpha=0.6, label='Raw reward')

# 折线图：展示滑动平均趋势
plt.plot(avg_episodes, moving_avg, color='tab:red', linewidth=2,
         label=f'Moving average (window={window})')

# 添加标签和标题
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Reward Curve')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# 保存图片到当前目录，设置清晰度
plt.savefig('reward_curve.png', dpi=150, bbox_inches='tight')
print("图片已保存为 reward_curve.png")

# 显示图形（可选）
plt.show()