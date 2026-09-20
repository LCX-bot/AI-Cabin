# 学习内容：Gym 环境接口
# 实践任务：写 random_cartpole.py，记录 reward 曲线

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

# 让中文输出在 UTF-8 终端中正常显示（Windows 默认用 GBK，否则会乱码）
sys.stdout.reconfigure(encoding='utf-8')
# Windows 系统自带中文字体，优先用"微软雅黑"，它覆盖的汉字最全
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
# 负号在默认字体里会被渲染成方块，需要额外指定它用拉丁字体
plt.rcParams['axes.unicode_minus'] = False


# ============ 一、随机策略跑一个 episode ============
def run_one_episode(env, seed=None):
    """用随机策略跑一个 episode，返回该 episode 的总奖励和步数。

    参数：
        env  : Gym 环境
        seed : 可选，环境随机种子（同一 seed 下结果可复现）

    返回：
        total_reward : 该 episode 的累计奖励
        steps        : 持续了多少步
    """
    # reset 返回 (初始观测, 额外信息)；传 seed 让环境随机性可复现
    obs, info = env.reset(seed=seed)

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    # 循环直到 episode 结束（终止或截断）
    while not (terminated or truncated):
        # 随机策略：从动作空间中均匀采样一个动作
        action = env.action_space.sample()

        # step 返回 5 元组：观测、奖励、是否终止、是否截断、额外信息
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

    return total_reward, steps


# ============ 二、跑多次 episode，记录奖励曲线 ============
def run_random_policy(env_name='CartPole-v1', n_episodes=200, seed=42):
    """用随机策略跑 n_episodes 个 episode，返回奖励和步数序列。

    参数：
        env_name   : Gym 环境名
        n_episodes : episode 数量
        seed       : 随机种子

    返回：
        rewards : 每个 episode 的总奖励，形状 (n_episodes,)
        lengths : 每个 episode 的步数，形状 (n_episodes,)
    """
    # 创建环境
    env = gym.make(env_name)

    rewards = []
    lengths = []

    for i in range(n_episodes):
        # 每个 episode 用不同 seed，保证独立又整体可复现
        r, steps = run_one_episode(env, seed=seed + i)
        rewards.append(r)
        lengths.append(steps)

    env.close()
    return np.array(rewards), np.array(lengths)


# ============ 三、可视化 ============
def plot_results(rewards, lengths, window=20):
    """绘制奖励曲线、步数曲线和奖励分布。"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    episodes = np.arange(1, len(rewards) + 1)

    # ---- (a) 奖励曲线 + 滑动平均 ----
    ax = axes[0]
    ax.plot(episodes, rewards, color='tab:blue', alpha=0.35,
            label='单次 reward')
    # 滑动平均：让趋势更清楚
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode='valid')
        ax.plot(episodes[window - 1:], moving_avg, color='tab:red', lw=2,
                label=f'滑动平均 (窗口={window})')
    ax.axhline(rewards.mean(), color='gray', linestyle=':', lw=1.5,
               label=f'总体均值 = {rewards.mean():.1f}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total reward')
    ax.set_title('随机策略的奖励曲线')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)

    # ---- (b) 步数曲线 ----
    ax = axes[1]
    ax.plot(episodes, lengths, color='tab:green', alpha=0.5, label='单次步数')
    if len(lengths) >= window:
        moving_avg_l = np.convolve(lengths, np.ones(window) / window, mode='valid')
        ax.plot(episodes[window - 1:], moving_avg_l, color='darkgreen', lw=2,
                label=f'滑动平均 (窗口={window})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('每 episode 的步数')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)

    # ---- (c) 奖励分布直方图 ----
    ax = axes[2]
    ax.hist(rewards, bins=30, color='tab:purple', alpha=0.75,
            edgecolor='k', linewidth=0.4)
    ax.axvline(rewards.mean(), color='tab:red', linestyle='--', lw=2,
               label=f'均值 = {rewards.mean():.1f}')
    ax.set_xlabel('Total reward')
    ax.set_ylabel('频次')
    ax.set_title('奖励分布')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')

    fig.tight_layout()
    out_path = Path(__file__).parent / 'random_cartpole.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'图片已保存为 {out_path}')


# ============ 四、主流程 ============
def main():
    np.set_printoptions(precision=4, suppress=True)

    env_name = 'CartPole-v1'
    n_episodes = 200

    print('=' * 60)
    print('随机策略跑 CartPole')
    print('=' * 60)
    print(f'环境: {env_name}')
    print(f'episode 数: {n_episodes}\n')

    # ---- 先看看环境的结构 ----
    env = gym.make(env_name)
    print(f'观测空间: {env.observation_space}')
    print(f'  形状: {env.observation_space.shape}')
    print(f'  范围: low = {env.observation_space.low}')
    print(f'        high = {env.observation_space.high}')
    print(f'动作空间: {env.action_space}')
    print(f'  动作数: {env.action_space.n}')
    env.close()

    # ---- 跑随机策略 ----
    rewards, lengths = run_random_policy(env_name, n_episodes=n_episodes, seed=42)

    print(f'\n统计结果:')
    print(f'  奖励均值: {rewards.mean():.2f}')
    print(f'  奖励标准差: {rewards.std():.2f}')
    print(f'  奖励范围: [{rewards.min():.0f}, {rewards.max():.0f}]')
    print(f'  步数均值: {lengths.mean():.2f}')

    # ---- 可视化 ----
    plot_results(rewards, lengths, window=20)

    plt.show()


if __name__ == '__main__':
    main()