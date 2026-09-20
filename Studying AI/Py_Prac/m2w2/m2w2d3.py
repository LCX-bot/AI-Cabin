# 学习内容：Q-learning 更新
# 实践任务：FrozenLake Q-learning

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


# ============ 一、Q-learning 训练 ============
def q_learning(env, n_episodes=5000, alpha=0.1, gamma=0.99,
               epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995,
               seed=42, track_every=100):
    """Q-learning 训练。

    核心更新：
        Q(s, a) ← Q(s, a) + α · [r + γ·max_a' Q(s', a') - Q(s, a)]
    """
    rng = np.random.default_rng(seed)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))
    epsilon = epsilon_start

    rewards_per_episode = []
    success_per_episode = []
    history = []

    for ep in range(n_episodes):
        s, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        done = False

        while not done:
            # ---- ε-greedy 选动作 ----
            if rng.random() < epsilon:
                a = int(rng.integers(n_actions))
            else:
                a = int(np.argmax(Q[s]))

            # ---- 与环境交互一步 ----
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            # ---- Q-learning 更新 ----
            # 真正终止时未来价值为 0；被截断时仍要 bootstrap
            if terminated:
                td_target = r
            else:
                td_target = r + gamma * np.max(Q[s_next])
            td_error = td_target - Q[s, a]
            Q[s, a] += alpha * td_error

            s = s_next
            total_reward += r

        rewards_per_episode.append(total_reward)
        success_per_episode.append(1.0 if total_reward > 0 else 0.0)

        # ε 衰减
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if (ep + 1) % track_every == 0:
            recent = rewards_per_episode[-track_every:]
            history.append(np.mean(recent))

    return Q, rewards_per_episode, success_per_episode, history


# ============ 二、贪心策略评估 ============
def evaluate_greedy(env, Q, n_eval=200, seed=1234):
    """用学到的 Q 做纯贪心策略评估（无探索）。"""
    n_success = 0
    for ep in range(n_eval):
        s, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            a = int(np.argmax(Q[s]))
            s, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            if r > 0:
                n_success += 1
                break
    return n_success / n_eval


# ============ 三、可视化 ============
def visualize_curves(rewards_per_episode, success_per_episode, env_name, window=100):
    """画奖励曲线和成功率曲线。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    episodes = np.arange(1, len(rewards_per_episode) + 1)

    # ---- (a) 奖励曲线 ----
    ax = axes[0]
    if len(rewards_per_episode) >= window:
        smooth = np.convolve(rewards_per_episode,
                             np.ones(window) / window, mode='valid')
        ax.plot(episodes[window - 1:], smooth, color='tab:blue', lw=2,
                label=f'滑动平均 (窗口={window})')
    ax.plot(episodes, rewards_per_episode, color='tab:blue', alpha=0.15,
            label='单次奖励')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total reward')
    ax.set_title(f'{env_name}：每 episode 奖励')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)

    # ---- (b) 成功率曲线 ----
    ax = axes[1]
    if len(success_per_episode) >= window:
        smooth_s = np.convolve(success_per_episode,
                               np.ones(window) / window, mode='valid')
        ax.plot(episodes[window - 1:], smooth_s, color='tab:green', lw=2,
                label=f'滑动平均 (窗口={window})')
    ax.plot(episodes, success_per_episode, color='tab:green', alpha=0.15)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success rate')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('成功率（到达终点比例）')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    out_path = Path(__file__).parent / 'frozenlake_qlearning_curves.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'图片已保存为 {out_path}')


def visualize_Q(Q):
    """画 Q 表热力图和贪心策略箭头图。"""
    n_states = Q.shape[0]
    grid_size = int(np.sqrt(n_states))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---- (a) max_a Q(s, a) 热力图 ----
    ax = axes[0]
    v = np.max(Q, axis=1).reshape(grid_size, grid_size)
    im = ax.imshow(v, cmap='viridis')
    for r in range(grid_size):
        for c in range(grid_size):
            ax.text(c, r, f'{v[r, c]:.2f}', ha='center', va='center',
                    color='white', fontsize=9)
    ax.set_title('max_a Q(s, a)（状态价值）')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ---- (b) 贪心策略箭头 ----
    ax = axes[1]
    arrows = ['←', '↓', '→', '↑']            # FrozenLake: 0=左,1=下,2=右,3=上
    policy = np.argmax(Q, axis=1).reshape(grid_size, grid_size)
    ax.imshow(np.zeros((grid_size, grid_size)), cmap='gray', alpha=0.1)
    for r in range(grid_size):
        for c in range(grid_size):
            a = policy[r, c]
            ax.text(c, r, arrows[a], ha='center', va='center',
                    fontsize=22, color='tab:blue')
    ax.set_title('贪心策略 π(s) = argmax_a Q(s, a)')
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    out_path = Path(__file__).parent / 'frozenlake_qlearning_Q.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'图片已保存为 {out_path}')


# ============ 四、主流程 ============
def main():
    np.set_printoptions(precision=4, suppress=True)

    env_name = 'FrozenLake-v1'
    n_episodes = 5000

    print('=' * 60)
    print('FrozenLake Q-learning')
    print('=' * 60)

    # is_slippery=False 让环境变成确定性，容易学
    env = gym.make(env_name, is_slippery=False)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    print(f'环境: {env_name}（is_slippery=False）')
    print(f'状态数: {n_states}, 动作数: {n_actions}')
    print(f'episode 数: {n_episodes}\n')

    # ---- 训练 ----
    Q, rewards, successes, history = q_learning(
        env, n_episodes=n_episodes,
        alpha=0.1, gamma=0.99,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995,
        seed=42,
    )

    # ---- 统计 ----
    print(f'训练完成:')
    print(f'  最近 500 episode 平均奖励: {np.mean(rewards[-500:]):.3f}')
    print(f'  最近 500 episode 成功率: {np.mean(successes[-500:]):.3f}')
    eval_success = evaluate_greedy(env, Q, n_eval=200)
    print(f'  纯贪心策略评估成功率: {eval_success:.3f}')

    # ---- Q 表 ----
    print(f'\nQ 表（每行一个状态，4 个动作值）:')
    for s in range(n_states):
        print(f'  s={s:2d}: {np.round(Q[s], 3)}')

    # ---- 贪心策略 ----
    arrows = ['←', '↓', '→', '↑']
    grid_size = int(np.sqrt(n_states))
    print(f'\n贪心策略（箭头）:')
    for r in range(grid_size):
        row = '  '
        for c in range(grid_size):
            s = r * grid_size + c
            row += f'{arrows[int(np.argmax(Q[s]))]:^4}'
        print(row)

    env.close()

    # ---- 可视化 ----
    visualize_curves(rewards, successes, env_name, window=100)
    visualize_Q(Q)
    plt.show()


if __name__ == '__main__':
    main()