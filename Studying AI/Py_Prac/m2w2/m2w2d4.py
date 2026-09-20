# 学习内容：epsilon-greedy
# 实践任务：比较不同 epsilon 的 reward 曲线

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


# ============ 一、epsilon-greedy 选动作 ============
def epsilon_greedy(Q, s, n_actions, epsilon, rng):
    """ε-greedy 策略：
        - 以 ε 概率随机探索
        - 以 1-ε 概率选当前 Q 最大的动作（greedy）
    """
    if rng.random() < epsilon:
        # 探索：均匀随机选一个动作
        return int(rng.integers(n_actions))
    else:
        # 利用：取 Q[s] 最大的动作；并列时随机选一个
        q_values = Q[s]
        max_q = np.max(q_values)
        best_actions = np.flatnonzero(q_values == max_q)
        return int(rng.choice(best_actions))


# ============ 二、Q-learning 训练（固定 epsilon） ============
def q_learning_fixed_epsilon(env, epsilon, n_episodes=3000,
                             alpha=0.1, gamma=0.99, seed=42,
                             track_every=50):
    """用固定的 ε 跑 Q-learning，返回奖励曲线。"""
    rng = np.random.default_rng(seed)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))
    rewards_per_episode = []
    success_per_episode = []
    history = []

    for ep in range(n_episodes):
        s, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        done = False

        while not done:
            a = epsilon_greedy(Q, s, n_actions, epsilon, rng)
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            # Q-learning 更新
            if terminated:
                td_target = r
            else:
                td_target = r + gamma * np.max(Q[s_next])
            Q[s, a] += alpha * (td_target - Q[s, a])

            s = s_next
            total_reward += r

        rewards_per_episode.append(total_reward)
        success_per_episode.append(1.0 if total_reward > 0 else 0.0)

        if (ep + 1) % track_every == 0:
            recent = rewards_per_episode[-track_every:]
            history.append(np.mean(recent))

    return Q, rewards_per_episode, success_per_episode, history


# ============ 三、对比不同 epsilon ============
def compare_epsilons(env_name='FrozenLake-v1', epsilons=(0.01, 0.1, 0.3, 0.5, 1.0),
                     n_episodes=3000, seed=42):
    """对每个 epsilon 跑一次 Q-learning，返回结果字典。"""
    results = {}
    for eps in epsilons:
        print(f'训练中: ε = {eps}')
        env = gym.make(env_name, is_slippery=False)
        Q, rewards, successes, history = q_learning_fixed_epsilon(
            env, epsilon=eps, n_episodes=n_episodes, seed=seed,
        )
        env.close()

        # 用最终 Q 做贪心评估（衡量"学到的策略"有多好）
        env = gym.make(env_name, is_slippery=False)
        eval_success = evaluate_greedy(env, Q, n_eval=200)
        env.close()

        results[eps] = {
            'Q': Q,
            'rewards': rewards,
            'successes': successes,
            'history': history,
            'eval_success': eval_success,
        }
        print(f'  最近 500 episode 平均奖励: {np.mean(rewards[-500:]):.3f}')
        print(f'  贪心评估成功率: {eval_success:.3f}\n')

    return results


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


# ============ 四、可视化对比 ============
def visualize_comparison(results, epsilons, env_name):
    """对比不同 epsilon 的奖励曲线、成功率、评估成功率。"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(epsilons)))

    # ---- (a) 训练奖励曲线（滑动平均） ----
    ax = axes[0]
    for eps, color in zip(epsilons, colors):
        hist = results[eps]['history']
        x = np.arange(1, len(hist) + 1) * 50
        ax.plot(x, hist, color=color, lw=2, label=f'ε = {eps}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('平均奖励（滑动窗口=50）')
    ax.set_title(f'{env_name}：不同 ε 的训练奖励')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)

    # ---- (b) 训练成功率（滑动平均） ----
    ax = axes[1]
    for eps, color in zip(epsilons, colors):
        successes = results[eps]['successes']
        window = 50
        smooth = np.convolve(successes, np.ones(window) / window, mode='valid')
        x = np.arange(window, len(successes) + 1)
        ax.plot(x, smooth, color=color, lw=2, label=f'ε = {eps}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('成功率（滑动窗口=50）')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('训练过程中的成功率')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)

    # ---- (c) 最终贪心评估成功率 ----
    ax = axes[2]
    eval_success = [results[eps]['eval_success'] for eps in epsilons]
    bars = ax.bar([str(eps) for eps in epsilons], eval_success,
                  color=colors, alpha=0.85, edgecolor='k')
    for bar, v in zip(bars, eval_success):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f'{v:.2f}', ha='center', fontsize=10)
    ax.set_xlabel('ε')
    ax.set_ylabel('贪心评估成功率')
    ax.set_ylim(0, 1.1)
    ax.set_title('训练后的策略质量（纯贪心）')
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')

    fig.tight_layout()
    out_path = Path(__file__).parent / 'epsilon_comparison.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'图片已保存为 {out_path}')


# ============ 五、主流程 ============
def main():
    np.set_printoptions(precision=4, suppress=True)

    env_name = 'FrozenLake-v1'
    epsilons = (0.01, 0.1, 0.3, 0.5, 1.0)
    n_episodes = 3000

    print('=' * 60)
    print('不同 ε 的 Q-learning 对比')
    print('=' * 60)
    print(f'环境: {env_name}（is_slippery=False）')
    print(f'episode 数: {n_episodes}')
    print(f'对比的 ε: {epsilons}\n')

    # ---- 跑对比实验 ----
    results = compare_epsilons(env_name, epsilons, n_episodes, seed=42)

    # ---- 汇总表 ----
    print('=' * 60)
    print('结果汇总')
    print('=' * 60)
    print(f'{"ε":<8}{"最近500平均奖励":<20}{"贪心评估成功率":<20}')
    for eps in epsilons:
        r = results[eps]
        recent_reward = np.mean(r['rewards'][-500:])
        print(f'{eps:<8}{recent_reward:<20.3f}{r["eval_success"]:<20.3f}')

    # ---- 可视化 ----
    visualize_comparison(results, epsilons, env_name)
    plt.show()


if __name__ == '__main__':
    main()