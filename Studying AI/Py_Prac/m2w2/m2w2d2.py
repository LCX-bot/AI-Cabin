# 学习内容：Monte Carlo 与 TD
# 实践任务：实现 TD(0)

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# 让中文输出在 UTF-8 终端中正常显示（Windows 默认用 GBK，否则会乱码）
sys.stdout.reconfigure(encoding='utf-8')
# Windows 系统自带中文字体，优先用"微软雅黑"，它覆盖的汉字最全
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
# 负号在默认字体里会被渲染成方块，需要额外指定它用拉丁字体
plt.rcParams['axes.unicode_minus'] = False


# ============ 一、定义 MDP ============
def make_mdp():
    """3 状态 MDP，S2 终止。

    S0 --a=1--> S1 (70%) 或留在 S0 (30%, r=-1)
    S1 --a=1--> S2 (r=+10)
    S2 终止
    """
    states = [0, 1, 2]
    P = {
        0: {
            1: [(0.7, 1, 0.0), (0.3, 0, -1.0)],
        },
        1: {
            1: [(1.0, 2, 10.0)],
        },
        2: {},
    }
    policy = lambda s: 1 if s != 2 else None    # 固定策略
    gamma = 0.9
    return states, P, policy, gamma


# ============ 二、采样一条轨迹（和 MC 版一样） ============
def sample_trajectory(P, policy, start_state, max_steps=200, rng=None):
    """按策略采样一条完整轨迹，返回 (states, actions, rewards)。"""
    if rng is None:
        rng = np.random.default_rng()

    states, actions, rewards = [start_state], [], []
    s = start_state
    for _ in range(max_steps):
        if s not in P or len(P[s]) == 0:          # 终止状态
            break
        a = policy(s)
        transitions = P[s][a]
        probs = np.array([t[0] for t in transitions])
        idx = rng.choice(len(transitions), p=probs)
        _, s_next, r = transitions[idx]           # 从MDP表中提取出下一状态和即时奖励
        actions.append(a)
        rewards.append(r)
        states.append(s_next)
        s = s_next
    return states, actions, rewards               # 这里得到的是一条轨迹的信息，并不含有状态价值函数


# ============ 三、TD(0) 策略评估 ============
def td0_evaluate(P, policy, gamma, states, alpha=0.05, n_episodes=2000,
                 start_state=0, max_steps=200, seed=42, track_every=10):
    """TD(0) 策略评估。

    核心更新（每一步）：
        V(s) ← V(s) + α · [r + γ·V(s') - V(s)]

    与 MC 的区别：
        MC 要等 episode 结束才能算 G_t；
        TD 走一步就能更新（用 r + γV(s') 作为目标）。
    """
    rng = np.random.default_rng(seed)
    V = {s: 0.0 for s in states}

    # 记录 V(S0) 随 episode 变化的曲线，方便可视化
    history = []

    for ep in range(n_episodes):
        s = start_state
        for _ in range(max_steps):
            if s not in P or len(P[s]) == 0:
                break
            # 走一步
            a = policy(s)
            transitions = P[s][a]
            probs = np.array([t[0] for t in transitions])
            idx = rng.choice(len(transitions), p=probs)
            _, s_next, r = transitions[idx]

            # ---- TD(0) 更新 ----
            # 终止状态的 V 固定为 0
            v_next = 0.0 if (s_next not in P or len(P[s_next]) == 0) else V[s_next]
            td_target = r + gamma * v_next      # 用走一步后获得的价值更新当前步价值
            td_error = td_target - V[s]
            V[s] += alpha * td_error

            s = s_next

        # 每隔 track_every 个 episode 记录一次 V(S0)
        if (ep + 1) % track_every == 0:
            history.append(V[start_state])

    return V, history


# ============ 四、MC 策略评估（对比用） ============
def mc_evaluate(P, policy, gamma, states, n_episodes=2000, start_state=0,
                max_steps=200, seed=42, track_every=10):
    """蒙特卡洛策略评估（每次访问式）：跑完整轨迹后更新 V。"""
    rng = np.random.default_rng(seed)
    V = {s: 0.0 for s in states}
    counts = {s: 0 for s in states}

    history = []
    # 使用多次采样取平均的方法得到状态价值
    for ep in range(n_episodes):
        states_seq, _, rewards = sample_trajectory(P, policy, start_state,
                                                   max_steps, rng)
        # 从后往前计算折扣回报
        G = 0.0
        visited = set()
        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G
            s_t = states_seq[t]
            if s_t not in visited:             # 首次访问
                visited.add(s_t)
                counts[s_t] += 1
                V[s_t] += (G - V[s_t]) / counts[s_t]   # 在线平均

        if (ep + 1) % track_every == 0:
            history.append(V[start_state])

    return V, history


# ============ 五、精确解（作为参照） ============
def exact_evaluate(P, policy, gamma, states, tol=1e-10, max_iter=10000):
    """用迭代法精确求解贝尔曼方程，作为真值参照。"""
    V = {s: 0.0 for s in states}
    for _ in range(max_iter):
        V_new = {}
        for s in states:
            if s not in P or len(P[s]) == 0:
                V_new[s] = 0.0
                continue
            a = policy(s)
            total = 0.0
            for prob, s_next, r in P[s][a]:
                v_next = 0.0 if (s_next not in P or len(P[s_next]) == 0) else V[s_next]
                total += prob * (r + gamma * v_next)
            V_new[s] = total
        diff = max(abs(V_new[s] - V[s]) for s in states)
        V = V_new
        if diff < tol:
            break
    return V


# ============ 六、可视化 ============
def visualize(V_td, V_mc, V_exact, hist_td, hist_mc, states, gamma):
    """可视化：V 收敛曲线 + 最终 V 对比。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # ---- (a) V(S0) 的收敛曲线 ----
    ax = axes[0]
    x_td = np.arange(1, len(hist_td) + 1) * 10
    x_mc = np.arange(1, len(hist_mc) + 1) * 10
    ax.plot(x_td, hist_td, color='tab:blue', label='TD(0)', linewidth=1.8)
    ax.plot(x_mc, hist_mc, color='tab:orange', label='Monte Carlo', linewidth=1.8)
    ax.axhline(V_exact[0], color='gray', linestyle='--',
               label=f'精确解 = {V_exact[0]:.3f}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('V(S0)')
    ax.set_title(f'V(S0) 的收敛过程（γ = {gamma}）')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    # ---- (b) 各状态的最终 V 对比 ----
    ax = axes[1]
    labels = [f'S{s}' for s in states]
    x = np.arange(len(states))
    width = 0.25
    v_td = [V_td[s] for s in states]
    v_mc = [V_mc[s] for s in states]
    v_ex = [V_exact[s] for s in states]

    ax.bar(x - width, v_ex, width, label='精确解', color='gray', alpha=0.8)
    ax.bar(x, v_td, width, label='TD(0)', color='tab:blue', alpha=0.85)
    ax.bar(x + width, v_mc, width, label='MC', color='tab:orange', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('V(s)')
    ax.set_title('最终状态价值对比')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')

    fig.tight_layout()
    out_path = Path(__file__).parent / 'td0_vs_mc.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'图片已保存为 {out_path}')


# ============ 七、主流程 ============
def main():
    np.set_printoptions(precision=4, suppress=True)

    states, P, policy, gamma = make_mdp()
    n_episodes = 2000

    print('=' * 60)
    print('TD(0) vs Monte Carlo 策略评估')
    print('=' * 60)
    print(f'折扣因子 γ = {gamma}')
    print(f'episode 数: {n_episodes}\n')

    # ---- 精确解 ----
    V_exact = exact_evaluate(P, policy, gamma, states)
    print('精确解（贝尔曼迭代）:')
    for s in states:
        print(f'  V(S{s}) = {V_exact[s]:.4f}')

    # ---- TD(0) ----
    V_td, hist_td = td0_evaluate(P, policy, gamma, states,
                                 alpha=0.05, n_episodes=n_episodes)
    print(f'\nTD(0) 估计 (α=0.05):')
    for s in states:
        print(f'  V(S{s}) = {V_td[s]:.4f}  (误差 {abs(V_td[s] - V_exact[s]):.4f})')

    # ---- MC ----
    V_mc, hist_mc = mc_evaluate(P, policy, gamma, states,
                                n_episodes=n_episodes)
    print(f'\nMonte Carlo 估计:')
    for s in states:
        print(f'  V(S{s}) = {V_mc[s]:.4f}  (误差 {abs(V_mc[s] - V_exact[s]):.4f})')

    # ---- 收敛速度对比 ----
    print(f'\n收敛速度对比（V(S0) 到精确解的差距）:')
    td_err = abs(hist_td[-1] - V_exact[0])
    mc_err = abs(hist_mc[-1] - V_exact[0])
    print(f'  最终 TD 误差: {td_err:.4f}')
    print(f'  最终 MC 误差: {mc_err:.4f}')

    # ---- 可视化 ----
    visualize(V_td, V_mc, V_exact, hist_td, hist_mc, states, gamma)

    plt.show()


if __name__ == '__main__':
    main()