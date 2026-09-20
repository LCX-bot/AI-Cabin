# 学习内容：MDP 五元组
# 实践任务：手写 3 状态 MDP，计算 return

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


# ============ 一、定义 MDP 五元组 ============
def make_mdp():
    """构造一个 3 状态 MDP 的五元组 (S, A, P, R, γ)。

    状态：
        S0 —— 起始状态，动作 a=1 时大概率向右到 S1，小概率打滑留在 S0
        S1 —— 中间状态，动作 a=1 时确定性到达 S2
        S2 —— 吸收/终止状态，不再产生任何动作

    动作：
        a=0 —— 向左（S0 会留在原地并惩罚；S1 会回到 S0）
        a=1 —— 向右（S0 有 70% 到 S1；S1 直接到 S2 且奖励 +10）

    转移表 P[s][a] = [(概率, 下一状态, 奖励), ...]
    """
    states = [0, 1, 2]
    actions = [0, 1]
    P = {
        0: {
            0: [(1.0, 0, -1.0)],                    # 留在 S0，每步惩罚 -1
            1: [(0.7, 1,  0.0),                     # 70% 成功到 S1，奖励 0
                (0.3, 0, -1.0)],                    # 30% 打滑留在 S0，仍受惩罚
        },
        1: {
            0: [(1.0, 0,  0.0)],                    # 回到 S0，奖励 0
            1: [(1.0, 2, 10.0)],                    # 到 S2，奖励 +10
        },
        2: {},                                      # 终止状态无动作
    }
    gamma = 0.9                                     # 折扣因子
    return states, actions, P, gamma


# ============ 二、策略 ============
def policy(state):
    """确定性策略 π：S0 → a=1，S1 → a=1，S2 终止（返回 None）。"""
    if state == 2:
        return None
    return 1


# ============ 三、采样一条轨迹 ============
def sample_trajectory(P, policy, start_state, max_steps=200, rng=None):
    """从 start_state 出发按策略 π 采样一条轨迹。

    返回：
        states  : 状态序列 [s0, s1, ..., s_T]
        actions : 动作序列 [a0, a1, ..., a_{T-1}]
        rewards : 奖励序列 [r1, r2, ..., r_T]
    """
    if rng is None:
        rng = np.random.default_rng()

    states = [start_state]
    actions = []
    rewards = []

    s = start_state
    for _ in range(max_steps):
        if s == 2:                                  # 终止状态退出
            break
        a = policy(s)
        actions.append(a)

        # 从 P[s][a] 里按概率采样一个转移
        transitions = P[s][a]
        probs = np.array([t[0] for t in transitions])   # 取出概率
        idx = rng.choice(len(transitions), p=probs)     # 根据概率抽取
        _, s_next, r = transitions[idx]                 # 取出下个状态及即时奖励

        rewards.append(r)
        states.append(s_next)
        s = s_next

    return states, actions, rewards


# ============ 四、计算折扣回报 ============
def compute_return(rewards, gamma):
    """从奖励序列计算折扣回报 G = r1 + γ r2 + γ² r3 + ..."""
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
    return G


# ============ 五、蒙特卡洛估计状态价值 ============
def monte_carlo_evaluate(P, policy, gamma, n_episodes=1000, start_state=0,
                         max_steps=200, seed=42):
    """通过反复采样轨迹，估计状态价值 V^π。

    做法：每采样一条轨迹，计算从起点出发的折扣回报，
          多次平均得到 V(start_state)。
    """
    rng = np.random.default_rng(seed)
    returns = []
    for _ in range(n_episodes):
        _, _, rewards = sample_trajectory(P, policy, start_state, max_steps, rng)
        returns.append(compute_return(rewards, gamma))
    returns = np.array(returns)
    return returns.mean(), returns.std(), returns


# ============ 六、贝尔曼方程精确求解 V^π ============
def policy_evaluation_exact(P, policy, gamma, states, tol=1e-10, max_iter=10000):
    """用迭代法解贝尔曼方程，得到精确的 V^π。

    贝尔曼方程：
        V(s) = Σ_{s', r} P(s', r | s, a) · [r + γ · V(s')]
    其中 a = π(s)。终止状态 V = 0。
    """
    # 用索引 0..n-1 表示状态，方便操作数组
    idx = {s: i for i, s in enumerate(states)}  # 给状态链的每个状态赋予索引
    n = len(states)
    V = np.zeros(n)

    for _ in range(max_iter):
        V_new = np.zeros(n)
        for s in states:
            if s == 2:
                V_new[idx[s]] = 0.0
                continue
            a = policy(s)
            total = 0.0
            for prob, s_next, r in P[s][a]:
                total += prob * (r + gamma * V[idx[s_next]])
            V_new[idx[s]] = total
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    return {s: V[idx[s]] for s in states}


# ============ 七、打印与可视化 ============
def demo_trajectory():
    """采样一条轨迹，逐步计算折扣回报，直观展示 G 的含义。"""
    print('=' * 60)
    print('一、单条轨迹与折扣回报')
    print('=' * 60)

    states, actions, P, gamma = make_mdp()[:3] + (make_mdp()[3],)
    # 上面写错了，直接重新构造
    states, actions, P, gamma = make_mdp()

    rng = np.random.default_rng(seed=7)
    s_seq, a_seq, r_seq = sample_trajectory(P, policy, start_state=0, rng=rng)

    print(f'折扣因子 γ = {gamma}')
    print(f'策略 π: S0→a=1, S1→a=1')
    print(f'\n采样轨迹（长度 {len(r_seq)} 步）:')
    for i, (s, a, r) in enumerate(zip(s_seq[:-1], a_seq, r_seq)):
        s_next = s_seq[i + 1]
        print(f'  step {i}: s={s} --a={a}--> s\'={s_next}, r={r:+.2f}')
    print(f'  终止于状态 s={s_seq[-1]}')

    # 逐步计算折扣回报（从后往前累积）
    print(f'\n逐步计算折扣回报 G（从轨迹末尾向前累加）:')
    G = 0.0
    for i in range(len(r_seq) - 1, -1, -1):
        G = r_seq[i] + gamma * G
        print(f'  G_{i} = r_{i+1} + γ·G_{i+1} = {r_seq[i]:+.2f} + {gamma}·{G - r_seq[i]:+.4f} = {G:.4f}')

    print(f'\n最终折扣回报 G_0 = {G:.4f}')


def demo_monte_carlo_vs_exact():
    """对比蒙特卡洛估计与贝尔曼方程精确解。"""
    print('\n' + '=' * 60)
    print('二、蒙特卡洛估计 vs 贝尔曼精确解')
    print('=' * 60)

    states, actions, P, gamma = make_mdp()

    # ---- 蒙特卡洛 ----
    mean_mc, std_mc, returns = monte_carlo_evaluate(
        P, policy, gamma, n_episodes=5000, start_state=0, seed=42
    )
    print(f'蒙特卡洛（5000 次采样，从 S0 出发）:')
    print(f'  V(S0) 估计 = {mean_mc:.4f} ± {std_mc:.4f}（标准差）')

    # ---- 贝尔曼精确解 ----
    V_exact = policy_evaluation_exact(P, policy, gamma, states)
    print(f'\n贝尔曼方程精确解:')
    for s in states:
        print(f'  V(S{s}) = {V_exact[s]:.4f}')

    # ---- 手动推导验证 S0 ----
    # V(S2) = 0
    # V(S1) = 10 + γ·0 = 10
    # V(S0) = 0.7·(0 + γ·V(S1)) + 0.3·(-1 + γ·V(S0))
    #       = 0.7·9 + 0.3·(-1 + 0.9·V(S0))
    #       = 6.3 - 0.3 + 0.27·V(S0)
    #       = 6.0 + 0.27·V(S0)
    # V(S0) = 6.0 / 0.73 ≈ 8.2192
    V_S0_manual = 6.0 / 0.73
    print(f'\n手算验证 V(S0) = 6.0 / 0.73 ≈ {V_S0_manual:.4f}')
    print(f'  与精确解差异: {abs(V_S0_manual - V_exact[0]):.2e}')
    print(f'  与 MC 估计差异: {abs(V_S0_manual - mean_mc):.4f}（MC 有采样噪声）')

    return returns, V_exact


def demo_visualize(returns, V_exact, gamma):
    """可视化：回报分布、状态价值对比、MDP 转移图。"""
    print('\n' + '=' * 60)
    print('三、可视化')
    print('=' * 60)

    fig = plt.figure(figsize=(15, 5))

    # ---- (a) 蒙特卡洛回报的直方图 ----
    ax = fig.add_subplot(1, 3, 1)
    ax.hist(returns, bins=60, color='tab:blue', alpha=0.7, edgecolor='k', linewidth=0.3)
    ax.axvline(returns.mean(), color='tab:red', linestyle='--', linewidth=2,
               label=f'均值 = {returns.mean():.3f}')
    ax.axvline(V_exact[0], color='tab:green', linestyle=':', linewidth=2,
               label=f'精确解 = {V_exact[0]:.3f}')
    ax.set_xlabel('折扣回报 G')
    ax.set_ylabel('频次')
    ax.set_title('从 S0 出发的回报分布')
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)

    # ---- (b) 状态价值对比 ----
    ax = fig.add_subplot(1, 3, 2)
    state_labels = ['S0', 'S1', 'S2']
    values = [V_exact[0], V_exact[1], V_exact[2]]
    colors = ['tab:blue', 'tab:orange', 'tab:gray']
    bars = ax.bar(state_labels, values, color=colors, alpha=0.8, edgecolor='k')
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.1,
                f'{v:.3f}', ha='center', fontsize=10)
    ax.set_ylabel('V^π(s)')
    ax.set_title(f'状态价值（γ = {gamma}）')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')

    # ---- (c) MDP 状态转移示意图 ----
    ax = fig.add_subplot(1, 3, 3)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-1.0, 1.5)
    ax.axis('off')

    # 三个状态的坐标
    positions = {0: (0, 0), 1: (1.8, 0), 2: (3.3, 0)}
    for s, (x, y) in positions.items():
        color = 'lightgray' if s == 2 else 'lightblue'
        circle = plt.Circle((x, y), 0.28, color=color, ec='k', zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, f'S{s}', ha='center', va='center', fontsize=12, zorder=4)

    # 画转移箭头
    arrow_kw = dict(arrowstyle='->', lw=1.8, color='k')
    # S0 --a=1 (0.7)--> S1
    ax.annotate('', xy=(1.55, 0.08), xytext=(0.28, 0.08),
                arrowprops=arrow_kw)
    ax.text(0.9, 0.22, 'a=1: 0.7', fontsize=9, ha='center')

    # S0 --a=1 (0.3)--> S0（打滑）
    ax.annotate('', xy=(0.05, 0.30), xytext=(0.25, 0.30),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='tab:red',
                                connectionstyle='arc3,rad=1.2'))
    ax.text(-0.1, 0.75, 'a=1: 0.3, r=-1', fontsize=8, color='tab:red', ha='center')

    # S0 --a=0--> S0（惩罚）
    ax.annotate('', xy=(0.05, -0.30), xytext=(0.25, -0.30),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='tab:blue',
                                connectionstyle='arc3,rad=-1.2'))
    ax.text(-0.1, -0.75, 'a=0: r=-1', fontsize=8, color='tab:blue', ha='center')

    # S1 --a=1 (1.0)--> S2
    ax.annotate('', xy=(3.05, 0.08), xytext=(2.10, 0.08), arrowprops=arrow_kw)
    ax.text(2.55, 0.22, 'a=1: 1.0, r=+10', fontsize=9, ha='center')

    # S1 --a=0--> S0
    ax.annotate('', xy=(0.28, -0.08), xytext=(1.55, -0.08),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    ax.text(0.9, -0.25, 'a=0: r=0', fontsize=8, color='gray', ha='center')

    ax.set_title('MDP 状态转移图（策略 π: a=1）', fontsize=11)
    ax.set_ylim(-1.2, 1.2)

    fig.tight_layout()
    out_path = Path(__file__).parent / 'mdp_3state.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'图片已保存为 {out_path}')


# ============ 八、主流程 ============
def main():
    np.set_printoptions(precision=4, suppress=True)

    demo_trajectory()
    returns, V_exact = demo_monte_carlo_vs_exact()
    demo_visualize(returns, V_exact, gamma=0.9)

    plt.show()


if __name__ == '__main__':
    main()