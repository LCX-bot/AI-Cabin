# 学习内容：Bellman 方程
# 实践任务：实现 Bellman backup

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


# ============ 一、定义一个小 MDP ============
def make_mdp():
    """3 状态 MDP，S2 为终止状态。

    S0 --a=1--> S1 (80%) 或留在 S0 (20%, r=-1)
    S1 --a=1--> S2 (r=+10)
    S2 终止
    """
    states = [0, 1, 2]
    actions = [0, 1]
    P = {
        0: {
            0: [(1.0, 0, -1.0)],
            1: [(0.8, 1, 0.0), (0.2, 0, -1.0)],
        },
        1: {
            0: [(1.0, 0, 0.0)],
            1: [(1.0, 2, 10.0)],
        },
        2: {},
    }
    gamma = 0.9
    return states, actions, P, gamma


# ============ 二、通用的 Bellman backup ============
def bellman_backup(s, a, P, gamma, V, terminal_states=(2,)):
    """对 (s, a) 做一次 Bellman backup，返回 backup 后的新价值。

    公式：
        B(s, a) = Σ_{s', r} P(s', r | s, a) · [r + γ·V(s')]

    这是 Q 值的 backup，也是"给定 V，算 Q"的核心一步。
    对于策略评估场景，取 a = π(s) 就得到 V 的 backup。
    """
    total = 0.0
    for prob, s_next, r in P[s][a]:
        # 终止状态的 V 固定为 0
        v_next = 0.0 if s_next in terminal_states else V[s_next]
        total += prob * (r + gamma * v_next)
    return total


def bellman_backup_V(s, policy, P, gamma, V, terminal_states=(2,)):
    """对状态 s 做一次 V 的 backup（用当前策略的动作）。

    公式：
        B_V(s) = Σ_{s', r} P(s', r | s, π(s)) · [r + γ·V(s')]
    """
    if s in terminal_states:
        return 0.0
    a = policy(s)
    return bellman_backup(s, a, P, gamma, V, terminal_states)


def bellman_optimal_backup_V(s, P, gamma, V, terminal_states=(2,)):
    """V 的最优 Bellman backup：对所有动作取 max。

    公式：
        B*V(s) = max_a Σ_{s', r} P(s', r | s, a) · [r + γ·V(s')]
    """
    if s in terminal_states:
        return 0.0
    # 对该状态下所有可用动作做 backup，取最大
    q_values = [bellman_backup(s, a, P, gamma, V, terminal_states)
                for a in P[s].keys()]
    return max(q_values)


# ============ 三、用 backup 做策略评估 ============
def policy_evaluation_by_backup(P, policy, gamma, states, terminal_states=(2,),
                                tol=1e-10, max_iter=10000):
    """反复对所有状态做 Bellman backup，直到收敛。

    这正是 policy evaluation 的本质：
        每一轮迭代 = 对所有状态做一次 backup
    """
    V = {s: 0.0 for s in states}
    for it in range(max_iter):
        V_new = {}
        for s in states:
            V_new[s] = bellman_backup_V(s, policy, P, gamma, V, terminal_states)
        diff = max(abs(V_new[s] - V[s]) for s in states)
        V = V_new
        if diff < tol:
            return V, it + 1
    return V, max_iter


def value_iteration_by_backup(P, gamma, states, terminal_states=(2,),
                              tol=1e-10, max_iter=10000):
    """反复对所有状态做最优 Bellman backup，直到收敛。

    这是 value iteration：
        每一轮迭代 = 对所有状态做一次 max backup
    """
    V = {s: 0.0 for s in states}
    for it in range(max_iter):
        V_new = {}
        for s in states:
            V_new[s] = bellman_optimal_backup_V(s, P, gamma, V, terminal_states)
        diff = max(abs(V_new[s] - V[s]) for s in states)
        V = V_new
        if diff < tol:
            return V, it + 1
    return V, max_iter


# ============ 四、演示 ============
def demo_single_backup():
    """演示对单个状态做一次 backup 的数值变化。"""
    print('=' * 60)
    print('一、单次 Bellman backup 的效果')
    print('=' * 60)

    states, actions, P, gamma = make_mdp()
    policy = lambda s: 1

    # 初始 V 全 0
    V = {0: 0.0, 1: 0.0, 2: 0.0}
    print(f'初始 V: {V}')
    print(f'γ = {gamma}\n')

    # 对 S1 做一次 backup
    v1_new = bellman_backup_V(1, policy, P, gamma, V)
    print(f'对 S1 做 backup:')
    print(f'  B_V(S1) = 1.0 · [10 + γ·V(S2)] = 1.0 · [10 + 0.9·0] = {v1_new}')

    # 对 S0 做一次 backup
    v0_new = bellman_backup_V(0, policy, P, gamma, V)
    print(f'\n对 S0 做 backup:')
    print(f'  B_V(S0) = 0.8·[0 + γ·V(S1)] + 0.2·[-1 + γ·V(S0)]')
    print(f'         = 0.8·[0 + 0.9·0] + 0.2·[-1 + 0.9·0] = {v0_new}')

    # 对 S0 做一次最优 backup
    v0_opt = bellman_optimal_backup_V(0, P, gamma, V)
    print(f'\n对 S0 做最优 backup（取 max）：')
    q_a0 = bellman_backup(0, 0, P, gamma, V)
    q_a1 = bellman_backup(0, 1, P, gamma, V)
    print(f'  Q(S0, a=0) = {q_a0}')
    print(f'  Q(S0, a=1) = {q_a1}')
    print(f'  max = {v0_opt}')


def demo_convergence():
    """演示反复 backup 收敛到 V^π 和 V*。"""
    print('\n' + '=' * 60)
    print('二、反复 backup → 收敛')
    print('=' * 60)

    states, actions, P, gamma = make_mdp()
    policy = lambda s: 1

    # ---- 策略评估 ----
    V_pi, n_iter_pi = policy_evaluation_by_backup(P, policy, gamma, states)
    print(f'策略评估（反复 V backup）:')
    print(f'  收敛用了 {n_iter_pi} 轮')
    for s in states:
        print(f'  V^π(S{s}) = {V_pi[s]:.4f}')

    # ---- 价值迭代 ----
    V_star, n_iter_star = value_iteration_by_backup(P, gamma, states)
    print(f'\n价值迭代（反复最优 backup）:')
    print(f'  收敛用了 {n_iter_star} 轮')
    for s in states:
        print(f'  V*(S{s}) = {V_star[s]:.4f}')

    # ---- 手算验证 V(S0) ----
    # V(S1) = 10
    # V(S0) = 0.8·(0 + 0.9·10) + 0.2·(-1 + 0.9·V(S0))
    #       = 7.2 - 0.2 + 0.18·V(S0)
    #       = 7.0 + 0.18·V(S0)
    # V(S0) = 7.0 / 0.82 ≈ 8.5366
    v0_manual = 7.0 / 0.82
    print(f'\n手算验证 V^π(S0) = 7.0 / 0.82 = {v0_manual:.4f}')

    return V_pi, V_star


def demo_backup_trajectory():
    """展示每轮 backup 后 V 的变化轨迹。"""
    print('\n' + '=' * 60)
    print('三、backup 的收敛轨迹')
    print('=' * 60)

    states, actions, P, gamma = make_mdp()
    policy = lambda s: 1

    V = {s: 0.0 for s in states}
    history = [V.copy()]
    for it in range(30):
        V_new = {s: bellman_backup_V(s, policy, P, gamma, V) for s in states}
        V = V_new
        history.append(V.copy())

    print(f'{"iter":<6}{"V(S0)":<12}{"V(S1)":<12}{"V(S2)":<12}')
    for i, V in enumerate(history):
        if i % 5 == 0 or i == len(history) - 1:
            print(f'{i:<6}{V[0]:<12.4f}{V[1]:<12.4f}{V[2]:<12.4f}')

    # ---- 绘图 ----
    fig, ax = plt.subplots(figsize=(8, 5))
    iters = list(range(len(history)))
    v0_hist = [V[0] for V in history]
    v1_hist = [V[1] for V in history]
    ax.plot(iters, v0_hist, 'o-', label='V(S0)', markersize=4)
    ax.plot(iters, v1_hist, 's-', label='V(S1)', markersize=4)
    ax.axhline(7.0 / 0.82, color='tab:blue', linestyle='--', alpha=0.5,
               label=f'V*(S0) = {7.0/0.82:.4f}')
    ax.set_xlabel('backup 轮数')
    ax.set_ylabel('V(s)')
    ax.set_title('Bellman backup 的收敛过程')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    out_path = Path(__file__).parent / 'bellman_backup.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\n图片已保存为 {out_path}')


# ============ 五、主流程 ============
def main():
    np.set_printoptions(precision=4, suppress=True)

    demo_single_backup()
    demo_convergence()
    demo_backup_trajectory()

    plt.show()


if __name__ == '__main__':
    main()