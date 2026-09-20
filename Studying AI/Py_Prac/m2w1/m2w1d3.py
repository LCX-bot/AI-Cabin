# 学习内容：V(s)、Q(s,a)
# 实践任务：对小 MDP 做 policy evaluation

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
    """构造一个 4 状态 MDP，动作不同、奖励不同，便于对比 V 和 Q。

    状态（像一个一维走廊）：
        S0 --a=1--> S1
        S1 --a=1--> S2
        S2 --a=1--> S3（终点，奖励 +10）
        S3 是吸收/终止状态

    动作：
        a=0 —— 向左后退一步，S0 时留在原地
        a=1 —— 向右前进一步，S2 前进到 S3 得到 +10 奖励

    每个状态在动作 a=1 时有小概率（0.2）打滑，留在原地并扣 -1。
    """
    states = [0, 1, 2, 3]
    actions = [0, 1]

    # P[s][a] = [(概率, 下一状态, 奖励), ...]
    P = {
        0: {
            0: [(1.0, 0, -1.0)],                              # 撞墙，扣 1
            1: [(0.8, 1, 0.0), (0.2, 0, -1.0)],               # 80% 前进，20% 打滑
        },
        1: {
            0: [(1.0, 0, 0.0)],                               # 退到 S0
            1: [(0.8, 2, 0.0), (0.2, 1, -1.0)],               # 80% 前进，20% 打滑
        },
        2: {
            0: [(1.0, 1, 0.0)],                               # 退到 S1
            1: [(0.8, 3, 10.0), (0.2, 2, -1.0)],              # 80% 到终点，20% 打滑
        },
        3: {},                                                # 终止状态
    }
    gamma = 0.9
    return states, actions, P, gamma


# ============ 二、策略 ============
def policy(state):
    """确定性策略 π：所有非终止状态都选 a=1（一直往右走）。"""
    if state == 3:
        return None
    return 1


# ============ 三、策略评估：求 V^π ============
def policy_evaluation_V(P, policy, gamma, states, tol=1e-10, max_iter=10000):
    """用迭代法解贝尔曼方程，得到 V^π(s)。

    贝尔曼方程：
        V(s) = Σ_{s', r} P(s', r | s, π(s)) · [r + γ·V(s')]
    """
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    V = np.zeros(n)

    for it in range(max_iter):
        V_new = np.zeros(n)
        for s in states:
            if s == 3:                                        # 终止状态 V = 0
                V_new[idx[s]] = 0.0
                continue
            a = policy(s)
            total = 0.0
            for prob, s_next, r in P[s][a]:
                total += prob * (r + gamma * V[idx[s_next]])
            V_new[idx[s]] = total
        diff = np.max(np.abs(V_new - V))
        V = V_new
        if diff < tol:
            break
    return {s: V[idx[s]] for s in states}


# ============ 四、策略评估：求 Q^π ============
def policy_evaluation_Q(P, policy, gamma, states, actions, V=None,
                        tol=1e-10, max_iter=10000):
    """求 Q^π(s, a)，即在状态 s 先做动作 a、之后按策略 π 走的期望折扣回报。

    贝尔曼方程：
        Q(s, a) = Σ_{s', r} P(s', r | s, a) · [r + γ·V^π(s')]

    注意：Q 的方程里用到的是 V，不是 Q。所以一般先算 V，再用 V 直接算 Q。
    """
    # 如果没给 V，先自己算一遍
    if V is None:
        V = policy_evaluation_V(P, policy, gamma, states, tol, max_iter)

    Q = {}
    for s in states:
        if s == 3:                                            # 终止状态无动作
            continue
        for a in actions:
            if a not in P[s]:                                 # 该动作在此状态不可用
                continue
            total = 0.0
            for prob, s_next, r in P[s][a]:
                total += prob * (r + gamma * V[s_next])
            Q[(s, a)] = total
    return Q


# ============ 五、从 Q 导出 V（验证一致性） ============
def V_from_Q(Q, policy, states):
    """用 V(s) = Q(s, π(s)) 从 Q 反推 V，验证和直接算的 V 一致。"""
    V = {}
    for s in states:
        if s == 3:
            V[s] = 0.0
            continue
        a = policy(s)
        V[s] = Q[(s, a)]
    return V


# ============ 六、策略改进（用 Q 找更好的动作） ============
def policy_improvement(Q, states, actions):
    """基于 Q 做一步贪心改进：π'(s) = argmax_a Q(s, a)。"""
    new_policy = {}
    for s in states:
        if s == 3:
            new_policy[s] = None
            continue
        q_values = {a: Q[(s, a)] for a in actions if (s, a) in Q}
        best_a = max(q_values, key=q_values.get)
        new_policy[s] = best_a
    return new_policy


# ============ 七、打印与可视化 ============
def print_V_and_Q(V, Q, states, actions):
    """格式化打印 V 和 Q 表格。"""
    print('\nV(s) 表格（状态价值）:')
    print(f'{"状态":<6}{"V^π(s)":<12}')
    for s in states:
        print(f'S{s:<5}{V[s]:<12.4f}')

    print('\nQ(s, a) 表格（动作价值）:')
    header = f'{"状态":<6}' + ''.join([f'{"Q(s,a=" + str(a) + ")":<16}' for a in actions])
    print(header)
    for s in states:
        row = f'S{s:<5}'
        for a in actions:
            if (s, a) in Q:
                row += f'{Q[(s, a)]:<16.4f}'
            else:
                row += f'{"—":<16}'
        print(row)

    print('\nV 和 Q 的关系：V(s) = Q(s, π(s))')
    print('（如果 π(s) = argmax Q(s, ·)，则 V(s) = max_a Q(s, a)）')


def demo_V_Q_relationship():
    """展示 V 和 Q 的数学关系，并做一次策略改进。"""
    print('=' * 60)
    print('一、V(s) 与 Q(s, a) 的 policy evaluation')
    print('=' * 60)

    states, actions, P, gamma = make_mdp()
    print(f'折扣因子 γ = {gamma}')
    print(f'策略 π：所有非终止状态都选 a=1（一直往右走）')

    # ---- 第一步：算 V^π ----
    V = policy_evaluation_V(P, policy, gamma, states)

    # ---- 第二步：用 V 算 Q^π ----
    Q = policy_evaluation_Q(P, policy, gamma, states, actions, V=V)

    print_V_and_Q(V, Q, states, actions)

    # ---- 第三步：验证 V(s) = Q(s, π(s)) ----
    print('\n验证 V(s) = Q(s, π(s)):')
    V_from_q = V_from_Q(Q, policy, states)
    for s in states:
        if s == 3:
            continue
        a = policy(s)
        print(f'  V(S{s}) = {V[s]:.4f}, Q(S{s}, a={a}) = {Q[(s, a)]:.4f}, '
              f'差异 = {abs(V[s] - Q[(s, a)]):.2e}')

    # ---- 第四步：策略改进 ----
    print('\n基于 Q 做一步贪心改进：π\'(s) = argmax_a Q(s, a)')
    new_policy = policy_improvement(Q, states, actions)
    for s in states:
        if s == 3:
            continue
        old_a = policy(s)
        new_a = new_policy[s]
        marker = '（不变）' if old_a == new_a else '（改进了！）'
        print(f'  S{s}: 旧策略 a={old_a}  →  新策略 a={new_a}  {marker}')

    return V, Q, states, actions


def demo_V_Q_visualize(V, Q, states, actions):
    """可视化：V 柱状图 + Q 分组柱状图 + 状态转移示意图。"""
    fig = plt.figure(figsize=(16, 5))

    # ---- (a) V(s) 柱状图 ----
    ax = fig.add_subplot(1, 3, 1)
    state_labels = [f'S{s}' for s in states]
    v_values = [V[s] for s in states]
    colors = ['tab:blue' if s != 3 else 'tab:gray' for s in states]
    bars = ax.bar(state_labels, v_values, color=colors, alpha=0.85, edgecolor='k')
    for bar, v in zip(bars, v_values):
        offset = 0.15 if v >= 0 else -0.4
        ax.text(bar.get_x() + bar.get_width() / 2, v + offset,
                f'{v:.3f}', ha='center', fontsize=10)
    ax.set_ylabel('V^π(s)')
    ax.set_title('状态价值 V(s)')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')

    # ---- (b) Q(s, a) 分组柱状图 ----
    ax = fig.add_subplot(1, 3, 2)
    non_terminal_states = [s for s in states if s != 3]
    x = np.arange(len(non_terminal_states))
    width = 0.35
    q_a0 = [Q.get((s, 0), 0.0) for s in non_terminal_states]
    q_a1 = [Q.get((s, 1), 0.0) for s in non_terminal_states]

    bars0 = ax.bar(x - width/2, q_a0, width, label='Q(s, a=0) 左移',
                   color='tab:orange', alpha=0.85, edgecolor='k')
    bars1 = ax.bar(x + width/2, q_a1, width, label='Q(s, a=1) 右移',
                   color='tab:green', alpha=0.85, edgecolor='k')

    for bars in (bars0, bars1):
        for bar in bars:
            h = bar.get_height()
            offset = 0.15 if h >= 0 else -0.4
            ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                    f'{h:.2f}', ha='center', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f'S{s}' for s in non_terminal_states])
    ax.set_ylabel('Q^π(s, a)')
    ax.set_title('动作价值 Q(s, a)')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')

    # ---- (c) MDP 状态转移示意图 ----
    ax = fig.add_subplot(1, 3, 3)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-1.2, 1.5)
    ax.axis('off')

    positions = {0: (0, 0), 1: (1, 0), 2: (2, 0), 3: (3, 0)}
    for s, (x_, y_) in positions.items():
        color = 'lightgray' if s == 3 else 'lightblue'
        circle = plt.Circle((x_, y_), 0.28, color=color, ec='k', zorder=3)
        ax.add_patch(circle)
        ax.text(x_, y_, f'S{s}', ha='center', va='center', fontsize=12, zorder=4)
        # 在圆圈下方标注 V 值
        ax.text(x_, y_ - 0.45, f'V={V[s]:.2f}', ha='center', fontsize=9,
                color='tab:blue')

    # 右移箭头
    for s in [0, 1, 2]:
        x0 = positions[s][0] + 0.28
        x1 = positions[s + 1][0] - 0.28
        ax.annotate('', xy=(x1, 0.08), xytext=(x0, 0.08),
                    arrowprops=dict(arrowstyle='->', lw=1.8, color='k'))
        label = 'r=+10' if s == 2 else 'r=0'
        ax.text((x0 + x1) / 2, 0.22, f'a=1, {label}', fontsize=8, ha='center')

    # 左移箭头
    for s in [1, 2]:
        x0 = positions[s][0] - 0.28
        x1 = positions[s - 1][0] + 0.28
        ax.annotate('', xy=(x1, -0.08), xytext=(x0, -0.08),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
        ax.text((x0 + x1) / 2, -0.28, 'a=0, r=0', fontsize=8,
                ha='center', color='gray')

    ax.set_title('MDP 结构（策略 π: a=1）', fontsize=11)

    fig.tight_layout()
    out_path = Path(__file__).parent / 'mdp_V_Q.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\n图片已保存为 {out_path}')


# ============ 八、主流程 ============
def main():
    np.set_printoptions(precision=4, suppress=True)

    V, Q, states, actions = demo_V_Q_relationship()
    demo_V_Q_visualize(V, Q, states, actions)

    plt.show()


if __name__ == '__main__':
    main()

