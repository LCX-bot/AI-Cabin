# 学习内容：策略迭代和值迭代
# 实践任务：GridWorld value iteration

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


# ============ 一、定义 GridWorld ============
class GridWorld:
    """4×4 网格世界。

    规则：
        - 从左上角 (0,0) 出发，目标是右下角 (3,3)。
        - 每个格子可以走 4 个方向：上、下、左、右。
        - 每走一步奖励 -1（鼓励尽快到达）。
        - 到达 (3,3) 终止，且不再有后续奖励。
        - 撞墙或走出边界时留在原地，仍然奖励 -1。
    """

    def __init__(self, size=4, gamma=0.99):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4                                # 0=上, 1=下, 2=左, 3=右
        self.action_names = ['↑', '↓', '←', '→']
        self.gamma = gamma
        self.goal = (size - 1, size - 1)                  # 终点坐标
        self.start = (0, 0)

    # ---- 坐标与状态 ID 的互转 ----
    def state_to_rc(self, s):
        """状态 ID → (行, 列)。"""
        return divmod(s, self.size)

    def rc_to_state(self, r, c):
        """(行, 列) → 状态 ID。"""
        return r * self.size + c

    def is_terminal(self, s):
        """判断状态是否为终止状态。"""
        return self.state_to_rc(s) == self.goal

    # ---- 状态转移 ----
    def step(self, s, a):
        """从状态 s 执行动作 a，返回 (下一状态, 奖励)。

        撞墙时留在原地，奖励仍为 -1。
        """
        r, c = self.state_to_rc(s)

        # 计算移动后的位置
        if a == 0:      # 上
            nr, nc = r - 1, c
        elif a == 1:    # 下
            nr, nc = r + 1, c
        elif a == 2:    # 左
            nr, nc = r, c - 1
        else:           # 右
            nr, nc = r, c + 1

        # 边界检查：越界则留在原地
        if not (0 <= nr < self.size and 0 <= nc < self.size):
            nr, nc = r, c

        return self.rc_to_state(nr, nc), -1.0

    # ---- 对给定动作做一次 backup（GridWorld 是确定性的） ----
    def backup(self, s, a, V):
        """对 (s, a) 做一次 Bellman backup，返回 Q(s, a)。"""
        s_next, r = self.step(s, a)
        # 终止状态的 V 固定为 0，不再 bootstrap
        v_next = 0.0 if self.is_terminal(s_next) else V[s_next]
        return r + self.gamma * v_next      # 计算Q值


# ============ 二、值迭代 ============
def value_iteration(env, tol=1e-6, max_iter=1000):
    """值迭代：反复对所有状态做最优 backup，直到收敛。

    返回：
        V          : 最优状态价值，形状 (n_states,)
        policy     : 最优策略，形状 (n_states,)，每个状态对应的最优动作
        n_iter     : 收敛所需的迭代次数
        history    : 每次迭代后的 V（用于可视化收敛过程）
    """
    V = np.zeros(env.n_states)
    history = [V.copy()]

    for it in range(max_iter):
        V_new = np.zeros(env.n_states)
        for s in range(env.n_states):
            if env.is_terminal(s):
                V_new[s] = 0.0
                continue
            # 最优 backup：对所有动作取 max
            q_values = [env.backup(s, a, V) for a in range(env.n_actions)]
            V_new[s] = max(q_values)

        diff = np.max(np.abs(V_new - V))
        V = V_new
        history.append(V.copy())

        if diff < tol:
            break

    # 从 V* 提取最优策略：每个状态选使 Q 最大的动作
    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        if env.is_terminal(s):
            continue
        q_values = [env.backup(s, a, V) for a in range(env.n_actions)]
        policy[s] = int(np.argmax(q_values))

    return V, policy, it + 1, history


# ============ 三、从策略计算状态价值（用最优策略验证） ============
def evaluate_policy(env, policy, tol=1e-8, max_iter=10000):
    """给定策略，用迭代法求 V^π。

    这是 policy evaluation，和 value_iteration 唯一的区别是：
        value_iteration 用 max over a，
        这里用 policy[s] 指定的动作。
    """
    V = np.zeros(env.n_states)
    for _ in range(max_iter):
        V_new = np.zeros(env.n_states)
        for s in range(env.n_states):
            if env.is_terminal(s):
                V_new[s] = 0.0
                continue
            a = policy[s]
            V_new[s] = env.backup(s, a, V)
        diff = np.max(np.abs(V_new - V))
        V = V_new
        if diff < tol:
            break
    return V


# ============ 四、可视化 ============
def draw_grid(ax, env, V, policy, title):
    """在 ax 上画出 GridWorld 的 V 和策略箭头。"""
    n = env.size
    grid = V.reshape(n, n)

    # 画 V 的数值热力图
    im = ax.imshow(grid, cmap='viridis', alpha=0.85)

    # 在每个格子里写数值 + 画箭头
    for r in range(n):
        for c in range(n):
            s = env.rc_to_state(r, c)
            v = grid[r, c]

            if (r, c) == env.goal:
                ax.text(c, r, 'GOAL', ha='center', va='center',
                        color='white', fontsize=9, fontweight='bold')
                continue

            # V 的数值
            ax.text(c, r - 0.2, f'{v:.1f}', ha='center', va='center',
                    color='white', fontsize=9)

            # 策略箭头
            a = policy[s]
            arrow = env.action_names[a]
            ax.text(c, r + 0.25, arrow, ha='center', va='center',
                    color='white', fontsize=14, fontweight='bold')

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='gray', linewidth=1)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    return im


def visualize_results(env, V, policy, history):
    """可视化：V + 策略、收敛曲线、迭代快照。"""
    fig = plt.figure(figsize=(16, 4.8))

    # ---- (a) 最终 V 与策略 ----
    ax = fig.add_subplot(1, 3, 1)
    draw_grid(ax, env, V, policy, '最优 V* 与策略 π*')

    # ---- (b) 收敛曲线 ----
    ax = fig.add_subplot(1, 3, 2)
    # 用 V 的 L∞ 变化量作为收敛度量
    diffs = [np.max(np.abs(history[i + 1] - history[i]))
             for i in range(len(history) - 1)]
    ax.semilogy(diffs, color='tab:red')
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('max|V_new - V_old|（对数）')
    ax.set_title('值迭代的收敛过程')
    ax.grid(True, linestyle='--', alpha=0.5)

    # ---- (c) 中间快照 ----
    ax = fig.add_subplot(1, 3, 3)
    # 取前几次迭代的 V 作为快照，展示 V 如何"扩散"到整个网格
    if len(history) > 5:
        snap_idx = [0, len(history) // 5, len(history) // 2, len(history) - 1]
    else:
        snap_idx = list(range(len(history)))
    snap_V = history[snap_idx[-1]]
    draw_grid(ax, env, snap_V, policy, f'迭代 {snap_idx[-1]} 时的 V')

    fig.tight_layout()
    out_path = Path(__file__).parent / 'gridworld_value_iteration.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'图片已保存为 {out_path}')


# ============ 五、主流程 ============
def main():
    np.set_printoptions(precision=4, suppress=True)

    print('=' * 60)
    print('GridWorld 值迭代')
    print('=' * 60)

    # ---- 创建环境 ----
    env = GridWorld(size=4, gamma=0.99)
    print(f'网格大小: {env.size}×{env.size}')
    print(f'状态数: {env.n_states}, 动作数: {env.n_actions}')
    print(f'折扣因子 γ = {env.gamma}')
    print(f'起点: {env.start}, 终点: {env.goal}\n')

    # ---- 运行值迭代 ----
    V, policy, n_iter, history = value_iteration(env, tol=1e-6, max_iter=1000)
    print(f'值迭代收敛，用了 {n_iter} 轮')

    # ---- 打印结果 ----
    print(f'\n最优状态价值 V*（按网格排列）:')
    V_grid = V.reshape(env.size, env.size)
    for r in range(env.size):
        row = '  ' + '  '.join([f'{V_grid[r, c]:>7.2f}' for c in range(env.size)])
        print(row)

    print(f'\n最优策略 π*（按网格排列，箭头表示移动方向）:')
    for r in range(env.size):
        row = '  '
        for c in range(env.size):
            s = env.rc_to_state(r, c)
            if env.is_terminal(s):
                row += f'{"GOAL":^6}'
            else:
                row += f'{env.action_names[policy[s]]:^6}'
        print(row)

    # ---- 验证：用最优策略做 policy evaluation，应该等于 V* ----
    V_from_policy = evaluate_policy(env, policy)
    print(f'\n验证：用 π* 做 policy evaluation')
    print(f'  V* 和 V^π* 的最大差异: {np.max(np.abs(V - V_from_policy)):.2e}')
    print(f'  （应接近 0，说明值迭代的结果是一致的）')

    # ---- 可视化 ----
    visualize_results(env, V, policy, history)

    plt.show()


if __name__ == '__main__':
    main()