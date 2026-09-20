# 学习内容：2DOF 机械臂正运动学
# 实践任务：推导并实现 FK

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


# ============ 一、基础齐次变换 ============
def rotation_2d(theta):
    """2D 旋转矩阵（3x3 齐次形式），逆时针为正。"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def translation_2d(tx, ty):
    """2D 平移矩阵（3x3 齐次形式）。"""
    return np.array([
        [1.0, 0.0, tx],
        [0.0, 1.0, ty],
        [0.0, 0.0, 1.0],
    ])


# ============ 二、正运动学推导与实现 ============
def link_transform(length, theta):
    """单个连杆的齐次变换。

    矩阵形式：
        T = Rot(theta) @ Trans(length, 0)

    几何含义：
        - 新坐标系的 x 轴方向由 theta 决定（相对父坐标系逆时针转 theta）。
        - 新坐标系的原点在父坐标系里位于 (length * cosθ, length * sinθ)，
          即沿新 x 轴走 length 到达连杆末端。
        - 作用在原点 (0,0,1) 上，得到连杆末端在父坐标系中的位置。

    展开后：
        T = [[cosθ, -sinθ, length*cosθ],
             [sinθ,  cosθ, length*sinθ],
             [0,     0,    1              ]]
    """
    return rotation_2d(theta) @ translation_2d(length, 0.0)


def forward_kinematics_2dof(l1, l2, theta1, theta2):
    """2DOF 平面机械臂正运动学。

    参数：
        l1, l2  : 两段连杆的长度
        theta1  : 第一关节角（相对世界坐标系 x 轴，逆时针为正）
        theta2  : 第二关节角（相对第一连杆方向，逆时针为正）

    返回：
        p_ee   : 末端执行器位置 (x, y)，形状 (2,)
        p_j1   : 第一关节（肘部）位置 (x, y)，形状 (2,)
        T01    : 第一连杆末端的齐次变换矩阵（肘部坐标系）
        T02    : 第二连杆末端的齐次变换矩阵（末端坐标系）

    推导：
        T_0_1 = Rot(theta1) @ Trans(l1, 0)
        T_1_2 = Rot(theta2) @ Trans(l2, 0)
        T_0_2 = T_0_1 @ T_1_2

        末端位置 = T_0_2 的平移列，展开后为：
            x = l1 cos(theta1) + l2 cos(theta1 + theta2)
            y = l1 sin(theta1) + l2 sin(theta1 + theta2)
    """
    T_0_1 = link_transform(l1, theta1)              # 第一连杆变换
    T_1_2 = link_transform(l2, theta2)              # 第二连杆变换
    T_0_2 = T_0_1 @ T_1_2                           # 整链复合变换

    # 肘部位置（第一连杆末端）
    p_j1 = T_0_1[:2, 2]
    # 末端位置（第二连杆末端）
    p_ee = T_0_2[:2, 2]

    return p_ee, p_j1, T_0_1, T_0_2


def forward_kinematics_2dof_analytic(l1, l2, theta1, theta2):
    """2DOF 平面机械臂正运动学的解析公式（用于与矩阵法交叉验证）。

    x = l1 * cos(theta1) + l2 * cos(theta1 + theta2)
    y = l1 * sin(theta1) + l2 * sin(theta1 + theta2)
    """
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    return np.array([x, y])


# ============ 三、绘制机械臂 ============
def draw_arm_2dof(ax, l1, l2, theta1, theta2, color='tab:blue', label=None):
    """在给定坐标轴上绘制 2DOF 机械臂。"""
    p_ee, p_j1, _, _ = forward_kinematics_2dof(l1, l2, theta1, theta2)
    base = np.array([0.0, 0.0])

    # 画两个连杆（用粗线表示）
    ax.plot([base[0], p_j1[0]], [base[1], p_j1[1]],
            '-', color=color, linewidth=6, solid_capstyle='round',
            label=label)
    ax.plot([p_j1[0], p_ee[0]], [p_j1[1], p_ee[1]],
            '-', color=color, linewidth=6, solid_capstyle='round', alpha=0.7)

    # 画三个关节点（基座、肘部、末端）
    ax.plot(*base, 'o', color='k', markersize=10, zorder=5)
    ax.plot(*p_j1, 'o', color='w', markeredgecolor='k', markersize=8, zorder=5)
    ax.plot(*p_ee, 'o', color='tab:red', markersize=8, zorder=5)


# ============ 四、可视化：单个姿态 + 工作空间 ============
def demo_single_pose():
    """演示单个姿态：解析法与矩阵法结果对照，并绘制机械臂姿态。"""
    print('=' * 60)
    print('一、正运动学：单姿态验证')
    print('=' * 60)

    l1, l2 = 1.0, 0.8                          # 两段连杆长度
    theta1 = np.deg2rad(45.0)                  # 第一关节角
    theta2 = np.deg2rad(30.0)                  # 第二关节角

    # ---- 矩阵法 ----
    p_ee, p_j1, T_0_1, T_0_2 = forward_kinematics_2dof(l1, l2, theta1, theta2)

    # ---- 解析法 ----
    p_ee_analytic = forward_kinematics_2dof_analytic(l1, l2, theta1, theta2)

    print(f'连杆长度: l1 = {l1}, l2 = {l2}')
    print(f'关节角度: theta1 = {np.rad2deg(theta1):.1f}°, '
          f'theta2 = {np.rad2deg(theta2):.1f}°')
    print(f'肘部位置 p_j1     : {np.round(p_j1, 4)}')
    print(f'末端位置（矩阵法）: {np.round(p_ee, 4)}')
    print(f'末端位置（解析法）: {np.round(p_ee_analytic, 4)}')
    print(f'两种方法差异      : {np.max(np.abs(p_ee - p_ee_analytic)):.2e}')

    print(f'\n末端齐次变换矩阵 T_0_2:\n{np.round(T_0_2, 4)}')

    # ---- 验证矩阵含义：前三列是坐标轴方向，最后一列是位置 ----
    print(f'\nT_0_2 的最后一列（末端位置）: {np.round(T_0_2[:2, 2], 4)}')
    print(f'T_0_2 的第一列（末端 x 轴方向）: {np.round(T_0_2[:2, 0], 4)}')

    # ---- 绘制 ----
    fig, ax = plt.subplots(figsize=(7, 7))

    draw_arm_2dof(ax, l1, l2, theta1, theta2, color='tab:blue', label='当前姿态')

    # 画出基座坐标系（世界系）
    ax.quiver(0, 0, 0.4, 0, color='r', angles='xy', scale_units='xy', scale=1,
              width=0.008, zorder=4)
    ax.quiver(0, 0, 0, 0.4, color='g', angles='xy', scale_units='xy', scale=1,
              width=0.008, zorder=4)
    ax.text(0.42, 0, 'x', color='r', fontsize=11)
    ax.text(0, 0.42, 'y', color='g', fontsize=11)

    # 标注末端位置
    ax.annotate(f'EE ({p_ee[0]:.3f}, {p_ee[1]:.3f})',
                xy=p_ee, xytext=(p_ee[0] + 0.1, p_ee[1] + 0.15),
                fontsize=9, color='tab:red')

    ax.set_aspect('equal')
    ax.set_xlim(-0.5, 2.2)
    ax.set_ylim(-0.5, 2.2)
    ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
    ax.axvline(0, color='gray', lw=0.5, alpha=0.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'2DOF 机械臂：θ1={np.rad2deg(theta1):.0f}°, '
                 f'θ2={np.rad2deg(theta2):.0f}°')
    ax.legend(loc='upper right', fontsize=9)

    out_path = Path(__file__).parent / 'fk_single_pose.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\n单姿态图已保存为 {out_path}')


def demo_workspace():
    """演示机械臂的可达工作空间（末端位置集合）。"""
    print('\n' + '=' * 60)
    print('二、工作空间可视化')
    print('=' * 60)

    l1, l2 = 1.0, 0.8

    # 穷举所有 (theta1, theta2) 组合下的末端位置
    theta1_grid = np.linspace(-np.pi, np.pi, 200)
    theta2_grid = np.linspace(-np.pi, np.pi, 200)
    T1, T2 = np.meshgrid(theta1_grid, theta2_grid)

    # 用解析公式直接批量计算（比循环调用 FK 快得多）
    x = l1 * np.cos(T1) + l2 * np.cos(T1 + T2)
    y = l1 * np.sin(T1) + l2 * np.sin(T1 + T2)

    # ---- 理论边界 ----
    # 最大半径：两连杆完全伸直，r_max = l1 + l2
    # 最小半径：两连杆反向折叠，r_min = |l1 - l2|
    r_max = l1 + l2
    r_min = abs(l1 - l2)
    print(f'可达半径范围: [{r_min:.2f}, {r_max:.2f}]')

    # ---- 绘制 ----
    fig, ax = plt.subplots(figsize=(7, 7))

    # 画工作空间点云（用散点密度体现可达性）
    ax.scatter(x.ravel(), y.ravel(), s=0.3, alpha=0.15, color='tab:blue')

    # 画出理论上两个边界圆（虚线）
    theta_circle = np.linspace(0, 2 * np.pi, 200)
    ax.plot(r_max * np.cos(theta_circle), r_max * np.sin(theta_circle),
            'r--', linewidth=1.5, label=f'外边界 r = l1 + l2 = {r_max}')
    ax.plot(r_min * np.cos(theta_circle), r_min * np.sin(theta_circle),
            'g--', linewidth=1.5, label=f'内边界 r = |l1 - l2| = {r_min}')

    ax.set_aspect('equal')
    ax.set_xlim(-2.1, 2.1)
    ax.set_ylim(-2.1, 2.1)
    ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
    ax.axvline(0, color='gray', lw=0.5, alpha=0.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'2DOF 机械臂工作空间 (l1={l1}, l2={l2})')
    ax.legend(loc='upper right', fontsize=9)

    out_path = Path(__file__).parent / 'fk_workspace.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'工作空间图已保存为 {out_path}')


def demo_multi_pose():
    """演示多姿态：绘制多个不同关节角下的机械臂姿态。"""
    print('\n' + '=' * 60)
    print('三、多姿态对比')
    print('=' * 60)

    l1, l2 = 1.0, 0.8

    # 挑选几个代表性姿态：角度单位是度
    poses = [
        (0.0,   0.0),      # 完全伸直，沿 x 轴
        (45.0,  30.0),     # 常规姿态
        (90.0,  60.0),     # 向上弯折
        (135.0, -90.0),    # 肘部反向
        (0.0,   90.0),     # 末端向上
    ]

    fig, axes = plt.subplots(1, len(poses), figsize=(4 * len(poses), 4.5))

    for ax, (t1_deg, t2_deg) in zip(axes, poses):
        t1 = np.deg2rad(t1_deg)
        t2 = np.deg2rad(t2_deg)
        draw_arm_2dof(ax, l1, l2, t1, t2, color='tab:blue')

        # 计算末端位置标注
        p_ee, _, _, _ = forward_kinematics_2dof(l1, l2, t1, t2)

        ax.set_aspect('equal')
        ax.set_xlim(-0.5, 2.0)
        ax.set_ylim(-1.5, 2.0)
        ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
        ax.axvline(0, color='gray', lw=0.5, alpha=0.5)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'θ1={t1_deg:.0f}°, θ2={t2_deg:.0f}°\n'
                     f'EE=({p_ee[0]:.2f}, {p_ee[1]:.2f})', fontsize=10)

    out_path = Path(__file__).parent / 'fk_multi_pose.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'多姿态图已保存为 {out_path}')


# ============ 五、主流程 ============
def main():
    np.set_printoptions(precision=4, suppress=True)

    demo_single_pose()
    demo_workspace()
    demo_multi_pose()

    plt.show()


if __name__ == '__main__':
    main()