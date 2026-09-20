# 学习内容：FK 可视化
# 实践任务：写 two_link_fk.py，画机械臂和末端轨迹

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


# ============ 二、正运动学 ============
def link_transform(length, theta):
    """单个连杆的齐次变换：先绕 z 轴转 theta，再沿新 x 轴延伸 length。

    矩阵形式：T = Rot(theta) @ Trans(length, 0)
    作用在原点 (0,0,1) 上，得到连杆末端的位置：
        (length·cosθ, length·sinθ, 1)
    """
    return rotation_2d(theta) @ translation_2d(length, 0.0)


def forward_kinematics_2dof(l1, l2, theta1, theta2):
    """2DOF 平面机械臂正运动学。

    参数：
        l1, l2  : 两段连杆的长度
        theta1  : 第一关节角（相对世界 x 轴，逆时针为正）
        theta2  : 第二关节角（相对第一连杆方向，逆时针为正）

    返回：
        p_ee : 末端位置 (x, y)
        p_j1 : 肘部位置 (x, y)
    """
    T_0_1 = link_transform(l1, theta1)
    T_1_2 = link_transform(l2, theta2)
    T_0_2 = T_0_1 @ T_1_2

    p_j1 = T_0_1[:2, 2]
    p_ee = T_0_2[:2, 2]
    return p_ee, p_j1


def forward_kinematics_batch(l1, l2, theta1_arr, theta2_arr):
    """批量正运动学（向量化），输入是等长数组，返回末端点集 (N, 2)。"""
    x = l1 * np.cos(theta1_arr) + l2 * np.cos(theta1_arr + theta2_arr)
    y = l1 * np.sin(theta1_arr) + l2 * np.sin(theta1_arr + theta2_arr)
    return np.stack([x, y], axis=1)


# ============ 三、绘制工具 ============
def draw_arm(ax, l1, l2, theta1, theta2, color='tab:blue', alpha=1.0, lw=5):
    """在 ax 上绘制一个 2DOF 机械臂姿态。"""
    p_ee, p_j1 = forward_kinematics_2dof(l1, l2, theta1, theta2)
    base = np.array([0.0, 0.0])

    ax.plot([base[0], p_j1[0]], [base[1], p_j1[1]],
            '-', color=color, lw=lw, alpha=alpha, solid_capstyle='round')
    ax.plot([p_j1[0], p_ee[0]], [p_j1[1], p_ee[1]],
            '-', color=color, lw=lw, alpha=alpha, solid_capstyle='round')
    ax.plot(*base, 'o', color='k', ms=9, zorder=5)
    ax.plot(*p_j1, 'o', color='w', markeredgecolor='k', ms=7, zorder=5)
    ax.plot(*p_ee, 'o', color='tab:red', ms=7, zorder=5)
    return p_ee


def setup_axes(ax, xlim, ylim, title):
    """统一设置 2D 坐标轴的样式。"""
    ax.set_aspect('equal')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
    ax.axvline(0, color='gray', lw=0.5, alpha=0.5)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)


# ============ 四、演示 1：单个姿态 + 关键点标注 ============
def demo_single_pose():
    """画单个姿态，并标注两个关节和末端位置。"""
    print('=' * 60)
    print('一、单个姿态可视化')
    print('=' * 60)

    l1, l2 = 1.0, 0.8
    t1_deg, t2_deg = 45.0, 30.0
    t1, t2 = np.deg2rad(t1_deg), np.deg2rad(t2_deg)

    p_ee, p_j1 = forward_kinematics_2dof(l1, l2, t1, t2)
    print(f'l1 = {l1}, l2 = {l2}')
    print(f'θ1 = {t1_deg}°, θ2 = {t2_deg}°')
    print(f'肘部位置: {np.round(p_j1, 4)}')
    print(f'末端位置: {np.round(p_ee, 4)}')

    fig, ax = plt.subplots(figsize=(7, 7))
    draw_arm(ax, l1, l2, t1, t2, color='tab:blue')

    # 世界坐标系的小箭头
    ax.quiver(0, 0, 0.4, 0, color='r', angles='xy', scale_units='xy', scale=1,
              width=0.008, zorder=4)
    ax.quiver(0, 0, 0, 0.4, color='g', angles='xy', scale_units='xy', scale=1,
              width=0.008, zorder=4)
    ax.text(0.42, 0, 'x', color='r', fontsize=11)
    ax.text(0, 0.42, 'y', color='g', fontsize=11)

    # 标注文字
    ax.annotate(f'base (0,0)', xy=(0, 0), xytext=(-0.35, -0.25), fontsize=9)
    ax.annotate(f'joint1\n({p_j1[0]:.2f}, {p_j1[1]:.2f})',
                xy=p_j1, xytext=(p_j1[0] - 0.25, p_j1[1] + 0.12), fontsize=9)
    ax.annotate(f'EE\n({p_ee[0]:.2f}, {p_ee[1]:.2f})',
                xy=p_ee, xytext=(p_ee[0] + 0.08, p_ee[1] + 0.15),
                fontsize=9, color='tab:red')

    setup_axes(ax, (-0.6, 2.2), (-0.6, 2.2),
               f'2DOF FK 单姿态：θ1={t1_deg:.0f}°, θ2={t2_deg:.0f}°')

    out_path = Path(__file__).parent / 'fk_single_pose.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'图片已保存为 {out_path}')


# ============ 五、演示 2：末端轨迹（固定 θ2，扫 θ1） ============
def demo_end_effector_trace():
    """固定 θ2，让 θ1 从 0 扫到 2π，画出末端划过的弧线。

    几何直觉：θ2 固定时，末端轨迹是一个以基座为圆心的圆。
    半径 = |l1 · e^{iθ1} + l2 · e^{i(θ1+θ2)}| 是常数。
    """
    print('\n' + '=' * 60)
    print('二、末端轨迹（固定 θ2，扫 θ1）')
    print('=' * 60)

    l1, l2 = 1.0, 0.8

    theta1_arr = np.linspace(0, 2 * np.pi, 400)

    fig, ax = plt.subplots(figsize=(7, 7))

    # 用几个不同的 theta2 值画多条轨迹
    theta2_list = [0.0, np.deg2rad(45), np.deg2rad(90), np.deg2rad(135)]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    labels = ['θ2 = 0°', 'θ2 = 45°', 'θ2 = 90°', 'θ2 = 135°']

    for t2, color, label in zip(theta2_list, colors, labels):
        theta2_arr = np.full_like(theta1_arr, t2)
        pts = forward_kinematics_batch(l1, l2, theta1_arr, theta2_arr)
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=2, label=label)

    # 画一个示例姿态，让图更有"机械臂"的感觉
    t1_ex = np.deg2rad(45)
    t2_ex = np.deg2rad(45)
    draw_arm(ax, l1, l2, t1_ex, t2_ex, color='gray', alpha=0.6, lw=4)
    p_ee_ex, _ = forward_kinematics_2dof(l1, l2, t1_ex, t2_ex)
    ax.plot(*p_ee_ex, 'o', color='gray', ms=10, zorder=6)

    setup_axes(ax, (-2.1, 2.1), (-2.1, 2.1),
               '末端轨迹：固定 θ2，扫 θ1 ∈ [0, 2π]')
    ax.legend(loc='upper right', fontsize=9)

    out_path = Path(__file__).parent / 'fk_ee_trace_theta1.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'图片已保存为 {out_path}')
    print('说明：固定 θ2 时，末端轨迹是一个以基座为圆心的圆，')
    print('      圆的半径由 θ2 决定。')


# ============ 六、演示 3：末端轨迹（固定 θ1，扫 θ2） ============
def demo_end_effector_trace_theta2():
    """固定 θ1，让 θ2 从 0 扫到 2π，画出末端划过的弧线。

    几何直觉：θ1 固定时，末端轨迹是一个以"肘部位置"为圆心的圆，
    半径 = l2。
    """
    print('\n' + '=' * 60)
    print('三、末端轨迹（固定 θ1，扫 θ2）')
    print('=' * 60)

    l1, l2 = 1.0, 0.8

    theta2_arr = np.linspace(0, 2 * np.pi, 400)

    fig, ax = plt.subplots(figsize=(7, 7))

    # 用几个不同的 theta1 值画多条轨迹
    theta1_list = [0.0, np.deg2rad(45), np.deg2rad(90), np.deg2rad(135)]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    labels = ['θ1 = 0°', 'θ1 = 45°', 'θ1 = 90°', 'θ1 = 135°']

    for t1, color, label in zip(theta1_list, colors, labels):
        theta1_arr = np.full_like(theta2_arr, t1)
        pts = forward_kinematics_batch(l1, l2, theta1_arr, theta2_arr)
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=2, label=label)
        # 标注肘部（圆心）
        _, p_j1 = forward_kinematics_2dof(l1, l2, t1, 0.0)
        ax.plot(*p_j1, 'o', color=color, ms=6)

    # 示例姿态
    t1_ex = np.deg2rad(45)
    t2_ex = np.deg2rad(60)
    draw_arm(ax, l1, l2, t1_ex, t2_ex, color='gray', alpha=0.6, lw=4)

    setup_axes(ax, (-2.1, 2.1), (-2.1, 2.1),
               '末端轨迹：固定 θ1，扫 θ2 ∈ [0, 2π]')
    ax.legend(loc='upper right', fontsize=9)

    out_path = Path(__file__).parent / 'fk_ee_trace_theta2.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'图片已保存为 {out_path}')
    print('说明：固定 θ1 时，末端轨迹是一个以肘部为圆心、半径为 l2 的圆。')
    print('      图中的实心圆点就是每条轨迹对应的肘部（圆心）。')


# ============ 七、演示 4：动画式多姿态扫描 ============
def demo_sweep_snapshots():
    """在一张图上画出多个姿态的"残影"，展示运动过程。

    同时对不同 θ2 取值，画出末端轨迹。
    """
    print('\n' + '=' * 60)
    print('四、姿态扫描 + 轨迹叠加')
    print('=' * 60)

    l1, l2 = 1.0, 0.8

    fig, ax = plt.subplots(figsize=(8, 8))

    # ---- 让 θ1 从 0 到 2π，θ2 固定为 -60°，画出多个姿态的残影 ----
    theta2_fixed = np.deg2rad(-60)
    n_snapshots = 12
    theta1_snapshots = np.linspace(0, 2 * np.pi, n_snapshots, endpoint=False)

    for i, t1 in enumerate(theta1_snapshots):
        alpha = 0.15 + 0.45 * (i / n_snapshots)      # 渐变透明度，营造"残影"效果
        draw_arm(ax, l1, l2, t1, theta2_fixed,
                 color='tab:purple', alpha=alpha, lw=4)

    # ---- 叠加末端轨迹 ----
    theta1_sweep = np.linspace(0, 2 * np.pi, 400)
    theta2_arr = np.full_like(theta1_sweep, theta2_fixed)
    pts = forward_kinematics_batch(l1, l2, theta1_sweep, theta2_arr)
    ax.plot(pts[:, 0], pts[:, 1], color='tab:red', lw=2,
            label=f'末端轨迹 (θ2 = -60°)')

    setup_axes(ax, (-2.1, 2.1), (-2.1, 2.1),
               '姿态扫描：θ1 扫一圈，θ2 = -60°')
    ax.legend(loc='upper right', fontsize=9)

    out_path = Path(__file__).parent / 'fk_sweep.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'图片已保存为 {out_path}')


# ============ 八、演示 5：工作空间点云 ============
def demo_workspace_cloud():
    """穷举所有 (θ1, θ2)，画出末端可达点云。"""
    print('\n' + '=' * 60)
    print('五、工作空间点云')
    print('=' * 60)

    l1, l2 = 1.0, 0.8

    # 用较细的网格穷举
    n = 300
    t1_grid = np.linspace(-np.pi, np.pi, n)
    t2_grid = np.linspace(-np.pi, np.pi, n)
    T1, T2 = np.meshgrid(t1_grid, t2_grid)

    pts = forward_kinematics_batch(l1, l2, T1.ravel(), T2.ravel())
    x, y = pts[:, 0], pts[:, 1]

    r = np.hypot(x, y)
    print(f'连杆长度: l1 = {l1}, l2 = {l2}')
    print(f'理论外半径 r_max = l1 + l2 = {l1 + l2}')
    print(f'理论内半径 r_min = |l1 - l2| = {abs(l1 - l2)}')
    print(f'实际点云的 r 范围: [{r.min():.4f}, {r.max():.4f}]')

    fig, ax = plt.subplots(figsize=(8, 8))

    # 用颜色表示半径，便于看结构
    sc = ax.scatter(x, y, c=r, cmap='viridis', s=0.5, alpha=0.4)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('末端到基座距离 r')

    # 理论边界圆
    theta_circle = np.linspace(0, 2 * np.pi, 200)
    ax.plot((l1 + l2) * np.cos(theta_circle), (l1 + l2) * np.sin(theta_circle),
            'r--', lw=1.5, label=f'外边界 r = {l1 + l2}')
    ax.plot(abs(l1 - l2) * np.cos(theta_circle), abs(l1 - l2) * np.sin(theta_circle),
            'g--', lw=1.5, label=f'内边界 r = {abs(l1 - l2)}')

    setup_axes(ax, (-2.1, 2.1), (-2.1, 2.1),
               f'工作空间点云 (l1={l1}, l2={l2})')
    ax.legend(loc='upper right', fontsize=9)

    out_path = Path(__file__).parent / 'fk_workspace_cloud.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'图片已保存为 {out_path}')


# ============ 九、主流程 ============
def main():
    np.set_printoptions(precision=4, suppress=True)

    demo_single_pose()
    demo_end_effector_trace()
    demo_end_effector_trace_theta2()
    demo_sweep_snapshots()
    demo_workspace_cloud()

    plt.show()


if __name__ == '__main__':
    main()

