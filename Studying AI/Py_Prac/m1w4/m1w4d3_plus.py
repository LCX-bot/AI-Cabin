# 学习内容：四元数
# 实践任务：四元数与旋转矩阵互转，写注释说明

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R, Slerp

# 让中文输出在 UTF-8 终端中正常显示（Windows 默认用 GBK，否则会乱码）
sys.stdout.reconfigure(encoding='utf-8')
# Windows 系统自带中文字体，优先用"微软雅黑"，它覆盖的汉字最全
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
# 负号在默认字体里会被渲染成方块，需要额外指定它用拉丁字体
plt.rcParams['axes.unicode_minus'] = False


# ============ 一、四元数的基本定义与性质 ============
def demo_quaternion_basics():
    """演示四元数的结构、单位约束、共轭、逆，以及它与轴角的关系。"""
    print('=' * 60)
    print('一、四元数的基本定义与性质')
    print('=' * 60)

    # ---- 四元数结构：q = w + xi + yj + zk，用 (x, y, z, w) 四个实数表示 ----
    # scipy 中四元数的顺序是 (x, y, z, w)，实部 w 在最后一维
    # 这里构造一个单位四元数：绕 z 轴转 90°
    axis = np.array([0.0, 0.0, 1.0])                    # 单位旋转轴
    angle = np.deg2rad(90.0)                            # 旋转角（弧度）

    # ---- 用轴角公式手动构造四元数 ----
    # 公式：q = [axis * sin(θ/2), cos(θ/2)]
    q_manual = np.empty(4)
    q_manual[:3] = axis * np.sin(angle / 2.0)           # 虚部
    q_manual[3] = np.cos(angle / 2.0)                   # 实部
    print('手动构造的四元数 (x, y, z, w):', np.round(q_manual, 4))

    # ---- 用 scipy 从轴角构造，验证是否一致 ----
    r = R.from_rotvec(axis * angle)                     # 旋转向量 = 轴 * 角度
    q_scipy = r.as_quat()                               # 导出为四元数
    print('scipy 构造的四元数 (x, y, z, w):', np.round(q_scipy, 4))

    # ---- 单位约束：四元数必须满足 x² + y² + z² + w² = 1 ----
    norm = np.linalg.norm(q_manual)
    print(f'\n四元数的模长: {norm:.6f}（应等于 1，这是单位四元数的约束）')

    # ---- 共轭和逆 ----
    # 共轭：q* = (-x, -y, -z, w)，即虚部取反、实部不变
    q_conj = np.array([-q_scipy[0], -q_scipy[1], -q_scipy[2], q_scipy[3]])
    print('共轭四元数:', np.round(q_conj, 4))

    # 对于单位四元数，逆 = 共轭
    # 也就是说：q⁻¹ = q*，这大幅简化了计算
    r_inv = r.inv()
    q_inv = r_inv.as_quat()
    print('逆四元数 (应等于共轭):', np.round(q_inv, 4))

    # ---- 验证：q 和 q⁻¹ 相乘得到单位旋转 ----
    r_identity = r * r_inv
    print('\nq * q⁻¹ 得到的旋转（应为单位矩阵）:\n', np.round(r_identity.as_matrix(), 4))


# ============ 二、四元数与旋转矩阵互转 ============
def demo_quat_matrix_conversion():
    """演示四元数与 3x3 旋转矩阵的双向转换，并手写一遍公式验证。"""
    print('\n' + '=' * 60)
    print('二、四元数与旋转矩阵互转')
    print('=' * 60)

    # ---- 用四元数构造一个旋转 ----
    q = np.array([0.2, 0.3, 0.4, 0.8426])               # 近似单位四元数
    q = q / np.linalg.norm(q)                           # 归一化，确保是单位四元数
    print('输入四元数 (x, y, z, w):', np.round(q, 4))

    # ---- 方式 1：用 scipy 转换 ----
    r = R.from_quat(q)
    M_scipy = r.as_matrix()
    print('\nscipy 转换得到的旋转矩阵:\n', np.round(M_scipy, 4))

    # ---- 方式 2：手写公式验证 ----
    # 设 q = (x, y, z, w)，对应的旋转矩阵为：
    #       [1-2(y²+z²)   2(xy-zw)    2(xz+yw)  ]
    # R =   [2(xy+zw)     1-2(x²+z²)  2(yz-xw)  ]
    #       [2(xz-yw)     2(yz+xw)    1-2(x²+y²)]
    x, y, z, w = q
    M_manual = np.array([
        [1 - 2 * (y**2 + z**2),  2 * (x * y - z * w),      2 * (x * z + y * w)],
        [2 * (x * y + z * w),    1 - 2 * (x**2 + z**2),    2 * (y * z - x * w)],
        [2 * (x * z - y * w),    2 * (y * z + x * w),      1 - 2 * (x**2 + y**2)],
    ])
    print('手写公式得到的旋转矩阵:\n', np.round(M_manual, 4))

    # ---- 验证两者一致 ----
    diff = np.max(np.abs(M_scipy - M_manual))
    print(f'\n两种方法的最大差异: {diff:.2e}（应接近 0）')

    # ---- 从旋转矩阵反推四元数 ----
    # 用 scipy
    q_back = R.from_matrix(M_scipy).as_quat()
    print('\n从矩阵反推的四元数 (x, y, z, w):', np.round(q_back, 4))
    print('与输入一致或互为相反数（q 和 -q 表示同一旋转）')


# ============ 三、四元数乘法：旋转的合成 ============
def demo_quaternion_composition():
    """演示四元数乘法如何对应旋转合成，以及为什么它天然无奇异。"""
    print('\n' + '=' * 60)
    print('三、四元数乘法与旋转合成')
    print('=' * 60)

    # ---- 两个基本旋转 ----
    r1 = R.from_euler('Z', 90.0, degrees=True)          # 绕 Z 转 90°
    r2 = R.from_euler('X', 90.0, degrees=True)          # 绕 X 转 90°

    # ---- 组合旋转（scipy 中 * 等价于矩阵 @，先 r2 后 r1） ----
    r_combined = r1 * r2

    # ---- 用四元数表示组合 ----
    q1 = r1.as_quat()
    q2 = r2.as_quat()
    q_combined = r_combined.as_quat()
    print('r1 的四元数 (x,y,z,w):', np.round(q1, 4))
    print('r2 的四元数 (x,y,z,w):', np.round(q2, 4))
    print('合成 r1*r2 的四元数:', np.round(q_combined, 4))

    # ---- 手写四元数乘法验证 ----
    # 四元数乘法：q1 * q2 = (w1 w2 - v1·v2, w1 v2 + w2 v1 + v1×v2)
    # 其中 v 表示虚部向量 (x, y, z)
    def quat_multiply(a, b):
        """四元数乘法 a * b，参数和返回均为 (x, y, z, w) 顺序。"""
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        # 实部：w1 w2 - v1·v2
        w = aw * bw - (ax * bx + ay * by + az * bz)
        # 虚部：w1 v2 + w2 v1 + v1×v2
        x = aw * bx + bw * ax + (ay * bz - az * by)
        y = aw * by + bw * ay + (az * bx - ax * bz)
        z = aw * bz + bw * az + (ax * by - ay * bx)
        return np.array([x, y, z, w])

    q_manual = quat_multiply(q1, q2)
    # 注意：q 和 -q 表示同一个旋转，此处允许符号差异
    if np.dot(q_manual, q_combined) < 0:
        q_manual = -q_manual
    print('\n手写四元数乘法结果:', np.round(q_manual, 4))
    print(f'与 scipy 结果的最大差异: {np.max(np.abs(q_manual - q_combined)):.2e}')

    # ---- 演示：四元数乘法连续且无奇异 ----
    # 连续改变中间角度，观察四元数的四个分量都平滑变化
    print('\n连续改变中间角 β，观察四元数的平滑性：')
    betas = np.array([88.0, 89.0, 89.9, 90.0, 90.1, 91.0, 92.0])
    for beta in betas:
        r = R.from_euler('ZYX', [30.0, beta, 60.0], degrees=True)
        q = r.as_quat()
        print(f'  β = {beta:5.1f}°  → q = {np.round(q, 4)}')
    print('（每个分量都连续变化，无跳变，说明四元数没有万向节锁）')


# ============ 四、四元数插值 SLERP ============
def demo_slerp():
    """演示四元数球面线性插值，用于姿态平滑过渡。"""
    print('\n' + '=' * 60)
    print('四、四元数 SLERP 插值')
    print('=' * 60)

    # ---- 两个关键姿态 ----
    r_start = R.from_euler('ZYX', [0.0, 0.0, 0.0], degrees=True)      # 单位姿态
    r_end = R.from_euler('ZYX', [90.0, 45.0, 30.0], degrees=True)     # 复杂姿态

    # ---- 用 Slerp 类做插值 ----
    # Slerp 的第一个参数是关键帧的时间点，第二个参数是对应的旋转对象
    key_times = np.array([0.0, 1.0])                                  # 关键帧时间
    key_rots = R.concatenate([r_start, r_end])                        # 关键帧旋转
    slerp = Slerp(key_times, key_rots)                                # 创建插值器对象

    # ---- 在指定时间点上插值 ----
    times = np.linspace(0, 1, 11)                                     # 插值时间点
    interp_rots = slerp(times)                                        # 调用插值器，返回 Rotation
    print('单位向量 [1,0,0] 在插值过程中的位置:')
    v = np.array([1.0, 0.0, 0.0])
    traj = interp_rots.apply(v)                                       # (11, 3)
    for t, p in zip(times, traj):
        print(f'  t = {t:.1f}: {np.round(p, 4)}')

    # ---- 可视化 ----
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 画一个参考的球面线框
    u = np.linspace(0, 2 * np.pi, 30)
    vv = np.linspace(0, np.pi, 15)
    xs = np.outer(np.cos(u), np.sin(vv))
    ys = np.outer(np.sin(u), np.sin(vv))
    zs = np.outer(np.ones_like(u), np.cos(vv))
    ax.plot_wireframe(xs, ys, zs, color='lightgray', alpha=0.3, linewidth=0.5)

    # 画坐标轴
    for axis, color, name in [(np.array([1., 0, 0]), 'r', 'x'),
                              (np.array([0, 1., 0]), 'g', 'y'),
                              (np.array([0, 0, 1.]), 'b', 'z')]:
        ax.quiver(0, 0, 0, *axis, color=color, arrow_length_ratio=0.1, linewidth=1.5)
        ax.text(*axis * 1.1, name, color=color, fontsize=12)

    # 画轨迹
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'o-', color='tab:purple',
            linewidth=2, markersize=6, label='SLERP 轨迹')
    ax.scatter(*traj[0], color='tab:green', s=100, label='起点', zorder=5)
    ax.scatter(*traj[-1], color='tab:red', s=100, label='终点', zorder=5)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('四元数 SLERP 插值：单位向量在球面上的轨迹')
    ax.legend(loc='upper left', fontsize=9)

    out_path = Path(__file__).parent / 'quaternion_slerp.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'\n插值轨迹图已保存为 {out_path}')


# ============ 五、主流程 ============
def main():
    np.set_printoptions(precision=4, suppress=True)

    demo_quaternion_basics()
    demo_quat_matrix_conversion()
    demo_quaternion_composition()
    demo_slerp()

    plt.show()


if __name__ == '__main__':
    main()