# 学习内容：坐标系、齐次变换
# 实践任务：写 2D/3D 齐次变换函数

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  注册 3D 投影

# 让中文输出在 UTF-8 终端中正常显示（Windows 默认用 GBK，否则会乱码）
sys.stdout.reconfigure(encoding='utf-8')
# Windows 系统自带中文字体，优先用"微软雅黑"，它覆盖的汉字最全
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
# 负号在默认字体里会被渲染成方块，需要额外指定它用拉丁字体
plt.rcParams['axes.unicode_minus'] = False


# ============ 一、2D 齐次变换 ============
def translation_2d(tx, ty):
    """2D 平移矩阵（3x3 齐次形式）。

    效果：点 (x, y) → (x + tx, y + ty)
    """
    return np.array([
        [1.0, 0.0, tx],
        [0.0, 1.0, ty],
        [0.0, 0.0, 1.0],
    ])


def rotation_2d(theta):
    """2D 旋转矩阵（3x3 齐次形式），theta 为弧度，逆时针为正。"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def scaling_2d(sx, sy=None):
    """2D 缩放矩阵（3x3 齐次形式）。

    参数：
        sx: x 方向缩放因子
        sy: y 方向缩放因子；若为 None，则与 sx 相同（等比缩放）
    """
    if sy is None:
        sy = sx
    return np.array([
        [sx,  0.0, 0.0],
        [0.0, sy,  0.0],
        [0.0, 0.0, 1.0],
    ])


# ============ 二、3D 齐次变换 ============
def translation_3d(tx, ty, tz):
    """3D 平移矩阵（4x4 齐次形式）。"""
    return np.array([
        [1.0, 0.0, 0.0, tx],
        [0.0, 1.0, 0.0, ty],
        [0.0, 0.0, 1.0, tz],
        [0.0, 0.0, 0.0, 1.0],
    ])


def rotation_x(theta):
    """绕 X 轴旋转 theta 弧度（4x4 齐次形式）。"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0,  c,  -s,  0.0],
        [0.0,  s,   c,  0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def rotation_y(theta):
    """绕 Y 轴旋转 theta 弧度（4x4 齐次形式）。"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c,  0.0,  s,  0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s,  0.0,  c,  0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def rotation_z(theta):
    """绕 Z 轴旋转 theta 弧度（4x4 齐次形式）。"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c,  -s,  0.0, 0.0],
        [ s,   c,  0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def scaling_3d(sx, sy=None, sz=None):
    """3D 缩放矩阵（4x4 齐次形式）。

    参数：
        sx: x 方向缩放因子
        sy: y 方向缩放因子；若为 None，则与 sx 相同
        sz: z 方向缩放因子；若为 None，则与 sx 相同
    """
    if sy is None:
        sy = sx
    if sz is None:
        sz = sx
    return np.array([
        [sx,  0.0, 0.0, 0.0],
        [0.0, sy,  0.0, 0.0],
        [0.0, 0.0, sz,  0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


# ============ 三、点变换工具 ============
def apply_transform_2d(points, T):
    """对 2D 点集应用齐次变换。

    参数：
        points: 形状 (N, 2) 或 (2,) 的点集
        T:      形状 (3, 3) 的齐次变换矩阵
    返回：
        变换后的点集，形状与输入一致
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))      # (N, 2) 这一行代码用于输入整形
    ones = np.ones((points.shape[0], 1))                         # (N, 1)
    homo = np.hstack([points, ones])                             # (N, 3)
    transformed = homo @ T.T                                     # 右乘 T 的转置
    return transformed[:, :2]
    # 这里是线性变换的转置

def apply_transform_3d(points, T):
    """对 3D 点集应用齐次变换。

    参数：
        points: 形状 (N, 3) 或 (3,) 的点集
        T:      形状 (4, 4) 的齐次变换矩阵
    返回：
        变换后的点集，形状与输入一致
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))      # (N, 3)
    ones = np.ones((points.shape[0], 1))                         # (N, 1)
    homo = np.hstack([points, ones])                             # (N, 4)
    transformed = homo @ T.T                                     # 右乘 T 的转置
    return transformed[:, :3]


def compose(*transforms):
    """按顺序组合多个齐次变换矩阵。

    返回 T_n @ ... @ T_2 @ T_1，
    表示"先应用 T_1，再应用 T_2，...，最后应用 T_n"。
    矩阵乘法从右往左作用在点上。
    """
    result = np.eye(transforms[0].shape[0])
    for T in transforms:
        result = T @ result
    return result


# ============ 四、可视化 ============
def draw_frame_2d(ax, T, label, color='k', length=0.6, lw=1.8):
    """在 2D 坐标轴上画出 T 表示的局部坐标系（原点 + x/y 轴箭头）。"""
    origin = T[:2, 2]                                  # 原点位置
    x_axis = T[:2, 0] * length                         # x 轴方向 * 长度
    y_axis = T[:2, 1] * length                         # y 轴方向 * 长度

    ax.annotate('', xy=origin + x_axis, xytext=origin,
                arrowprops=dict(arrowstyle='-|>', color='r', lw=lw))
    ax.annotate('', xy=origin + y_axis, xytext=origin,
                arrowprops=dict(arrowstyle='-|>', color='g', lw=lw))
    ax.plot(*origin, 'o', color=color, markersize=5)
    ax.text(origin[0] + 0.05, origin[1] + 0.05, label,
            color=color, fontsize=9)


def demo_2d():
    """演示 2D 齐次变换：对一组点先旋转、再平移，并画出坐标轴。"""
    # 构造一个矩形点集（单位正方形）
    square = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])

    theta = np.deg2rad(45.0)                            # 旋转 45 度
    T_rot = rotation_2d(theta)                          # 只旋转
    # 从固定坐标系看是“先平移再旋转”，从运动坐标系看则是“先旋转再平移”
    T_rot_trans = compose(translation_2d(2.5, 0.5), T_rot)  # 先旋转，再平移

    square_rot = apply_transform_2d(square, T_rot)
    square_rot_trans = apply_transform_2d(square, T_rot_trans)

    fig, ax = plt.subplots(figsize=(7, 6))

    # 画原始矩形
    ax.plot(*np.vstack([square, square[:1]]).T, '-o', color='tab:blue',
            label='原始矩形', markersize=4)

    # 画旋转后的矩形
    ax.plot(*np.vstack([square_rot, square_rot[:1]]).T, '-o', color='tab:orange',
            label='旋转 45°', markersize=4)

    # 画旋转 + 平移后的矩形
    ax.plot(*np.vstack([square_rot_trans, square_rot_trans[:1]]).T,
            '-o', color='tab:red', label='旋转 45° + 平移 (2.5, 0.5)',
            markersize=4)

    # 画出三个坐标系（局部坐标系）
    draw_frame_2d(ax, np.eye(3), '世界系', color='k')
    draw_frame_2d(ax, T_rot, '旋转后', color='tab:orange')
    draw_frame_2d(ax, T_rot_trans, '平移后', color='tab:red')

    ax.set_aspect('equal')
    ax.set_xlim(-1.0, 5.0)
    ax.set_ylim(-1.0, 3.0)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(0.0, color='gray', lw=0.5)
    ax.axvline(0.0, color='gray', lw=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('2D 齐次变换：旋转 45° 后平移')
    ax.legend(loc='upper left', fontsize=8)

    out_path = Path(__file__).parent / 'transform_2d.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'2D 演示图已保存为 {out_path}')


def draw_frame_3d(ax, T, label, length=0.6, lw=1.5):
    """在 3D 坐标轴上画出 T 表示的局部坐标系（原点 + x/y/z 轴箭头）。"""
    origin = T[:3, 3]
    for i, (color, name) in enumerate([('r', 'x'), ('g', 'y'), ('b', 'z')]):
        axis = T[:3, i] * length
        ax.quiver(*origin, *axis, color=color, linewidth=lw, arrow_length_ratio=0.15)
    ax.text(origin[0], origin[1], origin[2], label, fontsize=8)


def demo_3d():
    """演示 3D 齐次变换：立方体绕自身中心旋转。"""
    # 立方体的 8 个顶点
    cube = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],   # 底面
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],   # 顶面
    ], dtype=float)

    # 立方体 12 条棱，每条由两个顶点索引构成
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),               # 底面
        (4, 5), (5, 6), (6, 7), (7, 4),               # 顶面
        (0, 4), (1, 5), (2, 6), (3, 7),               # 竖直棱
    ]

    theta = np.deg2rad(30.0)
    # 先绕 z 轴转，再绕 x 轴转，再平移到原点附近
    T = compose(
        translation_3d(2.0, 0.0, 0.0),
        rotation_x(theta),
        rotation_z(theta * 2),
    )

    cube_t = apply_transform_3d(cube, T)

    fig = plt.figure(figsize=(12, 6))

    # 左侧：原始立方体
    ax = fig.add_subplot(121, projection='3d')
    for i, j in edges:
        ax.plot3D(*zip(cube[i], cube[j]), color='tab:blue', lw=1.5)
    draw_frame_3d(ax, np.eye(4), '世界系')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('原始立方体')
    ax.view_init(elev=20, azim=30)

    # 右侧：变换后的立方体
    ax = fig.add_subplot(122, projection='3d')
    for i, j in edges:
        ax.plot3D(*zip(cube_t[i], cube_t[j]), color='tab:red', lw=1.5)
    draw_frame_3d(ax, T, '变换后')
    ax.set_xlim(0, 4)
    ax.set_ylim(-1, 3)
    ax.set_zlim(-1, 3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('绕 Z 转 60° → 绕 X 转 30° → 平移 (2, 0, 0)')
    ax.view_init(elev=20, azim=30)

    out_path = Path(__file__).parent / 'transform_3d.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'3D 演示图已保存为 {out_path}')


# ============ 五、主流程 ============
def main():
    np.set_printoptions(precision=4, suppress=True)

    # ---- 演示 1：查看 2D 变换矩阵的形态 ----
    print('===== 2D 齐次变换矩阵 =====')
    print('平移 (2, 3):\n', translation_2d(2.0, 3.0))
    print('旋转 30°:\n', rotation_2d(np.deg2rad(30.0)))
    print('缩放 (2, 0.5):\n', scaling_2d(2.0, 0.5))

    # ---- 演示 2：组合多个变换 ----
    print('\n===== 组合变换：先旋转 45°，再平移 (2, 1) =====')
    T = compose(translation_2d(2.0, 1.0), rotation_2d(np.deg2rad(45.0)))
    print(T)

    # 对点 (1, 0) 应用这个组合变换
    p = np.array([1.0, 0.0])
    p_t = apply_transform_2d(p, T)[0]
    print(f'点 {p} 变换后 → {p_t}')

    # ---- 演示 3：3D 变换矩阵 ----
    print('\n===== 3D 齐次变换矩阵 =====')
    print('绕 X 轴旋转 90°:\n', rotation_x(np.deg2rad(90.0)))
    print('绕 Z 轴旋转 45°:\n', rotation_z(np.deg2rad(45.0)))

    T3 = compose(rotation_y(np.deg2rad(30.0)), translation_3d(1.0, 2.0, 3.0))
    print('组合变换（先平移，再绕 Y 转 30°）:\n', T3)

    # 对 3D 点 (1, 0, 0) 应用
    p3 = np.array([1.0, 0.0, 0.0])
    p3_t = apply_transform_3d(p3, T3)[0]
    print(f'点 {p3} 变换后 → {p3_t}')

    # ---- 演示 4：可视化 ----
    demo_2d()
    demo_3d()
    plt.show()


if __name__ == '__main__':
    main()