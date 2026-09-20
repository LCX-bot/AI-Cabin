# 学习内容：旋转矩阵、欧拉角
# 实践任务：用 scipy.spatial.transform.Rotation 做转换

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# 让中文输出在 UTF-8 终端中正常显示（Windows 默认用 GBK，否则会乱码）
sys.stdout.reconfigure(encoding='utf-8')
# Windows 系统自带中文字体，优先用"微软雅黑"，它覆盖的汉字最全
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
# 负号在默认字体里会被渲染成方块，需要额外指定它用拉丁字体
plt.rcParams['axes.unicode_minus'] = False


# ============ 一、旋转矩阵与欧拉角的互转 ============
def demo_representations():
    """演示旋转矩阵与欧拉角之间的转换。"""
    print('=' * 60)
    print('一、旋转矩阵与欧拉角的互转')
    print('=' * 60)

    # ---- 用欧拉角构造一个旋转：绕 Z 转 30°，绕 Y 转 45°，绕 X 转 60° ----
    # scipy 约定：大写 'ZYX' 表示外旋（绕固定轴依次转），单位可选 degrees=True
    euler_zyx = np.array([30.0, 45.0, 60.0])              # 单位：度
    r = R.from_euler('ZYX', euler_zyx, degrees=True)      # 构造 Rotation 对象

    # ---- 转换为旋转矩阵 (3, 3) ----
    matrix = r.as_matrix()                                # 3x3 正交矩阵
    print('旋转矩阵 (3x3):\n', np.round(matrix, 4))

    # ---- 验证旋转矩阵的性质 ----
    # 性质 1：正交矩阵，R^T R = I
    print('\nR^T @ R（应为单位阵）:\n', np.round(matrix.T @ matrix, 4))
    # 性质 2：行列式 = +1（区别于镜像，镜像是 -1）
    print('det(R) =', round(np.linalg.det(matrix), 4))

    # ---- 从矩阵转换回欧拉角 ----
    # 用不同顺序导出，得到的三元组不同，但都表示同一个旋转
    euler_zyx_back = r.as_euler('ZYX', degrees=True)
    euler_xyz_back = r.as_euler('XYZ', degrees=True)
    euler_zyx_intrinsic = r.as_euler('zyx', degrees=True)   # 小写 = 内旋
    print('\n恢复的欧拉角 ZYX (外旋, deg):', np.round(euler_zyx_back, 4))
    print('同一旋转表示为 XYZ (外旋, deg):', np.round(euler_xyz_back, 4))
    print('同一旋转表示为 zyx (内旋, deg):', np.round(euler_zyx_intrinsic, 4))

    # ---- 从旋转矩阵反构造 Rotation，验证矩阵与欧拉角是同一个旋转 ----
    r_from_matrix = R.from_matrix(matrix)
    v = np.array([1.0, 0.0, 0.0])
    print('\n用 [1, 0, 0] 测试两种表示的旋转结果:')
    print('  从欧拉角构造:', np.round(r.apply(v), 4))
    print('  从矩阵构造  :', np.round(r_from_matrix.apply(v), 4))


# ============ 二、不同欧拉角顺序的区别 ============
def demo_euler_orders():
    """演示同一个旋转用不同欧拉角顺序表达时，数值完全不同。"""
    print('\n' + '=' * 60)
    print('二、不同欧拉角顺序的区别')
    print('=' * 60)

    # 用同一组数值 [30, 45, 60]，分别按不同顺序解释
    angles = np.array([30.0, 45.0, 60.0])
    orders = ['ZYX', 'XYZ', 'ZYZ', 'XYZ'.lower()]        # 大写外旋，小写内旋

    v = np.array([1.0, 0.0, 0.0])
    print(f'用同一组角度 {angles} 按不同顺序解释，作用于向量 {v}:\n')
    for order in orders:
        r = R.from_euler(order, angles, degrees=True)
        result = r.apply(v)
        print(f'  顺序 {order!r:8s} → {np.round(result, 4)}')

    print('\n结论：欧拉角必须搭配"顺序"才有意义。')
    print('      同一组数值配不同顺序，代表完全不同的旋转。')


# ============ 三、外旋（固定轴）与内旋（刚体轴）的等价关系 ============
def demo_intrinsic_extrinsic():
    """演示外旋和内旋的等价关系。"""
    print('\n' + '=' * 60)
    print('三、外旋与内旋的等价关系')
    print('=' * 60)

    angles = np.array([30.0, 45.0, 60.0])
    v = np.array([1.0, 0.0, 0.0])

    # 外旋 ZYX：绕固定世界系的 Z→Y→X 依次转
    r_extrinsic = R.from_euler('ZYX', angles, degrees=True)
    # 内旋 xyz：绕刚体自身的 x→y→z 依次转（与 ZYX 等价）
    r_intrinsic = R.from_euler('xyz', angles, degrees=True)

    print('外旋 ZYX 与内旋 xyz 是等价表示：')
    print('  外旋 ZYX 作用结果:', np.round(r_extrinsic.apply(v), 4))
    print('  内旋 xyz 作用结果:', np.round(r_intrinsic.apply(v), 4))
    print(f'  两者是否一致: {np.allclose(r_extrinsic.as_matrix(), r_intrinsic.as_matrix())}')

    # ---- 与矩阵相乘的对应关系 ----
    # 外旋 ZYX（绕固定轴的 Z→Y→X）等价于 R_z @ R_y @ R_x
    r_x = R.from_euler('X', angles[2], degrees=True)
    r_y = R.from_euler('Y', angles[1], degrees=True)
    r_z = R.from_euler('Z', angles[0], degrees=True)
    r_composed = r_z * r_y * r_x                         # scipy 中 * 等价于矩阵 @
    print('\n外旋 ZYX 等价于 R_z @ R_y @ R_x（固定坐标系下从左往右依次应用）：')
    print('  复合结果:', np.round(r_composed.as_matrix(), 4))
    print(f'  与 r_extrinsic 一致: {np.allclose(r_composed.as_matrix(), r_extrinsic.as_matrix())}')


# ============ 四、万向节锁演示 ============
def demo_gimbal_lock():
    """演示万向节锁：欧拉角在某些姿态下会丢失一个自由度。"""
    print('\n' + '=' * 60)
    print('四、万向节锁 (Gimbal Lock)')
    print('=' * 60)

    print('ZYX 欧拉角：第一个绕 Z，第二个绕 Y，第三个绕 X')
    print('当中间角（Y）接近 90° 时，Z 和 X 轴的效果重合，丢失一个自由度。\n')

    # ---- 姿态 1：Y = 89°，Z = 30°，X = 0° ----
    e1 = np.array([30.0, 89.0, 0.0])
    # ---- 姿态 2：Y = 89°，Z = 0°，X = 30° ----
    e2 = np.array([0.0, 89.0, 30.0])

    r1 = R.from_euler('ZYX', e1, degrees=True)
    r2 = R.from_euler('ZYX', e2, degrees=True)

    v = np.array([1.0, 0.0, 0.0])
    print(f'姿态 1 (Z={e1[0]}, Y={e1[1]}, X={e1[2]}): → {np.round(r1.apply(v), 6)}')
    print(f'姿态 2 (Z={e2[0]}, Y={e2[1]}, X={e2[2]}): → {np.round(r2.apply(v), 6)}')
    print('两姿态结果几乎相同，说明 Z 和 X 的旋转此时等效，丢失一个自由度。')

    # ---- 极端情况：Y 恰好 90° ----
    e3 = np.array([45.0, 90.0, 0.0])
    e4 = np.array([0.0, 90.0, 45.0])
    r3 = R.from_euler('ZYX', e3, degrees=True)
    r4 = R.from_euler('ZYX', e4, degrees=True)
    print('\n极端情况 Y = 90°：')
    print(f'  (Z=45, Y=90, X=0):  → {np.round(r3.apply(v), 6)}')
    print(f'  (Z=0,  Y=90, X=45): → {np.round(r4.apply(v), 6)}')
    print('两者结果完全一致，Z 和 X 完全耦合，欧拉角在此处奇异。')


# ============ 五、可视化：欧拉角与旋转矩阵的效果 ============
def demo_plot():
    """可视化：绘制原始坐标系与经过旋转后的坐标系。"""
    print('\n' + '=' * 60)
    print('五、可视化：旋转后的坐标系')
    print('=' * 60)

    # ---- 构造三个旋转：分别绕 X、Y、Z 轴转 45° ----
    theta = np.deg2rad(45.0)
    rotations = {
        '绕 X 轴转 45°': R.from_euler('X', theta),
        '绕 Y 轴转 45°': R.from_euler('Y', theta),
        '绕 Z 轴转 45°': R.from_euler('Z', theta),
    }

    # ---- 用同一组欧拉角 [30, 45, 60] 但不同顺序，得到不同姿态 ----
    angles_deg = np.array([30.0, 45.0, 60.0])
    euler_demos = {
        '外旋 ZYX [30,45,60]': R.from_euler('ZYX', angles_deg, degrees=True),
        '外旋 XYZ [30,45,60]': R.from_euler('XYZ', angles_deg, degrees=True),
    }

    # ---- 绘图：2 行 × 3 列 ----
    fig = plt.figure(figsize=(15, 10))

    # 第一行：单轴旋转
    for i, (title, rot) in enumerate(rotations.items()):
        ax = fig.add_subplot(2, 3, i + 1, projection='3d')
        _draw_frame(ax, np.eye(3), 'k', '原始')
        _draw_frame(ax, rot.as_matrix(), 'tab:red', '旋转后')
        _set_3d_axes(ax, title)

    # 第二行：不同欧拉角顺序
    for i, (title, rot) in enumerate(euler_demos.items()):
        ax = fig.add_subplot(2, 3, i + 4, projection='3d')
        _draw_frame(ax, np.eye(3), 'k', '原始')
        _draw_frame(ax, rot.as_matrix(), 'tab:red', '旋转后')
        _set_3d_axes(ax, title)

    out_path = Path(__file__).parent / 'rotation_euler.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'可视化图已保存为 {out_path}')


def _draw_frame(ax, matrix, color, label):
    """在 3D 坐标轴上画出 matrix 表示的局部坐标系。"""
    origin = np.zeros(3)
    for i, (axis_color, name) in enumerate([('r', 'x'), ('g', 'y'), ('b', 'z')]):
        axis = matrix[:, i] * 1.0
        ax.quiver(*origin, *axis, color=axis_color, linewidth=2,
                  arrow_length_ratio=0.15)
    # 坐标系原点画一个小点
    ax.scatter(0, 0, 0, color=color, s=30)


def _set_3d_axes(ax, title):
    """统一设置 3D 坐标轴的范围、视角、标题。"""
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(-1.2, 1.2)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title, fontsize=10)
    ax.view_init(elev=20, azim=30)


# ============ 六、主流程 ============
def main():
    np.set_printoptions(precision=4, suppress=True)

    demo_representations()
    demo_euler_orders()
    demo_intrinsic_extrinsic()
    demo_gimbal_lock()
    demo_plot()

    plt.show()


if __name__ == '__main__':
    main()