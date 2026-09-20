# 学习内容：矩阵乘法与线性变换
# 实践任务：写二维旋转矩阵，旋转点云并画图

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# 让中文输出在 UTF-8 终端中正常显示（Windows 默认用 GBK，否则会乱码）
sys.stdout.reconfigure(encoding='utf-8')


def rotation_matrix(theta):
    """二维旋转矩阵：绕原点逆时针旋转 theta 弧度。"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])


def make_arrow():
    """生成「箭头」形状的点云 (2, N)：方向明确，旋转后一眼能看出。"""
    shaft_x = np.linspace(0.0, 3.0, 40)          # 箭杆：从原点到 (3, 0)
    shaft_y = np.zeros_like(shaft_x)
    h1_x = np.linspace(3.0, 2.5, 12)             # 箭头两撇
    h1_y = np.linspace(0.0, 0.5, 12)
    h2_x = np.linspace(3.0, 2.5, 12)
    h2_y = np.linspace(0.0, -0.5, 12)
    x = np.concatenate([shaft_x, h1_x, h2_x])
    y = np.concatenate([shaft_y, h1_y, h2_y])
    return np.vstack([x, y])                     # 形状 (2, N)


def main():
    points = make_arrow()

    # ---- 1. 数学验证：旋转矩阵的性质 ----
    R = rotation_matrix(np.deg2rad(30.0))
    print('30° 旋转矩阵 R =\n', R)
    print('RᵀR（应为单位阵）=\n', np.round(R.T @ R, 12))
    print('det(R)（应为 1） =', round(np.linalg.det(R), 6))

    tip = np.array([3.0, 0.0])                   # 箭头尖端
    print(f'长度保持：||x|| = {np.linalg.norm(tip):.6f}  '
          f'||R x|| = {np.linalg.norm(R @ tip):.6f}')

    # ---- 2. 每 30° 旋转一次，叠加画出"旋转族" ----
    angles_deg = np.arange(0, 360, 30)
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(angles_deg.min(), angles_deg.max())

    fig, ax = plt.subplots(figsize=(7, 7))
    for th in angles_deg:
        rotated = rotation_matrix(np.deg2rad(th)) @ points   # (2,2) @ (2,N)
        ax.plot(rotated[0], rotated[1], color=cmap(norm(th)), linewidth=1.2)

    # 原始箭头黑色加粗；虚线圆 = 尖端(3,0)的旋转轨迹（到原点距离不变）
    ax.plot(points[0], points[1], color='black', linewidth=2.5, label='original (0°)')
    circle = np.linspace(0, 2 * np.pi, 200)
    ax.plot(3 * np.cos(circle), 3 * np.sin(circle), 'k--', linewidth=0.8, alpha=0.4)

    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('2D Rotation Matrix Applied to a Point Cloud (every 30°)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(angles_deg)
    fig.colorbar(sm, ax=ax, label='angle (°)')

    fig.tight_layout()
    out_path = Path(__file__).parent / 'rotation_pointcloud.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'图片已保存为 {out_path}')
    plt.show()


if __name__ == '__main__':
    main()
