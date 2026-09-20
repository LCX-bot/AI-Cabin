# 学习内容：特征值、SVD、PCA 直觉
# 实践任务：对二维点云做 PCA 并画主方向

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# 让中文输出在 UTF-8 终端中正常显示（Windows 默认用 GBK，否则会乱码）
sys.stdout.reconfigure(encoding='utf-8')


def rotation_matrix(theta):
    """二维旋转矩阵（和第 2 天一样），用来造一个"倾斜"的点云。"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],
                     [s,  c]])


def main():
    rng = np.random.default_rng(42)
    n = 500

    # ---- 1. 造一个"有主方向"的倾斜点云 ----
    # 各向同性高斯 → 沿 x 拉长、y 压扁 → 旋转 30°，让主方向不沿坐标轴
    X = rng.normal(0.0, 1.0, (n, 2))
    X[:, 0] *= 3.0
    X[:, 1] *= 0.5
    X = X @ rotation_matrix(np.deg2rad(30.0)).T   # 每个点绕原点旋转 30°

    # ---- 2. PCA 三步：中心化 → 协方差 → 特征分解 ----
    Xc = X - X.mean(axis=0)
    cov = (Xc.T @ Xc) / (n - 1)

    eigvals, eigvecs = np.linalg.eigh(cov)        # 升序
    order = np.argsort(eigvals)[::-1]             # 改成降序
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # ---- 3. SVD 能得到同样的结果：协方差特征值 = 奇异值²/(n-1) ----
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var_from_svd = S ** 2 / (n - 1)

    # ---- 4. 打印验证 ----
    print('协方差矩阵 =\n', np.round(cov, 3))
    print('特征值（= 各主方向的方差）:', np.round(eigvals, 3))
    print('SVD: σ²/(n-1)            :', np.round(var_from_svd, 3))
    print('特征向量（主方向，按列）=\n', np.round(eigvecs, 4))

    pc1 = eigvecs[:, 0]
    angle = np.degrees(np.arctan2(pc1[1], pc1[0]))
    print(f'恢复出的主方向与 x 轴夹角 ≈ {angle:.1f}°（真实 30°，符号可能相反）')

    # ---- 5. 画图：散点 + 均值 + 主方向箭头 + 2σ 椭圆 ----
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(X[:, 0], X[:, 1], s=10, alpha=0.35, color='tab:blue', label='points')

    mean = X.mean(axis=0)
    ax.scatter(*mean, color='black', marker='x', s=80, zorder=5, label='mean')

    colors = ['tab:red', 'tab:green']
    for i in range(2):
        length = 2.0 * np.sqrt(eigvals[i])        # 2 个标准差
        end = mean + length * eigvecs[:, i]
        ax.annotate('', xy=tuple(end), xytext=tuple(mean),
                    arrowprops=dict(arrowstyle='-|>', color=colors[i], lw=2.5))
        ax.text(end[0], end[1], f'  PC{i + 1}', color=colors[i],
                fontsize=11, va='center')

    # 2σ 置信椭圆：在主轴坐标系里画圆，再映射回原坐标系
    t = np.linspace(0, 2 * np.pi, 200)
    circle = np.vstack([np.cos(t), np.sin(t)])                    # (2, 200)
    ellipse = mean[:, None] + eigvecs @ (2 * np.sqrt(eigvals)[:, None] * circle)
    ax.plot(ellipse[0], ellipse[1], 'k--', linewidth=1, alpha=0.5, label='2σ ellipse')

    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('PCA: Principal Directions of a 2D Point Cloud')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    out_path = Path(__file__).parent / 'pca_directions.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'图片已保存为 {out_path}')
    plt.show()


if __name__ == '__main__':
    main()
