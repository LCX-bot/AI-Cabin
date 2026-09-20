# 学习内容：四元数
# 实践任务：四元数与旋转矩阵互转，写注释说明

import sys

import numpy as np

# 让中文输出在 UTF-8 终端中正常显示（Windows 默认用 GBK，否则会乱码）
sys.stdout.reconfigure(encoding='utf-8')


# ============ 四元数 → 旋转矩阵 ============
def quaternion_to_rotation_matrix(q):
    """单位四元数 q = [w, x, y, z] 转旋转矩阵。

    推导思路：四元数旋转公式 v' = q·v·q*，把 v 写成纯四元数 [0, vx, vy, vz]，
    展开四元数乘法后整理成矩阵形式。对单位四元数（w²+x²+y²+z²=1），得到：

        R = | 1-2(y²+z²)    2(xy-wz)     2(xz+wy)   |
            | 2(xy+wz)     1-2(x²+z²)    2(yz-wx)   |
            | 2(xz-wy)      2(yz+wx)    1-2(x²+y²)  |

    注意：q 必须是单位四元数，否则结果不是纯旋转（会带缩放）。
    """
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


# ============ 旋转矩阵 → 四元数 ============
def rotation_matrix_to_quaternion(R):
    """旋转矩阵转单位四元数（Shepperd 方法，数值稳定）。

    核心：由旋转矩阵的元素反解 w,x,y,z。用迹入手：
        trace = R00+R11+R22 = 4w² - 1  ⇒  w = sqrt(trace+1)/2

    若 trace > 0（旋转角 < 180°，w 较大），先求 w，再求 x,y,z；
    若 trace ≤ 0（旋转角接近 180°，w 很小，除以它会放大误差），
    则改成从「最大的对角线元素」对应的分量出发，避免除以接近 0 的数。
    """
    trace = np.trace(R)

    if trace > 0:
        # w 分量最大：trace = 4w²-1  ⇒  S = 4w
        S = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        # x 分量最大：1+R00-R11-R22 = 4x²  ⇒  S = 4x
        S = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        # y 分量最大：1+R11-R00-R22 = 4y²  ⇒  S = 4y
        S = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        # z 分量最大：1+R22-R00-R11 = 4z²  ⇒  S = 4z
        S = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)   # 归一化，消除浮点累计误差


# ============ 辅助：轴角 → 四元数 ============
def axis_angle_to_quaternion(axis, angle):
    """轴角 → 单位四元数（生成四元数最直观的方式）。

    绕单位轴 k 旋转 θ 角，对应 q = [cos(θ/2), sin(θ/2)·k]。
    这里的「半角」是四元数最反直觉的地方：转 θ 度，四元数里是 θ/2。
    """
    k = axis / np.linalg.norm(axis)
    half = angle / 2.0
    return np.array([np.cos(half),
                     k[0] * np.sin(half),
                     k[1] * np.sin(half),
                     k[2] * np.sin(half)])


# ============ 辅助：罗德里格斯公式（独立方法，用于交叉验证）============
def rodrigues_rotate(axis, angle, v):
    """绕轴 k 旋转向量 v（罗德里格斯公式），完全不经过四元数，用于验证。"""
    k = axis / np.linalg.norm(axis)
    return (v * np.cos(angle)
            + np.cross(k, v) * np.sin(angle)
            + k * np.dot(k, v) * (1 - np.cos(angle)))


def main():
    rng = np.random.default_rng(42)

    # ---- 1. 造一个旋转：随机轴 + 45° ----
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(45.0)

    q = axis_angle_to_quaternion(axis, angle)
    R = quaternion_to_rotation_matrix(q)
    q_back = rotation_matrix_to_quaternion(R)

    print('轴角：angle = 45°, axis =', np.round(axis, 4))
    print('四元数 q      =', np.round(q, 6))
    print('转回的 q_back =', np.round(q_back, 6))
    # 四元数 q 和 -q 表示同一个旋转，所以用 |q·q_back| 判断（应 = 1）
    print('|q · q_back| =', round(abs(float(np.dot(q, q_back))), 6), '（应为 1）')

    # ---- 2. 验证 R 是合法旋转矩阵：正交 + 行列式为 1 ----
    print('\nR =\n', np.round(R, 6))
    print('RᵀR（应为单位阵）=\n', np.round(R.T @ R, 12))
    print('det(R)（应为 1） =', round(float(np.linalg.det(R)), 6))

    # ---- 3. 交叉验证：四元数矩阵 vs 罗德里格斯公式，旋转同一向量 ----
    v = np.array([1.0, 0.0, 0.0])
    v_quat = R @ v
    v_rod = rodrigues_rotate(axis, angle, v)
    print('\n四元数旋转 v     =', np.round(v_quat, 6))
    print('罗德里格斯旋转 v =', np.round(v_rod, 6))
    print('两者差（应≈0）=', round(float(np.linalg.norm(v_quat - v_rod)), 12))

    # ---- 4. 演示 q 与 -q 等价 ----
    R_neg = quaternion_to_rotation_matrix(-q)
    print('\nq 与 -q 生成的旋转矩阵是否相同：', bool(np.allclose(R, R_neg)))


if __name__ == '__main__':
    main()
