# 学习内容：NumPy 数组、切片、广播、矩阵乘法
# 实践任务：写 numpy_basics.py，完成向量归一化、矩阵乘法、批量点变换

import numpy as np

# 数组标准化函数
def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

# 矩阵乘法函数
def matrix_multiply(A, B):
    A = np.asarray(A) # 不占用新内存
    B = np.asarray(B)
    if A.shape[1] != B.shape[0]:
        raise ValueError(...)
    return A @ B # 矩阵乘法用@

# 点变换函数
def batch_transform(points, M):
    points = np.asarray(points, dtype=float)
    M = np.asarray(M, dtype=float)
    if points.ndim != 2:
        raise ValueError("points must be (N, D)")
    if M.shape[0] != points.shape[1] or M.shape[1] != points.shape[1]:
        raise ValueError("M must be D x D")
    # (N, D) @ (D, D) -> (N, D)
    return points @ M.T

v1 = np.array([3.0, 4.0])

v2 = np.array([[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12]])

print(f"{v2}")

v_norm = normalize_vector(v2)
print(f"{v_norm}")

A = np.arange(12).reshape(3, 4)
B = np.arange(12).reshape(4, 3)

C = matrix_multiply(A, B)
print(f"{C}")

theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)], # 线性变换的矩阵
            [np.sin(theta), np.cos(theta)]])
points = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]]) # 线性变换的点
transformed = batch_transform(points, R)

# 切片取前两个点
print(points[:2])
print(transformed[:2])