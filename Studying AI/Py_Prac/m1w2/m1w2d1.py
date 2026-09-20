# 学习内容：向量、点积、叉积、范数
# 实践任务：实现向量夹角、点到直线距离

import numpy as np
import sys

sys.stdout.reconfigure(encoding='utf-8')

def angle_between(v1, v2):
    """返回两个向量之间的夹角（弧度）"""
    dot = np.dot(v1, v2)    # 计算点积
    norm_v1 = np.linalg.norm(v1)    # 计算向量范数
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        raise ValueError("零向量无法计算夹角")
    cos_theta = np.clip(dot / (norm_v1 * norm_v2), -1.0, 1.0)
    return np.arccos(cos_theta)

# 测试
a = np.array([1, 0, 0])
b = np.array([0, 1, 0])
print(f"夹角: {angle_between(a, b):.4f} rad = {np.degrees(angle_between(a,b)):.2f}°")
# 输出: 90°