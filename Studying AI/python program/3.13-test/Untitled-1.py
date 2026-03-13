# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# 设置中文字体（解决中文显示问题）
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 创建数据
x = np.linspace(0, 2 * np.pi, 100)  # 从0到2π生成100个点
y_sin = np.sin(x)  # 正弦波
y_cos = np.cos(x)  # 余弦波

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制两条曲线
ax.plot(x, y_sin, 'b-', linewidth=2, label='正弦波 sin(x)')
ax.plot(x, y_cos, 'r--', linewidth=2, label='余弦波 cos(x)')

# 添加标题和标签
ax.set_title('正弦波与余弦波对比图', fontsize=16, fontweight='bold')
ax.set_xlabel('X轴 (弧度)', fontsize=12)
ax.set_ylabel('Y轴 振幅', fontsize=12)

# 添加图例和网格
ax.legend(loc='best', fontsize=12)
ax.grid(True, alpha=0.3)

# 设置坐标轴范围
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(-1.2, 1.2)

# 添加关键点标记
key_points = [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]
key_labels = ['0', 'π/2', 'π', '3π/2', '2π']
ax.set_xticks(key_points)
ax.set_xticklabels(key_labels)

# 显示图形
plt.tight_layout()
plt.show()

# 控制台输出额外信息
print("=" * 50)
print("✅ 可视化图表生成成功！")
print(f"  使用的NumPy版本: {np.__version__}")
print(f"  使用的Matplotlib版本: {matplotlib.__version__}")
print("=" * 50)
print("图表说明：")
print("1. 蓝色实线: 正弦函数 sin(x)")
print("2. 红色虚线: 余弦函数 cos(x)")
print("3. 图表已显示网格、图例和中文标签")
print("=" * 50)