# 学习内容：CSV/JSON/Pickle 文件读写
# 实践任务：生成 1000 步关节数据，保存并读取可视化

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# 让中文输出在 UTF-8 终端中正常显示（Windows 默认用 GBK，否则会乱码）
sys.stdout.reconfigure(encoding='utf-8')

# ---------- 1. 生成 1000 步关节数据 ----------
def simulate_joint(n_steps=1000, dt=0.01, kp=50.0, kd=10.0, J=1.0,
                   limit=(-45.0, 45.0)):
    """用 PD 控制 + 欧拉积分模拟一个关节追踪正弦目标。

    状态方程（含惯量 J）：
        tau = kp*(target - q) - kd*dq
        dq  += (tau / J) * dt
        q   += dq * dt
    返回 dict，键：time / target / q / dq / tau，值都是长度 n_steps 的数组。
    """
    t = np.arange(n_steps) * dt
    # 目标轨迹：0.5 Hz 正弦（幅值 20°），前 3 秒叠加 +10° 偏置
    target = 20.0 * np.sin(2.0 * np.pi * 0.5 * t) + np.where(t < 3.0, 10.0, 0.0)

    q = np.empty(n_steps)
    dq = np.empty(n_steps)
    tau = np.empty(n_steps)
    q[0] = dq[0] = tau[0] = 0.0

    for i in range(1, n_steps):
        tau[i] = kp * (target[i - 1] - q[i - 1]) - kd * dq[i - 1]
        dq[i] = dq[i - 1] + (tau[i] / J) * dt
        q[i] = np.clip(q[i - 1] + dq[i] * dt, *limit)

    return {'time': t, 'target': target, 'q': q, 'dq': dq, 'tau': tau}


# ---------- 2. 三种格式的保存 / 读取 ----------
def save_csv(path, data):
    """CSV：文本表格，首行表头，Excel 可直接打开。"""
    header = list(data.keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(np.column_stack([data[k] for k in header]))


def load_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = np.array([[float(v) for v in row] for row in reader])
    return header, rows


def save_json(path, data):
    """JSON：文本、跨语言通用，numpy 数组要先转成 list。"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({k: v.tolist() for k, v in data.items()}, f)


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(path, data):
    """Pickle：Python 专用二进制，原样保留 numpy 数组和类型。"""
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# ---------- 3. 可视化 ----------
def plot_joint(data, title, out_path):
    """画 3 个子图：角度 q、角速度 dq、力矩 tau。"""
    t, target, q, dq, tau = (data[k] for k in
                             ('time', 'target', 'q', 'dq', 'tau'))

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(t, target, '--', color='gray', linewidth=1.5, label='target')
    axes[0].plot(t, q, color='tab:blue', linewidth=1.5, label='q')
    axes[0].set_ylabel('q (deg)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].plot(t, dq, color='tab:red', linewidth=1.5)
    axes[1].set_ylabel('dq (deg/s)')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    axes[2].plot(t, tau, color='tab:green', linewidth=1.5)
    axes[2].set_ylabel('tau (N·m)')
    axes[2].set_xlabel('time (s)')
    axes[2].grid(True, linestyle='--', alpha=0.6)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'图片已保存为 {out_path}')


# ---------- 4. 主流程 ----------
def main():
    # (1) 生成数据
    data = simulate_joint()
    print(f'已生成 {len(data["q"])} 步关节数据')

    # (2) 保存成三种格式
    files = {
        'CSV':    Path('joint_data.csv'),
        'JSON':   Path('joint_data.json'),
        'Pickle': Path('joint_data.pkl'),
    }
    save_csv(files['CSV'], data)
    save_json(files['JSON'], data)
    save_pickle(files['Pickle'], data)

    # (3) 读取回来并核对（pickle 精确一致，CSV/JSON 是浮点文本往返，用近似比较）
    header, csv_rows = load_csv(files['CSV'])
    json_data = load_json(files['JSON'])
    pkl_data = load_pickle(files['Pickle'])

    q0 = data['q']
    assert np.allclose(csv_rows[:, header.index('q')], q0), 'CSV 读回不一致'
    assert np.allclose(json_data['q'], q0), 'JSON 读回不一致'
    assert np.array_equal(pkl_data['q'], q0), 'Pickle 读回不一致'
    print('三种格式读回校验通过')

    # (4) 预览 CSV 前 3 行 + 对比三种格式文件大小
    print('CSV 前 3 行：')
    for row in csv_rows[:3]:
        print('  ' + ', '.join(f'{v:.3f}' for v in row))
    for name, p in files.items():
        print(f'  {name:6s} 文件大小: {p.stat().st_size:>8,} 字节')

    # (5) 可视化（用 pickle 读回的数据画图，走通"读出来再画"的完整链路）
    plot_joint(pkl_data, 'Joint Data (read from pickle)', 'joint_data.png')
    plt.show()


if __name__ == '__main__':
    main()
