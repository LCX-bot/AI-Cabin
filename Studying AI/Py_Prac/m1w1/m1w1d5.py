# 学习内容：综合练习
# 实践任务：写 plot_joint_log.py，画关节角/速度/力矩

# 和第 4 天的区别：第 4 天是"单关节 + 三种格式存读"，这次是
#   (1) 复用第 3 天的 RobotJoint 类，模拟一个 3 关节机械臂；
#   (2) 日志换成"长格式"（每行带 joint 名，同一时刻 3 行）；
#   (3) 画 q/dq/tau 时把多个关节叠加对比，并打印统计摘要。
# 用法：
#   python m1w1d5.py                     # 模拟并画图，输出 joint_log.png
#   python m1w1d5.py --steps 2000 -o out.png

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# 让中文输出在 UTF-8 终端中正常显示（Windows 默认用 GBK，否则会乱码）
sys.stdout.reconfigure(encoding='utf-8')


# ---------- 1. 复用第 3 天的 RobotJoint 类 ----------
class RobotJoint:
    """单关节：角度 q、角速度 dq、PD 增益 kp/kd、限位 limit。"""

    # 类初始化
    def __init__(self, name, q0=0.0, kp=50.0, kd=10.0, limit=(-45.0, 45.0)):
        self.name = name
        self.q = float(q0)
        self.dq = 0.0
        self.kp = float(kp)
        self.kd = float(kd)
        self.limit = limit

    # 算力矩的函数：由当前角度和目标角度的误差形成P环，得到要产生的力矩
    def torque(self, target):
        """PD 力矩：kp*(target-q) - kd*dq。"""
        return self.kp * (target - self.q) - self.kd * self.dq

    # 更新角速度、角度，得到一定的角度增量
    def step(self, target, dt):
        """推进一步（含限位），返回本步力矩。"""
        tau = self.torque(target)
        self.dq += tau * dt          # 简化：惯量 J = 1
        self.q = np.clip(self.q + self.dq * dt, *self.limit)
        return tau


# ---------- 2. 模拟 3 关节机械臂，生成"长格式"日志 ----------
def simulate_arm(n_steps=1000, dt=0.01):
    """返回 (rows, joints)。rows 是长格式：每行 (time, joint, target, q, dq, tau)。"""
    joints = [
        RobotJoint('shoulder', kp=40.0, kd=8.0,  limit=(-90.0, 90.0)),
        RobotJoint('elbow',    kp=60.0, kd=12.0, limit=(-90.0, 90.0)),
        RobotJoint('wrist',    kp=80.0, kd=16.0, limit=(-30.0, 30.0)),
    ]
    t = np.arange(n_steps) * dt
    # 三个关节追踪不同频率/相位/幅值的目标
    targets = {
        'shoulder': 30.0 * np.sin(2 * np.pi * 0.2 * t),          # 慢速大摆
        'elbow':    30.0 * np.sin(2 * np.pi * 0.5 * t + np.pi),  # 中速反相
        'wrist':    15.0 * np.sin(2 * np.pi * 1.0 * t),          # 快速小摆
    }

    rows = []
    for i in range(n_steps):
        for j in joints:
            target = targets[j.name][i]
            tau = j.step(target, dt)
            rows.append((t[i], j.name, target, j.q, j.dq, tau))
    return rows, joints


# ---------- 3. 保存 / 读取日志（长格式 CSV） ----------
HEADER = ('time', 'joint', 'target', 'q', 'dq', 'tau')


def save_log_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)


def load_log_csv(path):
    """读长格式 CSV，用 csv.DictReader 按关节名分组成 dict。"""
    grouped = defaultdict(lambda: {k: [] for k in ('time', 'target', 'q', 'dq', 'tau')})
    with open(path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            name = row['joint']
            for k in ('time', 'target', 'q', 'dq', 'tau'):
                grouped[name][k].append(float(row[k]))
    return {name: {k: np.array(v) for k, v in d.items()}
            for name, d in grouped.items()}


# ---------- 4. 画图 + 统计摘要 ----------
def plot_joint_log(data, out_path):
    """画 q/dq/tau 三张子图，每个关节一条线，方便对比。"""
    metrics = [('q', 'deg'), ('dq', 'deg/s'), ('tau', 'N·m')]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for ax, (m, unit) in zip(axes, metrics):
        for name, d in data.items():
            ax.plot(d['time'], d[m], linewidth=1.5, label=name)
        ax.set_ylabel(f'{m} ({unit})')
        ax.legend(fontsize=8, ncol=3)
        ax.grid(True, linestyle='--', alpha=0.6)

    axes[-1].set_xlabel('time (s)')
    fig.suptitle('Multi-Joint Log')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'图片已保存为 {out_path}')


def print_summary(data):
    """打印每个关节 |q|/|dq|/|tau| 的最大值，作简要分析。"""
    print('\n关节统计（绝对值最大值）：')
    print(f"  {'joint':<10}{'max|q|':>10}{'max|dq|':>12}{'max|tau|':>12}")
    for name, d in data.items():
        print(f"  {name:<10}"
              f"{np.abs(d['q']).max():>10.2f}"
              f"{np.abs(d['dq']).max():>12.2f}"
              f"{np.abs(d['tau']).max():>12.2f}")


# ---------- 5. 主流程 ----------
def main(argv=None):
    parser = argparse.ArgumentParser(description='多关节日志：模拟→存→读→画')
    parser.add_argument('-o', '--output', default='joint_log.png')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--log', default='joint_arm_log.csv',
                        help='日志 CSV 路径（存在则直接读，不存在则先模拟生成）')
    args = parser.parse_args(argv)

    if Path(args.log).exists():
        data = load_log_csv(args.log)
        print(f'从 {args.log} 读取 {len(data)} 个关节')
    else:
        rows, joints = simulate_arm(args.steps)
        save_log_csv(args.log, rows)
        data = load_log_csv(args.log)
        print(f'模拟 {args.steps} 步并保存到 {args.log}，共 {len(data)} 个关节：')
        for j in joints:
            print(f'  {j.name}: kp={j.kp}, kd={j.kd}, limit={j.limit}')

    print_summary(data)
    plot_joint_log(data, args.output)
    plt.show()


if __name__ == '__main__':
    main()
