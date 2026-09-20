# 学习内容：小分类任务
# 实践任务：用 sklearn make_moons + MLP 分类

import sys                                              # 用于重设 stdout 编码，解决中文乱码
from pathlib import Path                                # 面向对象的路径处理，跨平台更安全

import numpy as np                                      # 数值计算库
import matplotlib.pyplot as plt                         # 绘图库
from sklearn.datasets import make_moons                 # sklearn 提供的"双月牙"玩具数据集生成器
from sklearn.model_selection import train_test_split    # 用于切分训练/验证集

import torch                                            # PyTorch 主库，提供张量和自动求导
import torch.nn as nn                                   # 神经网络模块（层、损失函数等）
from torch.utils.data import TensorDataset, DataLoader  # 数据封装与批次加载工具

# 让中文输出在 UTF-8 终端中正常显示（Windows 默认用 GBK，否则会乱码）
sys.stdout.reconfigure(encoding='utf-8')


# ============ 1. 模型定义 ============
class MLP(nn.Module):
    """两层 MLP 分类器：输入 2 维 → 隐藏层 → 隐藏层 → 输出 2 类 logits。"""

    def __init__(self, hidden=32):
        # 参数 hidden：隐藏层神经元数量，控制模型容量
        super().__init__()                              # 必须调用父类构造，初始化 nn.Module 内部状态
        self.net = nn.Sequential(                       # 用 Sequential 把多层按顺序串起来，前向时逐层传递
            nn.Linear(2, hidden),                       # 第一层：输入 2 维（x1, x2）→ 隐藏维
            nn.ReLU(),                                  # 非线性激活函数，让网络有能力拟合非线性边界
            nn.Linear(hidden, hidden),                  # 第二层：隐藏 → 隐藏，增加表达能力
            nn.ReLU(),                                  # 第二层后再接一个非线性
            nn.Linear(hidden, 2),                       # 输出层：隐藏 → 2 类 logits（未归一化打分）
        )

    def forward(self, x):
        # 前向传播：输入 x 形状 (B, 2)，输出形状 (B, 2)
        return self.net(x)                              # 直接调用 Sequential 即完成全部层的前向


# ============ 2. 数据准备 ============
def make_moons_data(n=1000, noise=0.2, seed=42, val_ratio=0.2):
    """生成 make_moons 数据并做训练/验证切分。

    返回：
        X_train: (n_train, 2) float32
        y_train: (n_train,)   int64
        X_val:   (n_val, 2)   float32
        y_val:   (n_val,)     int64
    """
    # 调用 sklearn 生成 2 维、2 类的"双月牙"数据
    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    # X 默认是 float64，转成 float32 与 PyTorch 保持一致，避免 dtype 不匹配
    X = X.astype(np.float32)
    # y 转成 int64，CrossEntropyLoss 要求标签是整型
    y = y.astype(np.int64)
    # 切分训练/验证集，stratify=y 保证两边类别比例一致
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_ratio, random_state=seed, stratify=y
    )
    # 一次性返回四个数组
    return X_train, y_train, X_val, y_val


# ============ 3. 单个 epoch 训练与评估 ============
def train_one_epoch(model, loader, criterion, optimizer, device):
    """跑一个 epoch，返回 (按样本加权的平均损失, 准确率)。"""
    model.train()                                       # 切换到训练模式（影响 Dropout/BN 等层）
    total_loss, n_correct, n_samples = 0.0, 0, 0        # 累计损失、正确数、样本数
    for xb, yb in loader:                               # 逐批次迭代 DataLoader
        # 把数据搬到与模型相同的设备（GPU 训练时尤其重要）
        xb = xb.to(device, non_blocking=True)           # non_blocking 配合 pin_memory 实现异步拷贝
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad()                           # 清空上一步残留的梯度（PyTorch 梯度会累加）
        logits = model(xb)                              # 前向：得到 logits，形状 (B, 2)
        loss = criterion(logits, yb)                    # 计算交叉熵损失（内部含 log_softmax）
        loss.backward()                                 # 反向传播，计算各参数的梯度
        optimizer.step()                                # 用优化器更新参数

        total_loss += loss.item() * xb.size(0)          # loss 是 batch 内均值，乘样本数还原为总量
        pred = logits.argmax(dim=1)                     # 取每行最大 logit 的下标作为预测类别
        n_correct += (pred == yb).sum().item()          # 累加当前 batch 中预测正确的样本数
        n_samples += xb.size(0)                         # 累加当前 batch 的样本数
    # 计算 epoch 级别的平均损失和准确率
    return total_loss / n_samples, n_correct / n_samples


@torch.no_grad()                                        # 评估时禁用梯度计算，节省显存并加速
def evaluate(model, loader, criterion, device):
    """验证集评估，返回 (平均损失, 准确率)。"""
    model.eval()                                        # 切换到评估模式（Dropout 关闭、BN 用滑动统计量）
    total_loss, n_correct, n_samples = 0.0, 0, 0        # 同训练，累计统计量
    for xb, yb in loader:                               # 逐批次迭代
        xb = xb.to(device, non_blocking=True)           # 搬到模型所在设备
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)                              # 前向（不构建计算图）
        loss = criterion(logits, yb)                    # 计算损失（用于报告，不更新参数）
        total_loss += loss.item() * xb.size(0)          # 加权累计
        pred = logits.argmax(dim=1)                     # 预测类别
        n_correct += (pred == yb).sum().item()          # 累计正确数
        n_samples += xb.size(0)                         # 累计样本数
    # 返回平均损失与准确率
    return total_loss / n_samples, n_correct / n_samples


# ============ 4. 决策边界可视化工具 ============
def plot_decision_boundary(ax, model, device, xlim=(-2.5, 3.0), ylim=(-2.0, 2.0), res=300):
    """在给定坐标平面上画出模型的分类区域。"""
    # 生成 x、y 方向的等间距坐标序列
    xx = np.linspace(xlim[0], xlim[1], res)             # x 方向 res 个点
    yy = np.linspace(ylim[0], ylim[1], res)             # y 方向 res 个点
    # 生成网格：XX、YY 形状均为 (res, res)，每个位置对应平面上一个点的坐标
    XX, YY = np.meshgrid(xx, yy)
    # 把二维网格展平成 (res*res, 2) 的坐标点数组，符合模型输入要求
    grid = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)
    with torch.no_grad():                               # 推理不需要梯度
        # 把网格点送到设备上并前向，得到 logits
        logits = model(torch.from_numpy(grid).to(device))
        # argmax 得到每点的预测类别，再 reshape 回 (res, res) 便于绘制
        preds = logits.argmax(dim=1).cpu().numpy().reshape(XX.shape)
    # 用填充等值线按预测类别给区域着色；levels 设在 -0.5/0.5/1.5 让两类的分界线清晰
    ax.contourf(XX, YY, preds, alpha=0.25, cmap='coolarm' if False else 'coolwarm',
                levels=[-0.5, 0.5, 1.5])


# ============ 5. 主流程 ============
def main():
    torch.manual_seed(42)                               # 固定 torch 随机种子，保证结果可复现
    np.random.seed(42)                                  # 固定 numpy 随机种子

    # ---- 5.1 设备 ----
    # 优先用 GPU，若无则退回 CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'PyTorch 版本: {torch.__version__}')          # 打印版本，确认是 cpu 还是 cuXXX 版
    print(f'使用设备: {device}')                         # 打印当前使用的设备
    if device.type == 'cuda':                           # 若使用 GPU，额外打印型号
        print(f'GPU 型号: {torch.cuda.get_device_name(0)}')

    # ---- 5.2 数据 ----
    # 生成并切分数据
    X_train, y_train, X_val, y_val = make_moons_data(
        n=1000, noise=0.2, seed=42, val_ratio=0.2
    )
    # 打印数据集规模，便于确认切分是否正确
    print(f'训练集: {X_train.shape[0]} 个样本，验证集: {X_val.shape[0]} 个样本')
    # bincount 统计每类出现次数，检查类别是否平衡
    print(f'训练集类别分布: {np.bincount(y_train)}')

    # 把 numpy 数组转成 torch 张量并封装为 Dataset
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    batch_size = 32                                     # 每个批次样本数
    # 训练集 DataLoader：打乱、pin_memory 与设备联动
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        pin_memory=(device.type == 'cuda'), num_workers=0
    )
    # 验证集 DataLoader：不打乱，其它同训练
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        pin_memory=(device.type == 'cuda'), num_workers=0
    )

    # ---- 5.3 模型、损失函数、优化器 ----
    model = MLP(hidden=32).to(device)                   # 实例化 MLP 并搬到设备上
    criterion = nn.CrossEntropyLoss()                   # 多分类标准损失（内部含 softmax）
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)  # Adam 优化器，学习率 1e-2

    # ---- 5.4 训练循环 ----
    epochs = 200                                        # 总训练轮数
    train_losses, train_accs = [], []                   # 记录每轮的训练损失/准确率
    val_losses, val_accs = [], []                       # 记录每轮的验证损失/准确率
    best_val_loss, best_epoch = float('inf'), -1        # 记录验证损失最优的时刻
    ckpt_path = Path(__file__).parent / 'best_model_moons.pt'  # 最优模型保存路径

    for epoch in range(1, epochs + 1):                  # 逐轮训练
        # 跑一遍训练集
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # 跑一遍验证集
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # 记录本轮指标，供后面绘图
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # 若本轮验证损失更低，保存当前模型参数
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            torch.save(model.state_dict(), ckpt_path)   # 只保存参数字典，不保存整个模型对象

        # 每 20 轮打印一次日志
        if epoch == 1 or epoch % 20 == 0:
            print(f'Epoch [{epoch:3d}/{epochs}]  '
                  f'train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  '
                  f'val_loss={val_loss:.4f}  val_acc={val_acc:.3f}')

    # 训练结束后打印最优结果摘要
    print(f'\n训练完成，最佳 epoch = {best_epoch}，'
          f'最佳 val_loss = {best_val_loss:.4f}，'
          f'对应 val_acc = {val_accs[best_epoch - 1]:.4f}')

    # ---- 5.5 加载最佳模型 ----
    # 从磁盘加载最优参数；weights_only=True 只反序列化张量，更安全
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()                                        # 切换到评估模式

    # ---- 5.6 绘图 ----
    # 创建 1 行 3 列的子图，宽 16 英寸、高 4.8 英寸
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # (a) 损失曲线
    ax = axes[0]                                        # 取第一个子图
    ax.plot(train_losses, label='train loss', color='tab:blue')     # 训练损失曲线
    ax.plot(val_losses, label='val loss', color='tab:orange')       # 验证损失曲线
    ax.axvline(best_epoch - 1, color='gray', linestyle='--', alpha=0.7,
               label=f'best epoch = {best_epoch}')                  # 标记最优 epoch（0 索引对齐）
    ax.set_xlabel('epoch')                              # x 轴标签
    ax.set_ylabel('Cross-Entropy Loss')                 # y 轴标签
    ax.set_title(f'Loss Curve (on {device.type.upper()})')          # 标题含设备信息
    ax.legend()                                         # 显示图例
    ax.grid(True, linestyle='--', alpha=0.5)            # 网格线

    # (b) 准确率曲线
    ax = axes[1]                                        # 第二个子图
    ax.plot(train_accs, label='train acc', color='tab:blue')        # 训练准确率
    ax.plot(val_accs, label='val acc', color='tab:orange')          # 验证准确率
    ax.axvline(best_epoch - 1, color='gray', linestyle='--', alpha=0.7,
               label=f'best epoch = {best_epoch}')                  # 标记最优 epoch
    ax.set_xlabel('epoch')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.5, 1.02)                              # 准确率大多在 0.5~1.0，固定范围便于对比
    ax.set_title('Accuracy Curve')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)

    # (c) 决策边界 + 数据点
    ax = axes[2]                                        # 第三个子图
    plot_decision_boundary(ax, model, device)           # 先画分类区域背景
    # 训练样本点用圆点，按类别着色
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm',
               s=12, alpha=0.6, edgecolors='k', linewidths=0.3, label='train')
    # 验证样本点用三角，稍大透明度以突出显示
    ax.scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap='coolwarm',
               s=12, alpha=0.9, marker='^', edgecolors='k', linewidths=0.3, label='val')
    ax.set_xlabel('x1')                                 # x 轴
    ax.set_ylabel('x2')                                 # y 轴
    ax.set_title('Decision Boundary')                   # 标题
    ax.legend(loc='upper right', fontsize=8)            # 图例放在右上角，小字号
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()                                  # 自动调整子图间距，防止标签被裁
    out_path = Path(__file__).parent / 'moons_classification.png'   # 输出路径与脚本同目录
    fig.savefig(out_path, dpi=150, bbox_inches='tight') # 保存为 PNG，150 dpi，紧凑裁剪
    print(f'图片已保存为 {out_path}')
    plt.show()                                          # 弹出窗口显示图形


# 只有直接运行该文件时才执行 main()，被 import 时不执行
if __name__ == '__main__':
    main()