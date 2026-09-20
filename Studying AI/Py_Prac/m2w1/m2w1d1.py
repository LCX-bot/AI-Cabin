# 学习内容：Agent、Env、Reward、Return
# 实践任务：运行 CartPole 随机策略


import gymnasium as gym

# ---------- Env：创建环境 ----------
env = gym.make("CartPole-v1", render_mode="human")

# ---------- 开始一个回合 ----------
observation, info = env.reset(seed=42)
print(f"初始观测 (4维): {observation}")
print(f"info: {info}")

total_reward = 0.0   # 累积 Return
step_count = 0

for step in range(500):
    # ---------- Agent：随机选动作 ----------
    action = env.action_space.sample()

    # ---------- Env：执行一步，返回 5 个值 ----------
    observation, reward, terminated, truncated, info = env.step(action)

    # ---------- Reward：本步即时奖励 ----------
    total_reward += reward
    step_count += 1

    # 每一帧都渲染（human 模式下自动）
    if terminated or truncated:
        reason = "杆子倒了/车出界" if terminated else "到达 500 步上限"
        print(f"回合结束于第 {step_count} 步，原因：{reason}")
        print(f"Return = {total_reward}")
        break

env.close()