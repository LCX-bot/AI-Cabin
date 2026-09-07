# 学习内容：Python 类与面向对象
# 实践任务：写 RobotJoint 类，包含 q,dq,kp,kd,limit

import sys

# 让中文输出在 UTF-8 终端中正常显示（Windows 默认用 GBK，否则会乱码）
sys.stdout.reconfigure(encoding='utf-8')

class RobotJoint:
    """机器人关节类：封装角度、角速度、PD 控制增益和限位。

    属性：
        q      关节角度 (deg)
        dq     关节角速度 (deg/s)
        kp     比例增益
        kd     微分增益
        limit  限位 (min, max)
    """

    # __init__ 就是“类”的构造函数
    def __init__(self, name='joint', q=0.0, dq=0.0, kp=1.0, kd=0.1,
                 limit=(-90.0, 90.0)):
        self.name = name
        self.q = float(q)
        self.dq = float(dq)
        self.kp = float(kp)
        self.kd = float(kd)
        self.limit = limit  # 走 property，赋值时会校验

    # ---- 限位用 property 封装：赋值时校验下界 < 上界 ----
    @property
    def limit(self):
        
        return self._limit

    @limit.setter
    def limit(self, value):
        lo, hi = value
        if lo >= hi:
            raise ValueError(f'限位下界必须小于上界，收到 {value}')
        self._limit = (float(lo), float(hi))

    # ---- 核心方法 ----
    def clamp(self):
        """把角度 q 限制在限位范围内，返回裁剪后的角度。"""
        self.q = max(self.limit[0], min(self.limit[1], self.q))
        return self.q

    def pd_torque(self, target_q):
        """PD 控制力矩：tau = kp*(target - q) - kd*dq。"""
        return self.kp * (target_q - self.q) - self.kd * self.dq

    def step(self, target_q, dt):
        """简化的一步仿真：用 PD 力矩更新角速度与角度，并应用限位。

        简化假设：忽略惯量，把力矩直接当角加速度处理（教学用）。
        """
        torque = self.pd_torque(target_q)
        self.dq += torque * dt
        self.q += self.dq * dt
        self.clamp()
        return self.q

    def __repr__(self):
        return (f'RobotJoint({self.name!r}, q={self.q:.2f}, dq={self.dq:.2f}, '
                f'kp={self.kp}, kd={self.kd}, limit={self.limit})')


# 创建一个肩关节：kp=2, kd=0.5, 限位 ±45°
joint = RobotJoint(name='shoulder', kp=2.0, kd=0.5, limit=(-45.0, 45.0))
print('初始状态：', joint)
target = 30.0
print(f'目标角度：{target}°')
for i in range(10):
    q = joint.step(target, dt=0.1)
    print(f'  t={i * 0.1:.1f}s  q={q:+.2f}°  dq={joint.dq:+.2f}°/s')
# 演示限位：把目标设到限位之外，q 应被裁剪在 ±45° 内
joint.q = 0.0
joint.dq = 0.0
print(f'\n把目标设到 {120}°（超出限位），仿真 20 步：')
for _ in range(20):
    q = joint.step(120.0, dt=0.1)
print(f'  最终 q = {q:.2f}°（被限制在 {joint.limit} 内）')