"""
启发式脚本专家策略 (上帝视角抓取), 面向 VLA 微调深度优化:

  1. 虚拟牵引绳机制 (Virtual Leash Control): 闭环被控量严格采用环境积分设定点
     info["eef_target"], 但每步动态限制设定点领先真实末端 eef_pose 的距离 (MAX_LEASH).
     彻底杜绝纯 eef_pose 闭环导致的积分饱和与远伸横扫, 同时解决纯 setpoint 闭环
     瞬间到位导致的“动作提前归零死等”问题.
  2. 连续协同轨迹 (多自由度融合): 废除原 HOVER 与 YAW 的硬解耦分段, 改为 APPROACH
     阶段同时规划 XY 平移、Z 高度与 Yaw 偏航, 机械臂各维度保持协同平滑速度, 消除单通道置 0.
  3. 抓取阶段消除全 0: 合爪阶段持续维持末端稳定微控, 夹爪通道明确输出闭合意图 (-1.0),
     彻底抹除原脚本前 4 步出现的整行 [0, 0, 0, 0, 0] 样本.
  4. 移除死区断崖: 采用平滑连续速度衰减替代 if dist < 0.005: action=0 的阶跃截断.
  5. 剔除尾部无效停顿: 机械臂完成放置并撤退至安全高度后即刻停止数据记录,
     环境底层继续执行 40 步物理检验以确认成功, 消除占总轨迹 20% 以上的末尾全 0 静止帧.
"""

import numpy as np

from smolvla_tuning.envs import PickPlaceEnv
from smolvla_tuning.envs.pick_place_env import POS_SCALE, YAW_SCALE

# ================= 控制常量配置 =================
GRIPPER_OPEN = 1.0  # action[4] = +1 -> 张开
GRIPPER_CLOSE = -1.0  # action[4] = -1 -> 闭合

HOVER_HEIGHT = 0.06  # 接近/悬停高度 (高于方块)
GRASP_Z_OFFSET = 0.002  # 抓取对准高度
GRASP_GAP_BIAS = 0.025  # 沿间隙轴向固定指侧偏置
LIFT_HEIGHT = 0.12  # 托举高度
PLACE_HEIGHT = 0.055  # 放置释放高度
RETREAT_HEIGHT = 0.14  # 撤退安全高度

APPROACH_XY_TOL = 0.015  # 接近阶段水平容差
APPROACH_YAW_TOL = np.radians(15)  # 接近阶段偏航角容差
MOVE_TOL = 0.040  # 托盘对准容差
DROP_DIST = 0.06  # 途中掉落判定距离

DESCEND_STALL_STEPS = 6  # 下降触底判定步数
GRASP_SETTLE_STEPS = 20  # 稳定合爪步数
LIFT_CONFIRM_STEPS = 25  # 托举成功确认窗口

MAX_LEASH = 0.018  # 虚拟牵引绳长度 (m): 限制 ik_target 领先 eef_pose 的最大距离
MAX_STEPS = 260  # 回合上限步数

# 阶段超时 (依次为: APPROACH, DESCEND, GRASP, LIFT, MOVE, PLACE, RELEASE)
PHASE_TIMEOUTS = (90, 50, GRASP_SETTLE_STEPS + 5, LIFT_CONFIRM_STEPS, 90, 60, 60)


def _wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


class FSMExpert:
    """基于虚拟牵引绳与平滑多轴协同的示教专家策略"""

    # 7 阶段状态机 (合并了原 HOVER 与 YAW)
    APPROACH, DESCEND, GRASP, LIFT, MOVE, PLACE, RELEASE = range(7)

    def __init__(self, env: PickPlaceEnv):
        self.env = env
        self.tray_xy = np.asarray(env.tray_xy, dtype=np.float32)
        self.reset()

    def reset(self):
        cube = np.asarray(self.env.data.xpos[self.env._cube_id], dtype=np.float32)
        quat = self.env.data.qpos[self.env._cube_qadr + 3 : self.env._cube_qadr + 7]
        w, x, y, z = quat
        cube_yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

        cur_yaw = float(self.env._eef_pose()[3])
        # 4 重对称选取最近偏航角
        self.yaw_target = min(
            (cube_yaw + k * np.pi / 2 for k in range(4)),
            key=lambda c: abs(_wrap_angle(c - cur_yaw)),
        )
        self.gap_dir = np.array(
            [np.cos(self.yaw_target), np.sin(self.yaw_target)], dtype=np.float32
        )
        self.grasp_xy = cube[:2] + GRASP_GAP_BIAS * self.gap_dir
        self.cube_z0 = float(cube[2])

        self.state = self.APPROACH
        self.timer = 0
        self.prev_z = 1.0
        self.stall = 0
        self.failed = False
        self.fail_reason = ""
        self.recording_done = False  # 标志: 示教动作已完成, 停止记录冗余等待数据

    def _abort(self, reason: str):
        self.failed = True
        self.fail_reason = reason

    def _goto(self, wp: np.ndarray, info: dict, z_rate: float = 1.0) -> np.ndarray:
        """平滑连续牵引绳控制 (Leash Control):

        - 严格以 info["eef_target"] 为设定点闭环, 保持系统阻尼特性;
        - 引入 MAX_LEASH 限幅, 设定点不脱离物理末端, 根除积分超调和远伸横扫;
        - 移除硬死区断崖, 保持持续微控, 杜绝全 0 动作.
        """
        real_pose = info["eef_pose"]
        ik_target = info["eef_target"]

        # 1. 全局目标方向与距离 (基于设定点计算)
        err_pos = wp[:3] - ik_target[:3]
        dist_xy = np.linalg.norm(err_pos[:2])

        # 2. 牵引绳滞后校验 (设定点相对真实物理末端的超前量)
        lag_xy = np.linalg.norm(ik_target[:2] - real_pose[:2])

        action = np.zeros(5, dtype=np.float32)

        # 3. 水平 XY 平滑连续速度规划
        if dist_xy > 0.001:
            dir_xy = err_pos[:2] / dist_xy
            # 随距离平滑衰减, 远端以 1.0 推进, 最小保持 0.08 的导向输出避免断流
            speed = np.clip(dist_xy / (POS_SCALE * 1.5), 0.08, 1.0)

            # 若物理末端跟不上设定点, 动态缩减步长等待物理跟随, 防止积压发散
            if lag_xy > MAX_LEASH:
                speed *= max(0.1, MAX_LEASH / (lag_xy + 1e-6))

            action[:2] = dir_xy * speed
        else:
            action[:2] = np.clip(err_pos[:2] / POS_SCALE, -0.05, 0.05)

        # 4. 垂直 Z 轴速度规划 (带滞后保护)
        lag_z = abs(ik_target[2] - real_pose[2])
        err_z = wp[2] - ik_target[2]
        z_speed = np.clip(err_z / POS_SCALE, -z_rate, z_rate)
        if lag_z > MAX_LEASH and (err_z * (ik_target[2] - real_pose[2]) > 0):
            z_speed *= max(0.1, MAX_LEASH / (lag_z + 1e-6))
        action[2] = z_speed

        # 5. Yaw 轴平滑闭环
        err_yaw = _wrap_angle(float(wp[3] - ik_target[3]))
        action[3] = np.clip(err_yaw / YAW_SCALE, -0.05, 0.05)

        return action

    def _advance(self, info: dict):
        cur = info["eef_pose"]
        cube = info["cube_pos"]

        if self.timer > PHASE_TIMEOUTS[self.state]:
            self._abort(f"phase_timeout@{self.state}")
            return

        if self.state == self.APPROACH:
            # 协同收敛: XY 到位且 Yaw 偏航角对齐后进入下探
            ok_xy = np.linalg.norm(cur[:2] - self.grasp_xy) < APPROACH_XY_TOL
            ok_yaw = (
                abs(_wrap_angle(float(cur[3] - self.yaw_target))) < APPROACH_YAW_TOL
            )
            ok_z = cur[2] > self.cube_z0 + 0.035
            if ok_xy and ok_yaw and ok_z:
                self.state, self.timer = self.DESCEND, 0

        elif self.state == self.DESCEND:
            if cur[2] < self.prev_z - 0.0002:
                self.stall = 0
            else:
                self.stall += 1
            self.prev_z = cur[2]

            contact = (
                self.stall >= DESCEND_STALL_STEPS and cur[2] > cube[2] + 0.010
            ) or (cube[2] < self.cube_z0 - 0.002)
            at_depth = abs(cur[2] - cube[2]) < 0.008

            if at_depth or contact:
                self.state, self.timer = self.GRASP, 0

        elif self.state == self.GRASP:
            if self.timer >= GRASP_SETTLE_STEPS:
                self.state, self.timer = self.LIFT, 0

        elif self.state == self.LIFT:
            if cube[2] > self.cube_z0 + 0.03:
                self.state, self.timer = self.MOVE, 0
            elif self.timer >= LIFT_CONFIRM_STEPS:
                self._abort("cube_not_lifted")

        elif self.state == self.MOVE:
            if np.linalg.norm(cube[:2] - self.tray_xy) < MOVE_TOL:
                self.state, self.timer = self.PLACE, 0
            elif np.linalg.norm(cube[:2] - cur[:2]) > DROP_DIST:
                self._abort("cube_dropped_in_transit")

        elif self.state == self.PLACE:
            if cur[2] < PLACE_HEIGHT + 0.004:
                self.state, self.timer = self.RELEASE, 0

        elif self.state == self.RELEASE:
            # 撤退到安全高度后, 示教动作已经圆满完成
            if cur[2] >= RETREAT_HEIGHT - 0.008:
                self.recording_done = True

    def step(self, info: dict) -> np.ndarray:
        self.timer += 1
        self._advance(info)
        cur = info["eef_pose"]
        cube = info["cube_pos"]

        if self.state == self.APPROACH:
            # XY/Z/Yaw 全通道同时运动逼近预抓取点
            wp = np.array(
                [
                    self.grasp_xy[0],
                    self.grasp_xy[1],
                    self.cube_z0 + HOVER_HEIGHT,
                    self.yaw_target,
                ],
                dtype=np.float32,
            )
            action = self._goto(wp, info, z_rate=1.0)
            action[4] = GRIPPER_OPEN

        elif self.state == self.DESCEND:
            wp = np.array(
                [
                    self.grasp_xy[0],
                    self.grasp_xy[1],
                    self.cube_z0 + GRASP_Z_OFFSET,
                    self.yaw_target,
                ],
                dtype=np.float32,
            )
            action = self._goto(wp, info, z_rate=0.4)
            action[4] = GRIPPER_OPEN

        elif self.state == self.GRASP:
            # 抓取保持: 维持末端微弱下压闭环, 夹爪通道输出闭合指令, 杜绝全 0 动作
            wp = np.array(
                [
                    self.grasp_xy[0],
                    self.grasp_xy[1],
                    self.cube_z0 + GRASP_Z_OFFSET,
                    self.yaw_target,
                ],
                dtype=np.float32,
            )
            action = self._goto(wp, info, z_rate=0.1)
            action[4] = GRIPPER_CLOSE

        elif self.state == self.LIFT:
            # 保持当前抓取点水平位置提升
            wp = np.array(
                [self.grasp_xy[0], self.grasp_xy[1], LIFT_HEIGHT, self.yaw_target],
                dtype=np.float32,
            )
            action = self._goto(wp, info, z_rate=1.0)
            action[4] = GRIPPER_CLOSE

        elif self.state in (self.MOVE, self.PLACE):
            off = np.clip(cur[:2] - cube[:2], -0.030, 0.030)
            z = LIFT_HEIGHT if self.state == self.MOVE else PLACE_HEIGHT
            wp = np.array(
                [
                    self.tray_xy[0] + off[0],
                    self.tray_xy[1] + off[1],
                    z,
                    self.yaw_target,
                ],
                dtype=np.float32,
            )
            action = self._goto(
                wp, info, z_rate=1.0 if self.state == self.MOVE else 0.5
            )
            action[4] = GRIPPER_CLOSE

        else:  # RELEASE
            wp = np.array(
                [self.tray_xy[0], self.tray_xy[1], RETREAT_HEIGHT, self.yaw_target],
                dtype=np.float32,
            )
            action = self._goto(wp, info, z_rate=1.0)
            action[4] = GRIPPER_OPEN

        return action


def rollout_episode(env: PickPlaceEnv, expert: FSMExpert, seed: int):
    """运行单回合, 剔除末尾无动作的 40 步物理等待"""
    obs, info = env.reset(seed=seed)
    expert.reset()

    buffer = []
    steps = 0
    terminated = truncated = False

    while not (terminated or truncated) and steps < MAX_STEPS and not expert.failed:
        action = expert.step(info)

        # 核心优化: 只要机械臂撤退到安全位置, 就不再将后续静止数据写入 Buffer
        if not expert.recording_done:
            frame = {"state": obs["state"], "action": action}
            if "top" in obs:
                frame["top"] = obs["top"]
                frame["wrist"] = obs["wrist"]
            buffer.append(frame)

        # 环境物理循环继续推进, 直到满足 2 秒静止成功判定 (SUCCESS_HOLD_STEPS=40)
        next_obs, _, terminated, truncated, next_info = env.step(action)
        obs = next_obs
        info = next_info
        steps += 1

    success = bool(terminated and info.get("success"))
    reason = ""
    if not success:
        if expert.failed:
            reason = expert.fail_reason
        elif truncated:
            reason = "env_truncated"
        elif info.get("cube_escaped"):
            reason = "cube_escaped"
        else:
            reason = "step_cap"

    return buffer, success, len(buffer), steps, reason
