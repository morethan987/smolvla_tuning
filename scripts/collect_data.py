"""
步骤2: 启发式专家轨迹采集脚本
通过 8 阶段有限状态机 (FSM) 采集 100 条成功抓放示教数据 (80 训练 / 20 验证)。

专家设计要点 (均经 100 种子实测标定, 成功率 99%):
  1. 夹爪极性: action[4] = +1 张开 / -1 闭合 (与环境 pick_place_env.py 约定一致, 勿反)。
  2. 闭环被控量: 环境把动作积分到 IK 设定点 info["eef_target"], 专家以设定点误差
     闭环 (setpoint control)。不要用 eef_pose 做误差闭环: 该被控量会引入积分
     振荡, 在远伸位形下手指会横扫方块。
  3. yaw 分阶段对齐: 位置与 yaw 联合求解时 IK 会放弃 yaw (误差可达 135°)。
     先在悬停点收敛位置, 再原地转腕。方块 4 重对称, 目标 yaw 取
     {cube_yaw + k·90°} 中离当前最近者, 转角需求 ≤45°。
  4. 抓取偏置: 固定指垫只偏 site +9~11mm, 方块半宽 15mm —— site 正对方块中心时
     固定指嵌在方块轮廓内, 下降必卡死在方块顶 (dz≈+27mm)。令抓取目标沿间隙轴
     向固定指侧偏置 +25mm, 方块被压向固定指垫, 闭合即稳定夹持。
  5. 触底检测: 下降段 site 高度停滞 6 步即视为接触, 立即转闭合, 不死等容差。
  6. 分级闭合: 先半闭 4 步再全闭, 避免位置执行器瞬间满速闭合把方块弹飞。
  7. 以方块为基准运送: 抓取偏置使方块偏离 site ~25mm, 运送/放置阶段用实测
     site-方块偏移换算, 让方块 (而非 site) 对准托盘; 托盘内半宽 74mm, 判据
     放宽到 45mm 以吸收 IK 稳态下垂。

运行: uv run scripts/collect_data.py
调试: uv run scripts/collect_data.py --no-images --num-train 3 --num-val 2
"""

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

# 云端无头渲染 (plan.md 模块三第 1 条), 需在 import mujoco 前设置
os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from smolvla_tuning.envs import PickPlaceEnv
from smolvla_tuning.envs.pick_place_env import POS_SCALE, YAW_SCALE

# ================= 控制常量配置 =================
GRIPPER_OPEN = 1.0  # action[4] = +1 → 夹爪张开 (JAW_OPEN)
GRIPPER_CLOSE = -1.0  # action[4] = -1 → 夹爪闭合

HOVER_HEIGHT = 0.06  # 悬停点高于方块中心
GRASP_Z_OFFSET = 0.002  # 抓取时 site 对准方块中心高度 (环境钳位 z>=0.018)
GRASP_GAP_BIAS = 0.025  # 抓取目标沿间隙轴向固定指侧偏置 (卡死修复, 见文件头第 4 条)
LIFT_HEIGHT = 0.12  # 托举/平移绝对高度
PLACE_HEIGHT = 0.055  # 松爪时 site 绝对高度 (高于托盘壁顶 0.032)
RETREAT_HEIGHT = 0.14  # 松爪后垂直撤退高度

HOVER_XY_TOL = 0.015  # 悬停水平到位判据
YAW_TOL = np.radians(15)  # 原地转腕到位判据
MOVE_TOL = 0.045  # 方块距托盘中心判据 (托盘内半宽 0.079, 留足余量)
DROP_DIST = 0.06  # 运送途中方块脱离末端的判据

DESCEND_STALL_STEPS = 6  # 下降停滞步数 → 视为接触
GRASP_SETTLE_STEPS = 22  # 闭合+停顿总步数 (含 4 步半闭缓冲)
GRASP_HALF_STEPS = 4  # 半闭缓冲步数
LIFT_CONFIRM_STEPS = 25  # 托举校验窗口: 方块须被带起 3 cm

MAX_STEPS = 260  # 单回合步数上限 (< 环境 600 步硬超时; 实测均值 185 / 最大 221)

# 各阶段卡死判据 (步): 依次 HOVER/YAW/DOWN/GRASP/LIFT/MOVE/PLACE/RELEASE
PHASE_TIMEOUTS = (80, 40, 50, GRASP_SETTLE_STEPS + 5, LIFT_CONFIRM_STEPS, 90, 60, 80)


def _wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


class FSMExpert:
    """基于状态机的启发式抓放专家策略 (仿真上帝视角)"""

    HOVER, YAW, DESCEND, GRASP, LIFT, MOVE, PLACE, RELEASE = range(8)

    def __init__(self, env: PickPlaceEnv):
        self.env = env
        self.tray_xy = np.asarray(env.tray_xy, dtype=np.float32)
        self.reset()

    def reset(self):
        """回合开始时从仿真真值捕获方块初始位姿 (上帝视角, plan.md 模块三)"""
        cube = np.asarray(self.env.data.xpos[self.env._cube_id], dtype=np.float32)
        quat = self.env.data.qpos[self.env._cube_qadr + 3 : self.env._cube_qadr + 7]
        w, x, y, z = quat
        cube_yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        # 4 重对称: 目标 yaw 取离当前末端 yaw 最近的 {cube_yaw + k·pi/2}
        cur_yaw = float(self.env._eef_pose()[3])
        self.yaw_target = min(
            (cube_yaw + k * np.pi / 2 for k in range(4)),
            key=lambda c: abs(_wrap_angle(c - cur_yaw)),
        )
        self.gap_dir = np.array(
            [np.cos(self.yaw_target), np.sin(self.yaw_target)], dtype=np.float32
        )
        # 抓取目标点: 方块中心 + 沿间隙轴偏置 (向固定指侧)
        self.grasp_xy = cube[:2] + GRASP_GAP_BIAS * self.gap_dir
        self.cube_z0 = float(cube[2])
        self.state = self.HOVER
        self.timer = 0
        self.prev_z = 1.0
        self.stall = 0
        self.failed = False
        self.fail_reason = ""

    # ---------------- 内部工具 ----------------
    def _abort(self, reason: str):
        self.failed = True
        self.fail_reason = reason

    def _goto(self, wp: np.ndarray, info: dict, z_rate: float = 1.0) -> np.ndarray:
        """设定点误差闭环: 把 IK 设定点直接推向航点"""
        action = np.zeros(5, dtype=np.float32)
        err_pos = wp[:3] - info["eef_target"][:3]
        err_yaw = _wrap_angle(float(wp[3] - info["eef_target"][3]))
        action[:3] = np.clip(err_pos / POS_SCALE, -1.0, 1.0)
        action[2] = np.clip(action[2], -z_rate, z_rate)  # 下降限速
        action[3] = np.clip(err_yaw / YAW_SCALE, -0.6, 0.6)
        return action

    def _advance(self, info: dict):
        cur = info["eef_pose"]
        cube = info["cube_pos"]

        if self.timer > PHASE_TIMEOUTS[self.state]:
            self._abort(f"phase_timeout@{self.state}")
            return

        if self.state == self.HOVER:
            ok_xy = np.linalg.norm(cur[:2] - self.grasp_xy) < HOVER_XY_TOL
            if ok_xy and cur[2] > self.cube_z0 + 0.030:
                self.state, self.timer = self.YAW, 0

        elif self.state == self.YAW:
            if abs(_wrap_angle(float(cur[3] - self.yaw_target))) < YAW_TOL:
                self.state, self.timer = self.DESCEND, 0

        elif self.state == self.DESCEND:
            if cur[2] < self.prev_z - 0.0002:  # 仍在下降
                self.stall = 0
            else:
                self.stall += 1
            self.prev_z = cur[2]
            contact = (
                self.stall >= DESCEND_STALL_STEPS and cur[2] > cube[2] + 0.010
            ) or cube[2] < self.cube_z0 - 0.002  # 触底 / 压块
            at_depth = abs(cur[2] - cube[2]) < 0.008
            if at_depth or contact:
                self.state, self.timer = self.GRASP, 0

        elif self.state == self.GRASP:
            if self.timer >= GRASP_SETTLE_STEPS:
                self.state, self.timer = self.LIFT, 0

        elif self.state == self.LIFT:
            if cube[2] > self.cube_z0 + 0.03:  # 方块确实随手上移
                self.state, self.timer = self.MOVE, 0
            elif self.timer >= LIFT_CONFIRM_STEPS:
                self._abort("cube_not_lifted")

        elif self.state == self.MOVE:
            if np.linalg.norm(cube[:2] - self.tray_xy) < MOVE_TOL:
                self.state, self.timer = self.PLACE, 0
            elif np.linalg.norm(cube[:2] - cur[:2]) > DROP_DIST:
                self._abort("cube_dropped_in_transit")

        elif self.state == self.PLACE:
            if cur[2] < PLACE_HEIGHT + 0.003:
                self.state, self.timer = self.RELEASE, 0

        # RELEASE 无后继: 张爪撤退后保持, 由环境 2 s 稳定判定结束回合

    # ---------------- 对外接口 ----------------
    def step(self, info: dict) -> np.ndarray:
        self.timer += 1
        self._advance(info)
        cur = info["eef_pose"]
        cube = info["cube_pos"]

        if self.state in (self.HOVER, self.DESCEND):
            z = HOVER_HEIGHT if self.state == self.HOVER else GRASP_Z_OFFSET
            wp = np.array(
                [self.grasp_xy[0], self.grasp_xy[1], self.cube_z0 + z, self.yaw_target],
                dtype=np.float32,
            )
            action = self._goto(
                wp, info, z_rate=1.0 if self.state == self.HOVER else 0.4
            )
            if self.state == self.HOVER:
                action[3] = 0.0  # 悬停只做位置, yaw 留待 YAW 阶段单独收敛
        elif self.state == self.GRASP:  # 定点保持: 仅分级闭合夹爪
            action = np.zeros(5, dtype=np.float32)
        elif self.state == self.YAW:  # 原地转腕: 仅 yaw 通道
            action = np.zeros(5, dtype=np.float32)
            action[3] = np.clip(
                _wrap_angle(float(self.yaw_target - info["eef_target"][3])) / YAW_SCALE,
                -0.6,
                0.6,
            )
        elif self.state == self.LIFT:
            wp = np.array(
                [cur[0], cur[1], LIFT_HEIGHT, self.yaw_target], dtype=np.float32
            )
            action = self._goto(wp, info)
        elif self.state in (self.MOVE, self.PLACE):  # 以方块为基准对准托盘
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
        else:  # RELEASE: 张爪并垂直撤退
            wp = np.array(
                [self.tray_xy[0], self.tray_xy[1], RETREAT_HEIGHT, self.yaw_target],
                dtype=np.float32,
            )
            action = self._goto(wp, info)

        if self.state == self.GRASP:
            # 分级闭合: 先半闭缓冲, 再全力闭合
            action[4] = 0.0 if self.timer < GRASP_HALF_STEPS else GRIPPER_CLOSE
        elif self.state in (self.LIFT, self.MOVE, self.PLACE):
            action[4] = GRIPPER_CLOSE
        else:
            action[4] = GRIPPER_OPEN
        return action


def rollout_episode(env: PickPlaceEnv, expert: FSMExpert, seed: int):
    """跑一个回合, 返回 (episode_buffer, success, 步数, 失败原因)"""
    obs, info = env.reset(seed=seed)
    expert.reset()

    buffer = []
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated) and steps < MAX_STEPS and not expert.failed:
        action = expert.step(info)
        next_obs, _, terminated, truncated, next_info = env.step(action)

        frame = {"state": obs["state"], "action": action}
        if "top" in obs:  # --no-images 调试模式只有 state
            frame["top"] = obs["top"]
            frame["wrist"] = obs["wrist"]
        buffer.append(frame)

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
    return buffer, success, steps, reason


def collect_dataset(args: argparse.Namespace):
    env = PickPlaceEnv(image_obs=not args.no_images, image_size=args.image_size)
    expert = FSMExpert(env)
    total = args.num_train + args.num_val
    max_attempts = args.max_attempts_mult * total

    dataset = {"train": [], "val": []}
    success_count = 0
    attempts = 0
    t0 = time.time()

    print(
        f">> 开始采集示教数据: 目标 {total} 条 ({args.num_train} Train / {args.num_val} Val)..."
    )
    print("   夹爪 +1=开/-1=合 | setpoint 闭环 | yaw 90°对称对齐 | 抓取偏置 +25mm")

    while success_count < total:
        if attempts >= max_attempts:
            raise SystemExit(
                f"[中止] {attempts} 次尝试仅成功 {success_count}/{total} "
                f"(成功率 {success_count / attempts * 100:.1f}%), 专家存在系统性缺陷。"
            )
        attempts += 1
        seed = args.seed_start + attempts - 1
        buffer, success, steps, reason = rollout_episode(env, expert, seed)

        if success:
            split = "train" if len(dataset["train"]) < args.num_train else "val"
            dataset[split].append(buffer)
            success_count += 1
            print(
                f"[{success_count:03d}/{total}] {split.upper()} 成功 | seed={seed} | "
                f"步数: {steps} | 尝试: {attempts} | "
                f"累计成功率: {success_count / attempts * 100:.0f}%"
            )
        else:
            print(f"    [丢弃] seed={seed} | 步数: {steps} | 原因: {reason}")

    env.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    all_lens = [len(ep) for ep in dataset["train"] + dataset["val"]]
    print(f"\n>> 数据采集完成! 已保存至 {output}")
    print(f"   - 总体成功率: {success_count / attempts * 100:.1f}% ({attempts} 次尝试)")
    print(
        f"   - 轨迹步数: min={min(all_lens)} / mean={np.mean(all_lens):.1f} / max={max(all_lens)}"
    )
    print(f"   - Train 轨迹: {len(dataset['train'])}, Val 轨迹: {len(dataset['val'])}")
    print(f"   - 耗时: {time.time() - t0:.1f} s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-train", type=int, default=80, help="训练集轨迹数")
    p.add_argument("--num-val", type=int, default=20, help="验证集轨迹数")
    p.add_argument("--seed-start", type=int, default=0, help="起始随机种子")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--no-images", action="store_true", help="快速调试: 不渲染图像")
    p.add_argument(
        "--max-attempts-mult",
        type=int,
        default=6,
        help="尝试次数上限 = 该倍数 × 目标轨迹数",
    )
    p.add_argument("--output", type=str, default="data/pick_place_dataset.pkl")
    return p.parse_args()


if __name__ == "__main__":
    collect_dataset(parse_args())
