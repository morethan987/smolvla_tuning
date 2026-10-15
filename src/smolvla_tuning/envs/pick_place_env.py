"""
SO-100 机械臂抓放方块任务的标准 Gymnasium 环境

动作空间(5 维, 全部 [-1, 1]):
    Δx, Δy, Δz  : 末端(指尖)笛卡尔增量, 每步最多 3 cm
    Δyaw        : 末端绕世界 z 轴偏航增量, 每步最多 0.25 rad
    gripper     : >0 张开 / <0 闭合, 线性映射到夹爪行程 (1=开, 0=关, 与数据/评测统一)

观测空间(Dict):
    top / wrist : uint8 (H, W, 3), 俯视与手腕相机 (image_obs=True 时)
    state       : float32 (6,), 5 个臂关节角 + 夹爪开度归一化(1=开)

终止语义:
    terminated = True  → 方块在托盘内稳定保持 2 s(成功)或被推离桌面范围(失败)
    truncated  = True  → 600 步(30 s)超时

内部控制: 动作先积分到笛卡尔目标位姿, 经阻尼最小二乘 IK 转成关节位置目标,
由 MuJoCo position 执行器跟踪; 每个控制步执行 n_substeps 个 2 ms 物理子步。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

XML_PATH = Path(__file__).resolve().parent / "assets" / "so_arm100" / "pick_place.xml"
TASK = "Pick up the red cube and place it into the tray"

# 关节拓扑定义(与 XML actuator 顺序对齐)
ARM_JOINTS = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll")
JAW_JOINT = "Jaw"

# ---- 任务规则与数据分布 (与 XML 无关的高层任务设定) ----
CUBE_SPAWN_X = (-0.12, 0.08)
CUBE_SPAWN_Y = (-0.30, -0.16)
MAX_EPISODE_STEPS = 600  # 30 s
SUCCESS_HOLD_STEPS = 40  # 方块需在托盘内稳定保持 2 s
CUBE_SPEED_STILL = 0.02  # m/s, 视为静止
CUBE_ESCAPE_RADIUS = 0.45  # 方块被推离此半径判定失败

# ---- 控制器算法超参数 ----
CONTROL_DT = 0.05  # 20 Hz 控制频率
POS_SCALE = 0.03  # 3 cm / 步
YAW_SCALE = 0.25  # 0.25 rad / 步
JAW_OPEN = 0.65  # 抓取任务所需开度 (~5cm 容纳 3cm 方块)
IK_DAMPING = 0.1
DQ_MAX = 0.08  # 单步关节增量安全阀(rad)

# 工作空间
TARGET_XY_LOW = np.array([-0.31, -0.34], dtype=np.float32)
TARGET_XY_HIGH = np.array([0.31, 0.26], dtype=np.float32)
TARGET_Z_LOW, TARGET_Z_HIGH = 0.018, 0.30


class PickPlaceEnv(gym.Env):
    """SO-100 抓放方块环境, 兼容 gymnasium.Env 完整契约"""

    def __init__(self, image_obs: bool = True, image_size: int = 224):
        super().__init__()
        self.metadata = {
            "render_modes": ["rgb_array"],
            "render_fps": int(1.0 / CONTROL_DT),
        }
        self.image_obs = image_obs
        self.image_size = image_size
        self._renderer: mujoco.Renderer | None = None

        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.data = mujoco.MjData(self.model)

        tray_body = self.model.body("tray")
        self.tray_xy = tray_body.pos[:2].copy()  # 从 XML 托盘 body 提取 (x, y)
        tray_geom = self.model.geom("tray_bottom")
        self.tray_inner_half = float(tray_geom.size[0])  # 从底板提取内半宽

        cube_geom = self.model.geom("cube_geom")
        self.cube_half = float(cube_geom.size[0])  # 从方块提取半宽 (0.015m)
        self.cube_spawn_z = float(cube_geom.size[2]) + 0.0005  # 微高于地面防止穿模

        self._arm_qadr = np.array([self.model.joint(n).qposadr[0] for n in ARM_JOINTS])
        self._arm_cadr = np.array(
            [self.model.actuator_ctrladr[self.model.actuator(n).id] for n in ARM_JOINTS]
        )
        jaw_joint = self.model.joint(JAW_JOINT)
        self._jaw_qadr = jaw_joint.qposadr[0]
        self._jaw_cadr = self.model.actuator_ctrladr[self.model.actuator(JAW_JOINT).id]
        # 从模型动态读取物理闭合极限值 (-0.174 rad)
        self.jaw_closed = float(self.model.jnt_range[jaw_joint.id, 0])

        self._cube_qadr = self.model.joint("cube_joint").qposadr[0]
        self._cube_dadr = self.model.joint("cube_joint").dofadr[0]
        self._cube_id = self.model.body("red_cube").id
        self._site_id = self.model.site("grasp_site").id
        self._home_key = self.model.key("home").id
        self._arm_range = self.model.jnt_range[
            [self.model.joint(n).id for n in ARM_JOINTS]
        ]

        self._n_substeps = max(1, round(CONTROL_DT / self.model.opt.timestep))
        self._elapsed = 0
        self._hold = 0
        self._ik_target = np.zeros(4, dtype=np.float32)  # x, y, z, yaw

        # space declaration
        state_low = np.array([-2 * np.pi] * len(ARM_JOINTS) + [0.0], dtype=np.float32)
        state_high = np.array([2 * np.pi] * len(ARM_JOINTS) + [1.0], dtype=np.float32)
        obs: dict[str, spaces.Space] = {
            "state": spaces.Box(low=state_low, high=state_high, dtype=np.float32)
        }
        if image_obs:
            img = spaces.Box(0, 255, (image_size, image_size, 3), np.uint8)
            obs = {"top": img, "wrist": img, **obs}
        self.observation_space = spaces.Dict(obs)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)

    # Gymnasium API
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key)
        self.data.ctrl[:] = self.model.key("home").ctrl
        self.data.ctrl[self._jaw_cadr] = JAW_OPEN  # 初始抬手且夹爪张开

        self._randomize_cube()
        mujoco.mj_forward(self.model, self.data)

        self._elapsed = 0
        self._hold = 0
        self._ik_target[:] = self._eef_pose()
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)

        # 1) 笛卡尔目标积分 + 工作空间限幅
        self._ik_target[:2] = np.clip(
            self._ik_target[:2] + POS_SCALE * action[:2], TARGET_XY_LOW, TARGET_XY_HIGH
        )
        self._ik_target[2] = float(
            np.clip(
                self._ik_target[2] + POS_SCALE * action[2], TARGET_Z_LOW, TARGET_Z_HIGH
            )
        )
        self._ik_target[3] = _wrap_angle(self._ik_target[3] + YAW_SCALE * action[3])

        # 2) 夹爪(极性约定: action=+1 → JAW_OPEN, action=-1 → self.jaw_closed)
        self.data.ctrl[self._jaw_cadr] = self.jaw_closed + (action[4] + 1.0) * 0.5 * (
            JAW_OPEN - self.jaw_closed
        )

        # 3) IK: 笛卡尔目标 → 关节位置目标
        dq = self._solve_ik_delta()
        self.data.ctrl[self._arm_cadr] = np.clip(
            self.data.qpos[self._arm_qadr] + dq,
            self._arm_range[:, 0],
            self._arm_range[:, 1],
        )

        # 4) 物理子步
        self._elapsed += 1
        for _ in range(self._n_substeps):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        info = self._get_info()
        reward = -info["distance"] + (10.0 if info["success"] else 0.0)
        return (
            obs,
            reward,
            info["success"] or info["cube_escaped"],
            self._elapsed >= MAX_EPISODE_STEPS,
            info,
        )

    def render(self):
        if not self.image_obs:
            raise RuntimeError("render() 需要 image_obs=True")
        return self._render_camera("top")

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _randomize_cube(self):
        xy = self.np_random.uniform(
            (CUBE_SPAWN_X[0], CUBE_SPAWN_Y[0]), (CUBE_SPAWN_X[1], CUBE_SPAWN_Y[1])
        )
        yaw = self.np_random.uniform(-np.pi, np.pi)
        c, s = np.cos(yaw / 2), np.sin(yaw / 2)
        self.data.qpos[self._cube_qadr : self._cube_qadr + 7] = [
            *xy,
            self.cube_spawn_z,
            c,
            0,
            0,
            s,
        ]
        self.data.qvel[self._cube_dadr : self._cube_dadr + 6] = 0

    def _eef_pose(self) -> np.ndarray:
        """指尖抓取位姿 (x, y, z, yaw)"""
        pos = self.data.site_xpos[self._site_id].copy()
        opening = self.data.site_xmat[self._site_id].reshape(3, 3) @ np.array(
            [1.0, 0.0, 0.0]
        )
        yaw = np.arctan2(opening[1], opening[0])
        return np.array([pos[0], pos[1], pos[2], yaw], dtype=np.float32)

    def _solve_ik_delta(self) -> np.ndarray:
        """阻尼最小二乘 IK: 4D 任务(xyz + yaw) → 5 个臂关节增量"""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._site_id)
        jac_arm_p = jacp[:, self._arm_qadr]
        jac_arm_r = jacr[2, self._arm_qadr]
        J = np.vstack([jac_arm_p, jac_arm_r])  # (4, 5)

        err = self._ik_target - self._eef_pose()
        err[3] = _wrap_angle(err[3])
        JJt = J @ J.T + IK_DAMPING**2 * np.eye(4)
        dq = J.T @ np.linalg.solve(JJt, err)
        return np.clip(dq, -DQ_MAX, DQ_MAX)

    def _get_obs(self) -> dict[str, np.ndarray]:
        state = np.empty(len(ARM_JOINTS) + 1, dtype=np.float32)
        state[:-1] = self.data.qpos[self._arm_qadr]
        state[-1] = np.clip(
            (self.data.qpos[self._jaw_qadr] - self.jaw_closed)
            / (JAW_OPEN - self.jaw_closed),
            0.0,
            1.0,
        )
        obs: dict[str, Any] = {"state": state}
        if self.image_obs:
            obs["top"] = self._render_camera("top")
            obs["wrist"] = self._render_camera("wrist")
        return obs

    def _get_info(self) -> dict[str, Any]:
        cube_p = self.data.xpos[self._cube_id].copy()
        cube_v = float(
            np.linalg.norm(self.data.qvel[self._cube_dadr : self._cube_dadr + 3])
        )
        eef = self._eef_pose()
        in_tray = (
            abs(cube_p[0] - self.tray_xy[0]) < self.tray_inner_half - 0.005
            and abs(cube_p[1] - self.tray_xy[1]) < self.tray_inner_half - 0.005
            and 0.005 < cube_p[2] < 0.045
        )
        if in_tray and cube_v < CUBE_SPEED_STILL:
            self._hold += 1
        else:
            self._hold = 0
        success = self._hold >= SUCCESS_HOLD_STEPS
        return {
            "success": bool(success),
            "cube_escaped": bool(np.linalg.norm(cube_p[:2]) > CUBE_ESCAPE_RADIUS),
            "cube_pos": cube_p,
            "eef_pose": eef,
            "eef_target": self._ik_target.copy(),
            "distance": float(np.linalg.norm(eef[:3] - cube_p)),
            "elapsed": self._elapsed,
        }

    def _render_camera(self, name: str) -> np.ndarray:
        if self._renderer is None:
            try:
                self._renderer = mujoco.Renderer(
                    self.model, height=self.image_size, width=self.image_size
                )
            except Exception as e:
                raise RuntimeError(
                    "创建 MuJoCo 离屏渲染器失败: 无可用 OpenGL 后端。"
                    "云 GPU 机器请先 `export MUJOCO_GL=egl`(plan.md 模块三第 1 条)。"
                ) from e
        self._renderer.update_scene(self.data, camera=name)
        return self._renderer.render().copy()


def _wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)
