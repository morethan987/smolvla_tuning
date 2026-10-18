"""
步骤1:环境渲染与动作通断测试

验证内容:
  A. Gymnasium API 合规(check_env)
  B. 随机动作 rollout 无崩溃、obs/dtype 合法
  C. IK 跟踪: 持续指向方块悬停点, 末端应显著逼近
  D. 成功判定: 把方块放进托盘 → 2s 后 terminated=True
  E. 图像观测(需 OpenGL; 无 GPU 环境会给出提示后跳过)

运行: uv run scripts/test_env.py
"""

from pathlib import Path

import numpy as np

from smolvla_tuning.envs import PickPlaceEnv


def test_api_and_random_rollout() -> None:
    from gymnasium.utils.env_checker import check_env

    env = PickPlaceEnv(image_obs=False)
    check_env(env, skip_render_check=True)
    print("[A] check_env: API 合规通过")

    obs, info = env.reset(seed=0)
    assert obs["state"].shape == (6,) and obs["state"].dtype == np.float32
    print(
        f"[B] reset(seed=0): state={np.round(obs['state'], 3)}, cube={np.round(info['cube_pos'], 3)}"
    )

    total = 0.0
    for t in range(80):
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        total += r
        assert obs["state"].shape == (6,)
        if terminated or truncated:
            print(f"    step {t}: terminated={terminated} truncated={truncated}")
            break
    print(
        f"[B] 随机动作 80 步: 无崩溃, 累计 reward={total:.2f}, 方块位移 "
        f"{np.linalg.norm(info['cube_pos'] - np.array([0.0, -0.22, 0.016])):.3f} m"
    )
    env.close()


def test_ik_hover() -> None:
    env = PickPlaceEnv(image_obs=False)
    _obs, info = env.reset(seed=7)
    cube = info["cube_pos"]
    hover = np.array([cube[0], cube[1], 0.06])
    d0 = float(np.linalg.norm(info["eef_pose"][:3] - hover))
    for _ in range(200):
        err = hover - info["eef_target"][:3]
        if np.linalg.norm(info["eef_pose"][:3] - hover) < 0.005:
            break
        action = np.zeros(5, np.float32)
        action[:3] = np.clip(err / 0.03, -1.0, 1.0)  # 0.03 = POS_SCALE, 按"目标"闭环
        _obs, _, _, _, info = env.step(action)
    d1 = float(np.linalg.norm(info["eef_pose"][:3] - hover))
    print(
        f"[C] IK 悬停跟踪: 距悬停点 {d0 * 100:.1f} cm → {d1 * 100:.1f} cm "
        f"({'通过' if d1 < 0.03 and d1 < d0 else '未收敛'})"
    )
    env.close()


def test_success_detection() -> None:
    import mujoco

    env = PickPlaceEnv(image_obs=False)
    env.reset(seed=3)
    qa = env._cube_qadr  # 测试钩子: 直接摆位方块
    env.data.qpos[qa : qa + 7] = [0.21, -0.08, 0.0235, 1, 0, 0, 0]
    env.data.qvel[:6] = 0
    mujoco.mj_forward(env.model, env.data)

    null_action = np.zeros(5, np.float32)
    terminated, truncated, steps, info = False, False, 0, {}
    while not (terminated or truncated) and steps < 80:
        _, _, terminated, truncated, info = env.step(null_action)
        steps += 1
    ok = terminated and info.get("success")
    print(
        f"[D] 成功判定: {steps} 步后 terminated={terminated}, success={info.get('success')} "
        f"({'通过' if ok else '失败'})"
    )
    env.close()


def test_images() -> None:
    try:
        env = PickPlaceEnv(image_obs=True, image_size=224)
        obs, _ = env.reset(seed=1)
    except RuntimeError as e:
        print(f"[E] 图像观测跳过: {e}")
        return
    from imageio.v2 import imwrite

    imwrite("outputs/env_top.png", obs["top"])
    imwrite("outputs/env_wrist.png", obs["wrist"])
    print("[E] 图像观测: top/wrist (224x224x3 uint8) 已存 outputs/env_*.png")
    env.close()


if __name__ == "__main__":
    Path("outputs").mkdir(exist_ok=True)
    test_api_and_random_rollout()
    test_ik_hover()
    test_success_detection()
    test_images()
    print("\n01_test_env 全部完成")
