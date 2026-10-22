"""
步骤5: SmolVLA 闭环评测脚本 (成功率统计 + 视频录制)
运行: uv run scripts/evaluate.py --num-episodes 5 --save-video-mode all
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import numpy as np
import torch
from imageio.typing import ArrayLike
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAConfig, SmolVLAPolicy

from smolvla_tuning.envs.pick_place_env import (
    MAX_EPISODE_STEPS,
    TASK,
    PickPlaceEnv,
)

DEFAULT_CHECKPOINT = (
    "outputs/train/smolvla_pick_place/checkpoints/last/pretrained_model"
)
DEFAULT_OUTPUT_DIR = "outputs/eval_results"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT)
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--start-seed", type=int, default=1000)
    p.add_argument("--num-episodes", type=int, default=5)
    p.add_argument(
        "--action-horizon",
        type=int,
        default=50,
        help="动作分块执行步数: 50=官方默认整块执行; 较小值(如官方 RTC 建议的 10)提高闭环纠偏频率, 仅用于消融对比",
    )
    p.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument(
        "--save-video-mode",
        type=str,
        default="all",
        choices=["all", "fail", "none"],
        help="视频保存: all=全部 | fail=仅失败回合 | none=不保存",
    )
    return p.parse_args()


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    print(f">> 加载模型: {args.checkpoint_dir}")
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint_dir).to(args.device).eval()

    # 动作分块执行步数(官方默认 50 整块执行; 较小值仅用于消融, 不作为找补手段)
    assert isinstance(policy.config, SmolVLAConfig)
    policy.config.n_action_steps = args.action_horizon
    print(f">> 动作分块执行步数: n_action_steps = {args.action_horizon}")

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.checkpoint_dir,
        preprocessor_overrides={"device_processor": {"device": str(args.device)}},
    )

    env = PickPlaceEnv(image_obs=True, image_size=224)
    success_count = 0

    for seed in range(args.start_seed, args.start_seed + args.num_episodes):
        set_seed(seed)
        obs, _ = env.reset(seed=seed)
        policy.reset()

        frames: list[ArrayLike] = []
        is_success = False

        for step in range(MAX_EPISODE_STEPS):
            if args.save_video_mode != "none":
                frames.append(np.hstack([obs["top"], obs["wrist"]]))

            top_t = (
                torch.from_numpy(obs["top"]).permute(2, 0, 1).contiguous().float()
                / 255.0
            )
            wrist_t = (
                torch.from_numpy(obs["wrist"]).permute(2, 0, 1).contiguous().float()
                / 255.0
            )
            state_t = torch.from_numpy(obs["state"]).contiguous().float()

            batch = {
                "observation.images.camera1": top_t.unsqueeze(0).to(args.device),
                "observation.images.camera2": wrist_t.unsqueeze(0).to(args.device),
                "observation.state": state_t.unsqueeze(0).to(args.device),
                "task": [TASK],
            }
            if preprocessor is not None:
                batch = preprocessor(batch)

            with torch.no_grad():
                action = policy.select_action(batch)
                if postprocessor is not None:
                    action = postprocessor(action)

            act_numpy = action.detach().to("cpu").numpy().reshape(-1)

            # 不做任何评测期缩放/限幅: env.step 内部自带的 [-1,1] 限幅即全部约束,
            # 保证评测动作语义与训练数据完全一致。
            obs, _, terminated, truncated, info = env.step(act_numpy)

            if info.get("success", False):
                is_success = True
            if terminated or truncated:
                break

        if is_success:
            success_count += 1

        tag = "success" if is_success else "fail"
        print(f"Seed {seed}: {tag.upper()} | 步数: {step}")
        should_save = args.save_video_mode == "all" or (
            args.save_video_mode == "fail" and not is_success
        )
        if should_save and len(frames) > 0:
            imageio.mimsave(str(video_dir / f"seed_{seed}_{tag}.mp4"), frames, fps=20)

    env.close()
    print(
        f"\n>> 评测完成! 成功率: {success_count}/{args.num_episodes} ({(success_count / args.num_episodes) * 100:.1f}%)"
    )
    if args.save_video_mode != "none":
        print(f">> 视频已保存至: {video_dir}")


if __name__ == "__main__":
    main()
