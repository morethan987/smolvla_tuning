"""
步骤2: 启发式专家轨迹采集脚本 (面向 VLA 微调优化版)

专家策略 FSMExpert (虚拟牵引绳 + 多轴协同平滑规划) 与单回合 rollout 逻辑
位于 smolvla_tuning.expert.scripted_policy; 本脚本只负责批量采集循环、
失败过滤与落盘。

运行: uv run scripts/collect_data.py
调试: uv run scripts/collect_data.py --no-images --num-train 3 --num-val 2
"""

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from smolvla_tuning.envs import PickPlaceEnv
from smolvla_tuning.expert import FSMExpert, rollout_episode


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
        f">> 开始采集优化示教数据: 目标 {total} 条 ({args.num_train} Train / {args.num_val} Val)..."
    )
    print("   [特性]: 牵引绳限幅防横扫 | 多轴联合平滑规划 | 零硬死区 | 剔除尾部空转")

    while success_count < total:
        if attempts >= max_attempts:
            raise SystemExit(
                f"[中止] {attempts} 次尝试仅成功 {success_count}/{total} "
                f"(成功率 {success_count / attempts * 100:.1f}%), 策略存在异常."
            )
        attempts += 1
        seed = args.seed_start + attempts - 1
        buffer, success, recorded_len, env_steps, reason = rollout_episode(
            env, expert, seed
        )

        if success:
            split = "train" if len(dataset["train"]) < args.num_train else "val"
            dataset[split].append(buffer)
            success_count += 1
            print(
                f"[{success_count:03d}/{total}] {split.upper()} 成功 | seed={seed} | "
                f"记录步数: {recorded_len} (仿真总步: {env_steps}) | 尝试: {attempts} | "
                f"累计成功率: {success_count / attempts * 100:.0f}%"
            )
        else:
            print(f"    [丢弃] seed={seed} | 步数: {env_steps} | 原因: {reason}")

    env.close()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

    all_lens = [len(ep) for ep in dataset["train"] + dataset["val"]]
    print(f"\n>> 数据采集完成! 已保存至 {output}")
    print(f"   - 总体成功率: {success_count / attempts * 100:.1f}% ({attempts} 次尝试)")
    print(
        f"   - 净有效轨迹步数: min={min(all_lens)} / mean={np.mean(all_lens):.1f} / max={max(all_lens)}"
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
