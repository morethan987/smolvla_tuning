"""
示教数据抽样检验与可视化 (针对 VLA 训练优化版)
- 动作健康度深度体检: 全维度零动作比例、整行全0统计、动态分布
- 生成关键步骤拼图 (outputs/traj_sample.png)
- 生成左右双视角 + 动作遥测 HUD 视频 (outputs/traj_sample.mp4)

运行: uv run scripts/inspect_dataset.py
指定轨迹: uv run scripts/inspect_dataset.py --ep_idx 0
"""

import argparse
import pickle
import random
from pathlib import Path

import numpy as np

from smolvla_tuning.utils import create_contact_sheet, create_video

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def inspect_action_health(episodes: list[list[dict]], split_name: str = "Train"):
    """体检动作空间健康度，深度排查 0 动作与动作塌缩隐患"""
    all_actions = []
    for ep in episodes:
        for step in ep:
            all_actions.append(step["action"])

    if not all_actions:
        return

    actions = np.array(all_actions, dtype=np.float32)  # shape: (N, 5)
    total_steps = len(actions)

    # 1. 整行全 0 统计 (所有维度绝对值 < 1e-3)
    full_zero_mask = np.all(np.abs(actions) < 1e-3, axis=1)
    full_zero_count = int(np.sum(full_zero_mask))
    full_zero_ratio = full_zero_count / total_steps * 100.0

    # 2. 单维度 0 动作统计
    channel_zero_ratios = np.mean(np.abs(actions) < 1e-3, axis=0) * 100.0

    # 3. 夹爪开闭分布
    grip = actions[:, 4]
    open_ratio = np.mean(grip > 0.5) * 100.0
    close_ratio = np.mean(grip < -0.5) * 100.0
    mid_ratio = 100.0 - open_ratio - close_ratio

    print(f"\n--- [{split_name}] 动作健康体检 (共 {total_steps} 步) ---")
    print(
        f"  * [致命隐患检测] 整行全 0 步数: {full_zero_count}/{total_steps} ({full_zero_ratio:.2f}%)"
    )
    if full_zero_ratio > 1.0:
        print("    [警告] 存在连续全 0 动作，容易诱发 VLA 策略推理原地冻结！")
    else:
        print("    [健康] 全 0 动作比例极低/已清除，动作流连续。")

    names = ["Δx", "Δy", "Δz", "Δyaw", "Grip"]
    print("  * 各通道零值率 (Sparsity) 与范围:")
    for i, name in enumerate(names):
        mean_val = float(np.mean(actions[:, i]))
        std_val = float(np.std(actions[:, i]))
        min_val = float(np.min(actions[:, i]))
        max_val = float(np.max(actions[:, i]))
        print(
            f"    - {name:<5}: 0值率={channel_zero_ratios[i]:5.1f}% | 区间=[{min_val:+.2f}, {max_val:+.2f}] | 均值={mean_val:+.2f} ± {std_val:.2f}"
        )

    print(
        f"  * 夹爪动作意图分布: 张开(+1)={open_ratio:.1f}% | 闭合(-1)={close_ratio:.1f}% | 过渡态={mid_ratio:.1f}%"
    )


def load_dataset_stats(dataset_path: Path):
    print(f">> 正在读取数据集: {dataset_path} ...")
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"找不到数据集文件: {dataset_path}，请先运行数据采集脚本！"
        )

    with open(dataset_path, "rb") as f:
        data = pickle.load(f)

    train_eps = data.get("train", [])
    val_eps = data.get("val", [])

    train_lens = [len(ep) for ep in train_eps]
    val_lens = [len(ep) for ep in val_eps]

    print("\n========== 数据集宏观概况 ==========")
    print(
        f"训练集 (Train): {len(train_eps)} 条 | 步数: [{min(train_lens)}, {max(train_lens)}] | 平均: {np.mean(train_lens):.1f}"
    )
    print(
        f"验证集 (Val)  : {len(val_eps)} 条 | 步数: [{min(val_lens)}, {max(val_lens)}] | 平均: {np.mean(val_lens):.1f}"
    )

    sample_step = train_eps[0][0]
    has_images = "top" in sample_step and "wrist" in sample_step
    print(f"数据模态: {'[图像 + 状态]' if has_images else '[仅低维状态 (No Images)]'}")
    print(
        f"  - state 状态 : shape={sample_step['state'].shape}, dtype={sample_step['state'].dtype}"
    )
    print(
        f"  - action 动作: shape={sample_step['action'].shape}, dtype={sample_step['action'].dtype}"
    )
    if has_images:
        print(
            f"  - top 相机   : shape={sample_step['top'].shape}, dtype={sample_step['top'].dtype}"
        )
        print(
            f"  - wrist 相机 : shape={sample_step['wrist'].shape}, dtype={sample_step['wrist'].dtype}"
        )

    # 深度体检动作质量
    inspect_action_health(train_eps, "Train")
    print("=====================================\n")

    return data, has_images


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset", type=str, default="data/pick_place_dataset.pkl", help="数据集路径"
    )
    p.add_argument(
        "--ep_idx", type=int, default=None, help="指定抽取的 Episode 索引 (默认随机)"
    )
    p.add_argument("--fps", type=int, default=20, help="视频输出帧率")
    return p.parse_args()


def main():
    args = parse_args()
    data, has_images = load_dataset_stats(Path(args.dataset))

    train_eps = data["train"]
    if not train_eps:
        print("[提示] 训练集为空，退出可视化。")
        return

    # 选择 Episode
    if args.ep_idx is not None:
        sample_idx = max(0, min(args.ep_idx, len(train_eps) - 1))
    else:
        sample_idx = random.randint(0, len(train_eps) - 1)

    episode = train_eps[sample_idx]
    print(f">> 抽样第 {sample_idx} 条轨迹进行可视化 (长度: {len(episode)} 步)...")

    if has_images:
        # 1. 输出拼图
        create_contact_sheet(episode, OUTPUT_DIR / f"sample_ep{sample_idx}_strip.png")
        # 2. 输出视频
        create_video(episode, OUTPUT_DIR / f"sample_ep{sample_idx}.mp4", fps=args.fps)
    else:
        print("[提示] 数据集中不含图像，已跳过图片和视频渲染。")


if __name__ == "__main__":
    main()
