"""
步骤4: SmolVLA 监督微调执行脚本
调用 LeRobot 训练管线微调 lerobot/smolvla_base 预训练权重
训练完成后自动绘制 Loss 学习曲线

运行命令:
uv run scripts/train.py
# 显存紧张时指定 batch-size:
uv run scripts/train.py --batch-size 8 --steps 20000
PYTHONUNBUFFERED=1 nohup uv run scripts/train.py > train_run.log 2>&1 &
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

DEFAULT_DATASET_DIR = Path("data/lerobot_pick_place")
DEFAULT_REPO_ID = "local/pick_place"
DEFAULT_OUTPUT_DIR = Path("outputs/train/smolvla_pick_place")
DEFAULT_PRETRAINED_PATH = "lerobot/smolvla_base"


def check_prerequisites(dataset_dir: Path, device: str):
    """前置环境与数据集完整性检查"""
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"未找到数据集目录: {dataset_dir}，请先运行 uv run scripts/convert_dataset.py"
        )

    # 检查核心元数据与统计量
    meta_dir = dataset_dir / "meta"
    if not (meta_dir / "info.json").exists():
        raise FileNotFoundError(
            f"数据集缺少元数据 {meta_dir / 'info.json'}，请确认转换已完成 finalize。"
        )

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("检测到指定了 CUDA 但当前系统没有可用的 GPU！")

    print(">> 前置校验通过:")
    print(f"   - 本地数据集: {dataset_dir.resolve()}")
    if device == "cuda":
        print(f"   - GPU 设备: {torch.cuda.get_device_name(0)}")


def parse_step(step_str: str) -> int:
    """支持 200, 1K, 1.2K, 25K 等格式"""
    step_str = step_str.upper().strip()
    if step_str.endswith("K"):
        return int(float(step_str[:-1]) * 1000)
    elif step_str.endswith("M"):
        return int(float(step_str[:-1]) * 1000000)
    return int(step_str)


def plot_loss_curve(log_path: Path, output_dir: Path):
    """从训练日志中精准提取 step 与 loss，生成平滑曲线图"""
    if not log_path.exists():
        print("未找到训练日志，跳过绘制 Loss 曲线。")
        return

    steps, losses = [], []

    # 精准匹配 train 循环中的 `step:xxx ... loss:xxx`
    # 兼顾纯数字以及 1K, 1.2K 等格式，避免匹配到 eval/step
    pattern = re.compile(
        r"(?:ot_train\.py:\d+\s+)?step:([0-9.]+[kKmM]?)\s+.*?loss:([0-9.]+)"
    )

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # 过滤掉非 training 阶段（如 eval 阶阶段）
            if "eval" in line.lower() and "loss" not in line.lower():
                continue

            match = pattern.search(line)
            if match:
                try:
                    step_val = parse_step(match.group(1))
                    loss_val = float(match.group(2))
                    steps.append(step_val)
                    losses.append(loss_val)
                except ValueError:
                    continue

    if not steps:
        print(">> 未能在日志中提取到 step/loss 数据，跳过绘图。")
        return

    df = pd.DataFrame({"step": steps, "loss": losses})

    # 1. 过滤异常过小步数（如 eval 阶段重置的 step）
    # 2. 按 step 排序并去重（保留最新值）
    df = (
        df.drop_duplicates(subset=["step"], keep="last")
        .sort_values("step")
        .reset_index(drop=True)
    )

    csv_path = output_dir / "loss_curve.csv"
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(df["step"], df["loss"], alpha=0.3, color="gray", label="Raw Loss")

    # 移动平均平滑
    if len(df) >= 10:
        window_size = min(30, max(3, len(df) // 10))
        smoothed = df["loss"].rolling(window=window_size, min_periods=1).mean()
        plt.plot(
            df["step"], smoothed, color="crimson", linewidth=2, label="Smoothed Loss"
        )

    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("SmolVLA Fine-tuning Loss Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    fig_path = output_dir / "loss_curve.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f">> Loss 学习曲线已重新绘制保存至: {fig_path.resolve()}")


def run_training(args: argparse.Namespace):
    output_dir = Path(args.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir)

    check_prerequisites(dataset_dir, args.device)

    # 相机槽位说明(官方 docs "Rename Map and Empty Cameras"):
    # smolvla_base 预训练布局为 camera1/camera2/camera3 三个相机槽;
    # 本地数据集已直接使用 camera1/camera2 键名, 无需 rename_map,
    # 缺的第 3 槽用 empty_cameras=1 补齐(2 真实 + 1 空槽 = 3 槽)。
    # 组装 LeRobot 原生训练命令
    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--policy.path={args.pretrained_path}",
        f"--dataset.root={dataset_dir!s}",
        f"--dataset.repo_id={args.repo_id}",
        f"--output_dir={output_dir!s}",
        f"--job_name={args.job_name}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--save_freq={args.save_freq}",
        f"--policy.device={args.device}",
        "--policy.empty_cameras=1",
        "--policy.push_to_hub=false",
    ]

    print("\n>> 启动 SmolVLA 微调进程...")
    print(">> 执行命令: " + " ".join(cmd) + "\n")

    temp_log_path = output_dir.parent / f"{output_dir.name}_train.log"
    with open(temp_log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )

        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)

        process.wait()

    if process.returncode != 0:
        raise RuntimeError(
            f"微调进程异常退出，错误码: {process.returncode}，详情请查看 {temp_log_path}"
        )

    final_log_path = output_dir / "train.log"
    shutil.move(temp_log_path, final_log_path)

    print("\n>> 训练阶段顺利结束！")
    # 解析日志绘制交付所需的 Loss 曲线图
    plot_loss_curve(final_log_path, output_dir)

    best_ckpt = output_dir / "checkpoints" / "last" / "pretrained_model"
    print(f">> 权重与配置已保存至: {best_ckpt.resolve()}")
    print(">> 可以开始执行第五步 scripts/evaluate.py 进行闭环评估！")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(DEFAULT_DATASET_DIR),
        help="LeRobot 数据集根目录",
    )
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--pretrained-path",
        type=str,
        default=DEFAULT_PRETRAINED_PATH,
        help="Hugging Face SmolVLA 基底权重",
    )
    parser.add_argument("--job-name", type=str, default="smolvla_pick_place")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=20000, help="总训练迭代步数")
    parser.add_argument(
        "--save-freq", type=int, default=10000, help="Checkpoint 检查点保存频率"
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
