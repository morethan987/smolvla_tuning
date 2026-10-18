"""
示教数据抽样检验与可视化
- 打印数据集基本健康统计 (步数分布、键值检验)
- 生成关键步骤拼图 (outputs/traj_sample.png)
- 生成左右双视角视频 (outputs/traj_sample.mp4)

uv run scripts/03_inspect_dataset.py
"""

import pickle
import random
from pathlib import Path

import cv2
import numpy as np

DATASET_PATH = Path("data/pick_place_dataset.pkl")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_dataset_stats():
    print(">> 正在读取数据集 (文件大小较大，稍等几秒)...")
    with open(DATASET_PATH, "rb") as f:
        data = pickle.load(f)

    train_eps = data.get("train", [])
    val_eps = data.get("val", [])

    train_lens = [len(ep) for ep in train_eps]
    val_lens = [len(ep) for ep in val_eps]

    print("\n========== 数据集健康体检 ==========")
    print(
        f"训练集 (Train): {len(train_eps)} 条 | 步数区间: [{min(train_lens)}, {max(train_lens)}] | 平均步数: {np.mean(train_lens):.1f}"
    )
    print(
        f"验证集 (Val)  : {len(val_eps)} 条 | 步数区间: [{min(val_lens)}, {max(val_lens)}] | 平均步数: {np.mean(val_lens):.1f}"
    )

    sample_step = train_eps[0][0]
    print("单帧格式验证  :")
    print(
        f"  - top 相机   : shape={sample_step['top'].shape}, dtype={sample_step['top'].dtype}"
    )
    print(
        f"  - wrist 相机 : shape={sample_step['wrist'].shape}, dtype={sample_step['wrist'].dtype}"
    )
    print(
        f"  - state 状态 : shape={sample_step['state'].shape}, dtype={sample_step['state'].dtype}"
    )
    print(
        f"  - action 动作: shape={sample_step['action'].shape}, dtype={sample_step['action'].dtype}"
    )
    print("=====================================\n")

    return data


def create_contact_sheet(episode: list, save_path: Path, num_frames: int = 6):
    """提取 6 张阶段性关键帧，拼接成一张全流程长图"""
    indices = np.linspace(0, len(episode) - 1, num_frames, dtype=int)
    row_top = []
    row_wrist = []

    for idx in indices:
        step = episode[idx]
        t_img = step["top"].copy()
        w_img = step["wrist"].copy()

        # 标注步数
        cv2.putText(
            t_img, f"#{idx}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        cv2.putText(
            w_img, f"#{idx}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        row_top.append(t_img)
        row_wrist.append(w_img)

    top_banner = np.hstack(row_top)
    wrist_banner = np.hstack(row_wrist)
    contact_sheet = np.vstack([top_banner, wrist_banner])

    cv2.imwrite(str(save_path), cv2.cvtColor(contact_sheet, cv2.COLOR_RGB2BGR))
    print(f"[OK] 关键帧长图已保存: {save_path} (上排: Top 视角, 下排: Wrist 视角)")


def create_video(episode: list, save_path: Path, fps: int = 20) -> None:
    """将单条 episode 左右视角拼接，使用 OpenCV 流式写入 MP4"""
    if not episode:
        return

    # 计算画面尺寸 (top 224x224 + wrist 224x224 拼接为 448x224)
    sample_top = episode[0]["top"]
    h, w = sample_top.shape[:2]
    combined_w, combined_h = w * 2, h

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(save_path), fourcc, float(fps), (combined_w, combined_h)
    )

    for t, step in enumerate(episode):
        top_img = step["top"]
        wrist_img = step["wrist"]
        grip = "OPEN" if step["action"][4] > 0 else "CLOSE"

        # 左右拼接并转为 BGR (OpenCV 颜色通道)
        combined = np.hstack([top_img, wrist_img])
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

        # 状态标注
        cv2.putText(
            combined_bgr,
            f"Step: {t:03d}/{len(episode)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        grip_color = (0, 255, 0) if grip == "OPEN" else (0, 0, 255)
        cv2.putText(
            combined_bgr,
            f"Gripper: {grip}",
            (combined_w - 160, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            grip_color,
            2,
        )

        writer.write(combined_bgr)

    writer.release()
    print(f"[OK] 双视角回放视频已保存: {save_path}")


def main():
    data = load_dataset_stats()

    # 随机抽查训练集中的 1 个 Episode
    sample_idx = random.randint(0, len(data["train"]) - 1)
    episode = data["train"][sample_idx]
    print(f">> 随机抽样第 {sample_idx} 条轨迹进行可视化 (长度: {len(episode)} 步)...")

    # 1. 输出拼图
    create_contact_sheet(episode, OUTPUT_DIR / f"sample_ep{sample_idx}_strip.png")

    # 2. 输出视频
    create_video(episode, OUTPUT_DIR / f"sample_ep{sample_idx}.mp4")


if __name__ == "__main__":
    main()
