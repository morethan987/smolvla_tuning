"""
将 pick_place_dataset.pkl 转换为 LeRobot 原生格式并自动计算归一化统计量 (stats.safetensors)

uv run scripts/convert_dataset.py
"""

import pickle
import shutil
from pathlib import Path

from lerobot.datasets import LeRobotDataset

PKL_PATH = Path("data/pick_place_dataset.pkl")
OUTPUT_DIR = Path("data/lerobot_pick_place")
REPO_ID = "local/pick_place"
TASK_DESC = "Pick up the red cube and place it into the tray"
FPS = 20  # 环境默认控制频率


def convert():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    with open(PKL_PATH, "rb") as f:
        raw_data = pickle.load(f)

    # 合并 train 和 val（LeRobot 会在加载时按比例划分，或全量微调）
    all_episodes = raw_data["train"] + raw_data["val"]

    sample_frame = all_episodes[0][0]
    img_h, img_w, img_c = sample_frame["top"].shape
    state_dim = sample_frame["state"].shape[0]
    action_dim = sample_frame["action"].shape[0]

    # 1. 声明特征规范: 直接对齐 SmolVLA 期望的 camera1 和 camera2
    features = {
        "observation.images.camera1": {
            "dtype": "video",
            "shape": (img_h, img_w, img_c),
            "names": ["height", "width", "channel"],
        },
        "observation.images.camera2": {
            "dtype": "video",
            "shape": (img_h, img_w, img_c),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["action"],
        },
    }

    # 2. 初始化容器
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        root=OUTPUT_DIR,
        robot_type="panda",
        features=features,
        use_videos=True,
    )

    # 3. 逐帧录入
    for ep_idx, ep in enumerate(all_episodes):
        for frame in ep:
            dataset.add_frame(
                {
                    "observation.images.camera1": frame["top"],
                    "observation.images.camera2": frame["wrist"],
                    "observation.state": frame["state"].astype("float32"),
                    "action": frame["action"].astype("float32"),
                    "task": TASK_DESC,
                }
            )
        dataset.save_episode()
        print(
            f"Episode [{ep_idx + 1:03d}/{len(all_episodes)}] 写入完成, 共 {len(ep)} 帧"
        )

    # 4. 生成归一化参数与元数据
    dataset.finalize()
    print(f"\n>> 数据集转换完成！保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert()
