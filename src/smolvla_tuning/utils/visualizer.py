"""可视化辅助工具: 关键帧拼图与动作遥测回放视频"""

from pathlib import Path

import cv2
import numpy as np


def get_fourcc(codec_str: str = "mp4v"):
    """安全获取 OpenCV fourcc，兼容不同 opencv 版本"""
    if hasattr(cv2, "VideoWriter_fourcc"):
        return cv2.VideoWriter_fourcc(*codec_str)
    return cv2.VideoWriter.fourcc(*codec_str)


def create_contact_sheet(episode: list, save_path: Path, num_frames: int = 6):
    """提取阶段性关键帧拼成长图"""
    indices = np.linspace(0, len(episode) - 1, num_frames, dtype=int)
    row_top = []
    row_wrist = []

    for idx in indices:
        step = episode[idx]
        t_img = step["top"].copy()
        w_img = step["wrist"].copy()

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
    print(f"[OK] 关键帧长图已保存: {save_path} (上排: Top, 下排: Wrist)")


def create_video(episode: list, save_path: Path, fps: int = 20) -> None:
    """双视角左右拼接 + 实时 Action 遥测 HUD 写入 MP4"""
    if not episode:
        return

    sample_top = episode[0]["top"]
    h, w = sample_top.shape[:2]
    # 在图像下方留出 40px 高度用于绘制实时遥测控制条
    hud_h = 40
    combined_w, combined_h = w * 2, h + hud_h

    fourcc = get_fourcc("mp4v")
    writer = cv2.VideoWriter(
        str(save_path), fourcc, float(fps), (combined_w, combined_h)
    )

    for t, step in enumerate(episode):
        top_img = step["top"]
        wrist_img = step["wrist"]
        act = step["action"]
        grip = "OPEN" if act[4] > 0 else "CLOSE"

        # 1. 拼接相机视角
        cam_view = np.hstack([top_img, wrist_img])
        cam_view_bgr = cv2.cvtColor(cam_view, cv2.COLOR_RGB2BGR)

        # 2. 创建底部黑色 HUD 面板
        canvas = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
        canvas[:h, :combined_w] = cam_view_bgr

        # 3. 画面内标注步数与夹爪
        cv2.putText(
            canvas,
            f"Step: {t:03d}/{len(episode)}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        grip_color = (0, 255, 0) if grip == "OPEN" else (0, 0, 255)
        cv2.putText(
            canvas,
            f"Grip: {grip}",
            (combined_w - 140, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            grip_color,
            2,
        )

        # 4. 底部 HUD 标注实时的 5 维动作数值 (dx, dy, dz, dyaw, grip)
        act_text = f"Action: dx={act[0]:+.2f} | dy={act[1]:+.2f} | dz={act[2]:+.2f} | dyaw={act[3]:+.2f} | g={act[4]:+.2f}"
        cv2.putText(
            canvas,
            act_text,
            (10, h + 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            lineType=cv2.LINE_AA,
        )

        writer.write(canvas)

    writer.release()
    print(f"[OK] 动作遥测回放视频已保存: {save_path}")
