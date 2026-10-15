# SmolVLA 微调与评测实战

---

## 模块零：任务核心概要与交付物清单

* **任务目标**：机械臂将桌面上随机位置的红色方块抓起，平移放入托盘。
* **成功判定**：方块落入托盘并稳定保持 2 秒；若超时 30 秒、掉落或剧烈碰撞则判定失败。
* **核心输入输出**：
  * **输入**：RGB 相机画面（通常为全局/俯视视角 + 手腕视角）+ 机械臂状态（关节角/夹爪开合）+ 语言指令（如 `"Pick up the red cube and place it into the tray"`）。
  * **输出**：动作序列分块（Action Chunk）。
* **最终需提交物**：
  1. 训练好的 SmolVLA 权重（Checkpoint）及配置文件。
  2. 训练/验证损失学习曲线。
  3. 50 个固定随机种子的闭环测试指标（成功率、平均耗时、碰撞率）。
  4. 成功与失败的闭环测试回放视频（`.mp4`）。

---

## 模块一：推荐项目目录结构

保持工程清晰规范，建议目录结构如下：

```text
smolvla_tuning/
├── pyproject.toml
├── uv.lock
├── README.md
├── .gitignore
├── src/
│   └── smolvla_tuning/
│       ├── __init__.py
│       ├── envs/                     # 仿真环境核心逻辑
│       │   ├── __init__.py
│       │   ├── assets/               # MuJoCo XML 资源（机械臂、方块、托盘描述文件）
│       │   │   └── pick_place.xml
│       │   └── pick_place_env.py     # 封装的标准 Gymnasium 环境
│       ├── expert/                   # 示教数据专家策略
│       │   ├── __init__.py
│       │   └── scripted_policy.py    # 启发式轨迹规划算法（上帝视角抓取）
│       └── utils/                    # 辅助工具（归一化、视频保存、坐标变换）
│           ├── __init__.py
│           └── visualizer.py
├── scripts/                          # 5 步流水线可执行脚本
│   ├── 01_test_env.py                # 步骤1：环境渲染与动作通断测试
│   ├── 02_collect_data.py            # 步骤2：自动化采集示教数据
│   ├── 03_inspect_dataset.py         # 步骤3：回放数据、提取统计量
│   ├── 04_train.py                   # 步骤4：调用 LeRobot 微调 SmolVLA
│   └── 05_evaluate.py                # 步骤5：50 种子闭环评测与录制视频
├── data/                             # 【大文件】存放转换好的 LeRobot 数据集
└── outputs/                          # 【大文件】存放模型 checkpoint 权重、日志和评测视频
```

---

## 模块二：五步实操落地指南

### 第一步：同步环境与无头渲染打通

在云端 Linux 终端中运行：

```bash
# 1. 声明无头 GPU 渲染（每次打开终端必配，可写入 ~/.bashrc）
export MUJOCO_GL=egl

# 2. 一键同步并安装全部依赖
uv sync

# 3. Generate mujoco typing binding to avoid LSP errors
uv run --with pybind11-stubgen pybind11-stubgen mujoco -o typings

# 4. 验证 GPU、PyTorch 与 MuJoCo
uv run python -c "
import torch, mujoco
print('PyTorch CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))
print('MuJoCo Version:', mujoco.__version__)
"
ls -a typings/mujoco
# Expect:
# .  ..  _callbacks.pyi  _constants.pyi  _enums.pyi  _errors.pyi  _functions.pyi  glfw  __init__.pyi  rendering  _render.pyi  _specs.pyi  _structs.pyi
```

---

### 第二步：仿真环境搭建（Gymnasium Env）

在 `envs/pick_place_env.py` 中构建环境，重点完成三个方法：

1. **`reset(seed=None)`**：
* 托盘坐标固定于 `[0.21, -0.08, 0.0]`（原定 `[0.2, 0.2, 0.0]` 经工作空间实测：手指朝下位形距其上方 41mm 不可达，已移入可达区）。
* **方块位置随机化**：在指定矩形范围内随机抽取方块 $(x, y)$ 坐标：$x \in [-0.12, 0.08], y \in [-0.30, -0.16]$（原定 $y \in [-0.2, -0.05]$ 与机械臂底座重叠且部分超出可达集，按实测内缩）。
   * 机械臂复位到初始抬手位姿。
   * 返回初次观测：图像（Top + Wrist 视角，通常调整为 224x224 或 384x384 分辨率）、机械臂当前末端位姿/关节状态与语言指令。
2. **`step(action)`**：
   * 将 action 映射至 MuJoCo 执行一步物理仿真。
   * 提取当前仿真真值：方块坐标、夹爪接触力。
3. **状态判定逻辑**：
   * 检测方块高度与托盘水平范围，若释放后在托盘边界内静止超过 2 秒即返回 `success=True, terminated=True`。
   * 单步步数达到 30 秒（对应步数如 600~900 步）标记超时终止。

> **联调验证**：运行 `uv run python scripts/01_test_env.py`，随机发几个随机动作，用 `imageio.imwrite` 存一张第一帧图像，在本地打开查看画面是否正常。

---

### 第三步：示教数据采集（强烈建议：脚本专家）

PPT 要求采集 **100 条成功轨迹**（80 训练 / 20 验证）：
* **不要用键盘手动开飞机**：手操不仅慢，且轨迹极不稳定。
* **编写启发式脚本专家（Scripted Expert Policy）**：
  * 因为在仿真器内部你是拥有**上帝视角**的（可以直接从 `sim.data` 读取当前方块和托盘的精确 3D 坐标）。
  * 设计 6 个航路点（Waypoints）的状态机：
    1. **Hover-Pregrasp**：机械臂移动到方块上方 10cm。
    2. **Descend**：垂直下降至夹爪贴住方块。
    3. **Grasp**：夹爪闭合，并停顿几帧确认抓住。
    4. **Lift**：垂直抬高 15cm。
    5. **Move-to-Tray**：平移至托盘正上方。
    6. **Release**：夹爪张开，方块落入托盘，稳定等待。
* 运行 `02_collect_data.py` 循环 100 次不同方块初始位置，自动过滤掉失败案例，把这 100 条成功轨迹存成中间数据或直接封装为 `LeRobotDataset`。

---

### 第四步：构建 LeRobotDataset 与微调 SmolVLA

1. **整理为 LeRobot 规范格式**：
   * 使用 LeRobot 提供的官方 API 创建 Dataset：
     * 特征字典（Features）绑定：相机输入键名（如 `observation.images.top`、`observation.images.wrist`）、状态键名（`observation.state`）和动作（`action`）。
   * 调用 LeRobot 的内置统计方法计算训练集的 **均值与方差（Mean & Std）**，生成 `stats.json`，用于训练和推理时的动作归一化。
2. **启动 SmolVLA 监督微调**：
   * 加载基底权重：`lerobot/smolvla_base`。
   * 在 RTX 4090 上，可以设置 `batch_size = 8` 或 `16`。
   * 微调约 20,000 ~ 30,000 步。
   * 观察 Loss 曲线：训练中后期 Loss 会平稳下降并趋于收敛。保存最佳 Checkpoint。

---

### 第五步：闭环推理、50 种子评估与产出视频

训练完后，大模型需要在真实仿真闭环中“考试”：

1. **推理循环（Action Chunking）**：
   * 模型每次推理会预测出未来连续多步动作（例如长度为 $N=16$ 的动作块）。
   * 推荐策略：采用滚动窗口执行前 $k=8$ 步后，采集新一帧图像并重新推理，保证闭环反馈能力。
   * **动作反归一化**：模型输出的是归一化动作，必须用训练集的统计量（`mean/std`）还原为真实环境的控制量。
2. **跑 50 个独立测试集评估**：
   * 使用 `seed = 1000 ~ 1049`（确保这些方块随机位置在训练集中从未见过）。
   * 运行测试脚本，统计：
     * **成功率（Success Rate）**：成功 episode 占比。
     * **平均耗时（Average Time to Success）**。
     * **掉落/碰撞率**。
3. **输出可视化交付物**：
   * 利用 `imageio-ffmpeg` 将每个测试回合保存为 `.mp4` 视频文件。
   * 挑出 2~3 个具有代表性的成功抓取视频，以及 1~2 个边缘失败视频（用于分析边界工况和时序漂移问题），作为汇报的核心素材。

---

## 模块三：新手核心避坑清单（务必留心）

1. **EGL 环境变量**：
   云端没有物理屏幕，运行任何仿真渲染脚本前必须确认 `os.environ["MUJOCO_GL"] = "egl"` 或终端已 `export MUJOCO_GL=egl`。
2. **夹爪开合极性统一**：
   数据采集时若是 `1.0=开，0.0=关`，必须保证模型训练和闭环测试时**完全同一标准**，很多失败是因为夹爪开闭逻辑反了导致一抓就松。
3. **动作空间（Action Space）选择**：
   建议控制末端位姿增量（$\Delta x, \Delta y, \Delta z, \Delta \text{yaw}, \text{gripper}$），相比控制 7 个关节电机角度（Joint angles），位姿增量控制在 VLA 模型上更容易学出来且泛化性能更好。
  
