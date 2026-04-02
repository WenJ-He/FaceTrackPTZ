# FaceTrack PTZ - Scripts

本目录包含测试、初始化、和实用脚本，使用 onnxruntime 直接运行本地 ONNX 模型，无需 Triton。

## 文件说明

| 文件 | 用途 |
|---|---|
| `test_recognizer.py` | 测试识别模型：加载图片 → 检测人脸 → 提取 512 维 embedding |
| `init_vector_db.py` | 初始化人脸库：从 `data/photo/` 提取所有人人脸 embedding 写入 SQLite |
| `test_recognition_pipeline.py` | 完整联调：检测 → 识别 → 向量库 TopK 查询 |
| `test_visualize.py` | 可视化测试：检测 + 识别 + 在图片上标注结果框 |
| `start_triton.sh` | 启动 Triton Server (需要 Docker) |
| `stop_triton.sh` | 停止 Triton Server |
| `run_all.sh` | 一键运行：初始化向量库 + 测试识别器 + 完整联调 |

## 快速开始

```bash
# 一键运行全部测试
bash scripts/run_all.sh

# 或逐步运行
python3 scripts/test_recognizer.py
python3 scripts/init_vector_db.py
python3 scripts/test_recognition_pipeline.py

# Triton Docker (需要先安装 Docker)
bash scripts/start_triton.sh
```

## 可视化测试 (test_visualize.py)

完整流程：检测 → 排序 → 识别 → 在图片上绘制标注框和身份信息。

### 运行方式

```bash
# 方式1：自动从 data/photo/ 随机选取多张照片拼接成全景图，再检测识别
python3 scripts/test_visualize.py

# 方式2：指定一张图片
python3 scripts/test_visualize.py path/to/image.jpg
```

### 输出

- 结果图片保存到 `outputs/result.jpg`
- 每个检测到的人脸标注格式：`序号 姓名 相似度`，例如 `1 张三 0.87`
- 人脸框按 scanner 排序顺序编号（从左到右、从上到下）

### 前置条件

```bash
# 先初始化向量库（只需运行一次）
python3 scripts/init_vector_db.py
```

### 输出路径

| 输出 | 路径 |
|---|---|
| 可视化结果图片 | `outputs/result.jpg` |
