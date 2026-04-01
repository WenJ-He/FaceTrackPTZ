# FaceTrack PTZ - Scripts

本目录包含测试、初始化、和实用脚本，使用 onnxruntime 直接运行本地 ONNX 模型，无需 Triton。

## 文件说明

| 文件 | 用途 |
|---|---|
| `test_recognizer.py` | 测试识别模型：加载图片 → 检测人脸 → 提取 512 维 embedding |
| `init_vector_db.py` | 初始化人脸库：从 `data/photo/` 提取所有人人脸 embedding 写入 SQLite |
| `test_recognition_pipeline.py` | 完整联调：检测 → 识别 → 向量库 TopK 查询 |
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
