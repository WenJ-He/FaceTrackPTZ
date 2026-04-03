# FaceTrack PTZ - 项目架构说明

## 一句话概括

读取全景视频流 -> 检测人脸 -> 按位置排序 -> 逐个控制 PTZ 摄像头变焦放大 -> 多阶段识别 -> 输出身份和相似度。

---

## 目录结构

```
FaceTrack PTZ/
|
+-- main.py                 # 主入口：多阶段 PTZ 识别流程
+-- config.yaml             # 运行时配置
+-- requirements.txt        # Python 依赖
+-- docker-compose.yml      # Triton 推理服务器（可选）
|
+-- src/                    # 核心模块
|   +-- models.py           # 数据模型
|   +-- config.py           # 配置加载
|   +-- logger.py           # 结构化 JSON 日志
|   +-- state_machine.py    # 状态机
|   +-- scanner.py          # 人脸排序 + 扫描光标
|   +-- detector.py         # 人脸检测 + NMS 去重
|   +-- recognizer.py       # 人脸识别
|   +-- vector_db.py        # 人脸向量库
|   +-- ptz_controller.py   # PTZ 摄像头控制
|   +-- video.py            # RTSP 视频读取
|
+-- models/                 # ONNX 模型
|   +-- facedect/1/         # YOLO 人脸检测（640x640 -> [1,5,8400]）
|   +-- facerecognize/1/    # ArcFace 人脸识别（112x112 -> 512维）
|
+-- data/
|   +-- photo/              # 人脸库照片（20人）
|   +-- face_db.sqlite      # 向量库
|
+-- scripts/                # 测试脚本
|   +-- init_vector_db.py   # 初始化向量库
|   +-- test_recognizer.py  # 测试识别模型
|   +-- test_recognition_pipeline.py  # 端到端联调
|   +-- test_visualize.py   # 可视化标注
|   +-- test_grid_layout.py # 3x3 网格测试
|   +-- run_all.sh          # 一键回归测试
|
+-- outputs/                # 测试输出图片
    +-- result.jpg
    +-- grid_with_recognition.jpg
```

---

## 核心流程

**Step 1 输入** - 读取图片或从 RTSP 视频流抓帧

**Step 2 检测** - YOLO 模型检测人脸，输出 bbox + score。NMS 去重（IOU > 0.5 的重叠框只保留最高置信度的那个）。

**Step 3 排序** - scanner 按 (cy // row_bucket, cx) 排序。效果：从上到下，每行内从左到右。扫描光标跟踪位置，不后退。

**Step 4 多阶段识别循环** - 对每个 target：

- Stage 0（全景）：直接用检测 bbox 裁剪人脸，recognizer 提取 512 维 embedding，vector_db 做 TopK 匹配。Stage 0 的 Top1 作为"固定身份"(fixed_identity)。
- Stage 1..N（PTZ 变焦）：ptz_controller.calculate_coordinates(bbox, stage_num) 计算变焦坐标（像素 -> 0-255），模拟 PTZ 移动，重新提取 embedding + TopK 匹配。记录 delta_from_s0 和 delta_from_prev。
- 提前停止：sim >= 0.6 或 gain < 0.02。

**Step 5 输出** - 每个 target 每个阶段的 identity + similarity + delta。汇总表。

---

## 各模块详解

### src/models.py - 数据模型

- **BBox(x1, y1, x2, y2)** - 矩形框，像素坐标。属性：cx, cy, width, height
- **Detection(bbox, score)** - 单次检测结果，可选 landmark
- **RecognitionResult(identity, similarity, top_k, vector, fixed_identity_sim)** - 单次识别结果
- **StageRecord(stage_num, timestamp, bbox, recognition, delta_s0, delta_prev)** - 一个阶段的记录
- **TargetRecord(target_id, sort_index, start_time, stages, ...)** - 一个 target 完整记录
  - add_stage() 自动计算 delta、跟踪 fixed_identity
  - finalize() 设置 final_identity/similarity
- **ScanCursor(row_bucket, x_center)** - 扫描光标，is_before() 判断人脸是否在光标之后

### src/config.py - 配置加载

- Config(data) - 从字典创建，自动填充默认值 + 校验必填项
- load_config(path) - 从 YAML 文件加载
- config.get("scan.row_bucket", 80) - 点分路径访问，支持默认值
- 必填：video, device, triton, vector_db

### src/logger.py - 结构化日志

- setup_logging(level) 初始化
- log(event, result, **kwargs) 输出 JSON 格式
- 包含：timestamp, level, state, target_id, stage, event, result

### src/state_machine.py - 状态机

状态：INIT, CONNECTING_STREAM, DETECTING, SORTING, MOVING, RECOGNIZING, HOLDING, NEXT_TARGET, ERROR, RECOVER

转换链：INIT -> CONNECTING_STREAM -> DETECTING -> SORTING -> MOVING -> RECOGNIZING -> MOVING(继续变焦) 或 HOLDING(完成) -> NEXT_TARGET -> DETECTING(下一轮)

每次 transition() 校验合法性，非法转换记录日志。

### src/scanner.py - 排序和扫描光标

- sort_faces(detections) 按 (cy // row_bucket, cx) 排序
- select_next(sorted_faces) 返回光标之后的下一个 target，光标前进
- reset_round() 重置光标到左上角
- 规则：光标只能前进（右和下），不后退

### src/detector.py - 人脸检测

- **Detector 类**：通过 Triton gRPC 调用 YOLO 检测模型
  - 输入：images [1,3,640,640]
  - 输出：output0 [1,5,8400]，转置后解析为 bbox
- **nms() 函数**：独立的 NMS 实现，按置信度降序贪心去重
- 本地模式用 onnxruntime 直接推理

### src/recognizer.py - 人脸识别

- 通过 Triton gRPC 调用 ArcFace 识别模型
- 输入：input.1 [1,3,112,112]
- 输出：1333 [1,512]（512 维 embedding）
- 本地模式用 onnxruntime 直接推理

### src/vector_db.py - 向量库

- 存储：SQLite（向量以 BLOB 存储）
- 搜索：内存暴力余弦相似度（open 时全量加载到内存）
- search(query_vector, top_k) 返回 List[(name, similarity)]

### src/ptz_controller.py - PTZ 控制

- calculate_coordinates(bbox, stage_num)：像素坐标 -> 0-255 相对坐标
  - effective_percent = min(box_percent * (1 + 0.3 * stage_num), 1.0)
  - stage 0: 50%, stage 1: 65%, stage 2: 80%, stage 3: 95%, stage 4+: 100%
- 控制模式：isapi（HTTP PUT）或 subprocess（外部二进制）
- move_to_target() 发送命令（含重试），wait_stable() 等待稳定

### src/video.py - 视频读取

- OpenCV RTSP 读取，自动重连
- 返回视频帧供检测使用

---

## 测试脚本

| 脚本 | 做什么 | 验证什么 |
|---|---|---|
| init_vector_db.py | photo/ -> embedding -> SQLite | 向量库初始化 |
| test_recognizer.py | 单张图 -> 512 维向量 | 识别模型可用性 |
| test_recognition_pipeline.py | 检测 -> 识别 -> TopK | 端到端识别正确性 |
| test_visualize.py | 拼接图 -> 标注结果图 | 可视化验证（中文姓名、排序） |
| test_grid_layout.py | 3x3 网格排序+识别 | 多人脸排序和识别准确性 |
| run_all.sh | 一键回归测试 | 整体回归 |

---

## 关键配置参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| scan.row_bucket | 80 | 排序行高像素 |
| scan.max_zoom_stages | 5 | 最大变焦阶段 |
| recognition.recognition_accept_threshold | 0.6 | 识别接受阈值 |
| recognition.min_similarity_gain | 0.02 | 最小增益 |
| recognition.top_k | 5 | TopK 候选数 |
| detection.score_threshold | 0.5 | 检测置信度阈值 |
| detection.nms_iou_threshold | 0.5 | NMS IOU 阈值 |
| device.box_percent | 0.5 | 人脸视口占比基准 |

---

## 开发状态

| 能力 | 状态 |
|---|---|
| 人脸检测（ONNX） | 已完成 |
| NMS 去重 | 已完成 |
| 人脸识别（ONNX） | 已完成 |
| 向量库 | 已完成 |
| 扫描排序 | 已完成 |
| 多阶段 PTZ 循环 | 已完成 |
| 状态机 | 已完成 |
| PTZ 坐标计算 | 已完成 |
| PTZ 实际控制 | 代码完成，待接设备 |
| 视频流读取 | 代码完成，待接设备 |
| 可视化测试 | 已完成 |
| 报告生成 | 待开发 |
| 网页叠加层 | 待开发 |
