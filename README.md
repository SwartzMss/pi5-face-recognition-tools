# Pi5 人脸识别工具

一套在 Raspberry Pi 5 上进行人脸检测与识别的工具和示例结构说明。本仓库提供：环境搭建指南、性能优化思路，以及后续添加的采集、训练、识别脚本占位目录。

---

## 📋 项目概述

`pi5-face-recognition-tools` 旨在利用 Raspberry Pi 5 的 CPU 指令集优化及可选外部加速器，实现实时人脸检测与识别。虽然示例脚本（如 `capture.py`、`train.py`、`recognize.py`）将于后续版本添加，本 README 主要覆盖：

- 系统与 Python 环境准备
- dlib 源码编译及 NEON/OpenMP 加速
- 建议的目录结构与使用流程
- 性能调优要点

---

## ⚙️ 功能特性

- **dlib 源码编译**：启用 ARM NEON 与 OpenMP 支持
- **依赖清单**：相机接口与图像处理库
- **Shell 命令**：一步步搭建环境
- **加速器集成指南**：Edge TPU、Myriad X 等可选方案
- **目录结构示例**：capture/train/recognize 占位说明

---

## 📦 前置条件

- **硬件**
  - Raspberry Pi 5（4 核 Cortex-A76）
  - 摄像头：CSI 官方摄像头模块或 USB 摄像头
  - 可选：Coral Edge TPU USB 加速器、Intel NCS2、或其他 NPU

- **操作系统**
  - Raspberry Pi OS（64 位，Bullseye 或更新版本）
  - 若使用 CSI 摄像头，请在 `/boot/firmware/config.txt` 中启用 `libcamera` 及对应 `dtoverlay`

---

## 🔧 环境安装

### 1. 系统更新

```bash
sudo apt update && sudo apt upgrade -y
```

### 2. 安装系统依赖

```bash
sudo apt install -y \
  build-essential cmake libopenblas-dev liblapack-dev libjpeg-dev libtiff5-dev \
  libpng-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
  libxvidcore-dev libx264-dev libatlas-base-dev libgtk2.0-dev pkg-config \
  libhdf5-dev python3-venv libcamera-utils
```

### 3. 编译并安装 dlib（启用 NEON/OpenMP）

```bash
# 克隆 dlib 源码
git clone https://github.com/davisking/dlib.git
cd dlib
mkdir build && cd build
cmake .. \
  -DDLIB_USE_CUDA=OFF \
  -DDLIB_USE_OPENMP=ON \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DCMAKE_CXX_FLAGS="-march=armv8-a+simd -O3"
make -j4
sudo make install
cd ../..
```

### 4. 创建并激活 Python 虚拟环境

```bash
python3 -m venv venv
source venv/bin/activate
```

### 5. 安装 Python 库

```bash
pip install --upgrade pip
pip install \
  face_recognition \
  opencv-python \
  numpy
```

### 6. 克隆本仓库

```bash
git clone https://github.com/SwartzMss/pi5-face-recognition-tools.git
cd pi5-face-recognition-tools
```

---

## 🚀 使用流程（占位说明）

1. **采集人脸图片**
   ```bash
   python3 capture.py --output ./dataset/
   ```
   将人脸图片按人名分类保存在 `dataset/` 下。

2. **生成人脸编码**
   ```bash
   python3 train.py --input ./dataset/ --model encodings.pickle
   ```
   将人脸特征保存到 `encodings.pickle`。

3. **实时人脸识别**
   ```bash
   python3 recognize.py --encodings encodings.pickle
   ```
   启动摄像头，实时检测并标注已知人脸。

> **提示**：示例脚本将在后续版本的 `scripts/` 文件夹中提供。

---

## 📁 推荐目录结构

```plaintext
pi5-face-recognition-tools/
├── README.md            # 本说明文档
├── LICENSE              # MIT 协议
├── scripts/             # 占位：Python 示例脚本
│   ├── capture.py       # 采集脚本
│   ├── train.py         # 编码脚本
│   └── recognize.py     # 识别脚本
├── dataset/             # 用户采集的人脸图片
├── models/              # 保存的人脸编码与转换模型
└── utils/               # 辅助模块（相机、预处理等）
```

---

## ⚡ 性能优化建议

- **分辨率缩放**：缩小帧到 160×120 做检测，再裁剪原图做编码。
- **OpenMP 并行**：确保 `dlib` 编译时启用 OpenMP，可用所有 CPU 核心。
- **BLAS 加速**：`libopenblas-dev` 提供更快的线性代数运算。
- **异步流水线**：使用多线程分离采集、检测、识别。
- **外部加速器**：
  - **Coral Edge TPU**：使用 `edgetpu_compiler` 编译 TFLite 模型。
  - **Intel NCS2**：用 OpenVINO Model Optimizer 转为 FP16 IR 模型。

---

## 🤝 贡献指南

欢迎提出 Issue 或 Pull Request：

1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/YourFeature`)
3. 提交修改 (`git commit -m "Add ..."`)
4. 推送分支 (`git push origin feature/YourFeature`)
5. 发起 Pull Request

---

## 📄 协议许可

本项目基于 MIT 协议，详见 [LICENSE](LICENSE) 文件。

