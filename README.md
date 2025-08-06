# Pi5 人脸识别工具

在 Raspberry Pi 5 上进行人脸检测与识别的工具和示例结构说明。本仓库提供：环境搭建指南、性能优化思路，以及后续添加的采集、训练、识别脚本占位目录。

---

## 📋 项目概述

`pi5-face-recognition-tools` 旨在利用 Raspberry Pi 5 的 CPU 指令集优化，实现实时人脸检测与识别。仓库已包含一个基础的 `recognize.py` 实时识别脚本，后续将补充 `capture.py`、`train.py` 等，README 主要覆盖：

- 系统与 Python 环境准备

- dlib 源码编译及 NEON/OpenMP 加速

- 性能调优要点

---

## ⚙️ 功能特性

- **dlib 源码编译**：启用 ARM NEON 与 OpenMP 支持

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

### 3. 编译并安装 dlib（同时启用 NEON 指令集与 OpenMP 并行）
- #### 说明：NEON 与 OpenMP 可以同时启用。NEON 提供 SIMD 向量化加速，而 OpenMP 利用多核并行，共同提升计算性能。

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

---

## ⚡ 性能优化建议

- **分辨率缩放**：缩小帧到 160×120 做检测，再裁剪原图做编码。
- **OpenMP 并行**：确保 `dlib` 编译时启用 OpenMP，可用所有 CPU 核心。
- **BLAS 加速**：`libopenblas-dev` 提供更快的线性代数运算。
- **异步流水线**：使用多线程分离采集、检测、识别。

---

## 🧪 示例脚本：实时识别

在 `dataset/` 目录下为每个人创建一个子目录，目录名作为姓名标签，
目录中可以放置多张该人员的照片，例如：

```
dataset/
├── Alice/
│   ├── 1.jpg
│   └── 2.jpg
└── Bob/
    ├── 1.jpg
    └── 2.jpg
```

准备好数据集后运行：

```bash
python recognize.py
```

- 若检测到未在库中的陌生人，会将当前帧保存至 `unknown_images/` 并录制数秒短视频到 `unknown_videos/`。为避免重复保存，脚本设置了 `UNKNOWN_SAVE_COOLDOWN`（默认 10 秒）作为冷却时间。
- 若识别到已知人员，会在终端提示并触发蜂鸣，不保存视频。

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

