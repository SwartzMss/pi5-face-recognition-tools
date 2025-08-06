# Pi5 人脸识别工具

在 Raspberry Pi 5 上进行人脸检测与识别的工具和示例结构说明。本仓库提供：环境搭建指南、性能优化思路，以及后续添加的采集、训练、识别脚本占位目录。

---

## 📋 项目概述

`pi5-face-recognition-tools` 旨在利用 Raspberry Pi 5 的 CPU 指令集优化，实现实时人脸检测与识别。虽然示例脚本（如 `capture.py`、`train.py`、`recognize.py`）将于后续版本添加，本 README 主要覆盖：

- 系统与 Python 环境准备

- dlib 源码编译及 NEON/OpenMP 加速

- 性能调优要点

---

## ⚙️ 功能特性

- **dlib 源码编译**：启用 ARM NEON 与 OpenMP 支持

---

## 📦 前置条件

-

  **依赖清单**：相机接口与图像处理库

- **Shell 命令**：一步步搭建环境**硬件场景限制**

  - 摄像头：仅支持 CSI 官方摄像头模块（如 Raspberry Pi Camera Module v3）

- **操作系统**\*\*

  - Raspberry Pi OS（64 位，Bullseye 或更新版本）
  - 使用 CSI 摄像头，请在 `/boot/firmware/config.txt` 中启用 `libcamera` 及对应 `dtoverlay`

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

---

## ⚡ 性能优化建议

- **分辨率缩放**：缩小帧到 160×120 做检测，再裁剪原图做编码。
- **OpenMP 并行**：确保 `dlib` 编译时启用 OpenMP，可用所有 CPU 核心。
- **BLAS 加速**：`libopenblas-dev` 提供更快的线性代数运算。
- **异步流水线**：使用多线程分离采集、检测、识别。

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

