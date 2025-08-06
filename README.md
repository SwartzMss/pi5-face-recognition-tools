# pi5-face-recognition-tools

面向树莓派 Pi5 与原生 CSI 摄像头的人脸识别示例，使用 `dlib` 与 `face_recognition` 作为核心依赖。

## 硬件需求

- Raspberry Pi 5
- 树莓派官方 CSI 摄像头
- microSD 卡（建议 16GB 以上）

## 软件环境

- 64 位 Raspberry Pi OS（Bookworm 或更新版本）
- Python 3.11 及以上

## 环境准备

1. 更新系统：

   ```bash
   sudo apt update && sudo apt upgrade
   ```

2. 启用摄像头并配置 overlay：

   ```bash
   sudo raspi-config  # Interface Options -> Camera -> Enable
   # 如使用 CSI 摄像头 3，请在 /boot/firmware/config.txt 中确认已添加
   # dtoverlay=imx477（或相应的摄像头 overlay）。
   ```

3. 安装基础构建工具与依赖：

   ```bash
   sudo apt install -y git cmake build-essential python3-dev python3-pip \
       libopenblas-dev liblapack-dev python3-opencv libcamera-apps
   ```

4. 创建并激活 Python 虚拟环境：

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

5. 安装人脸识别库及常见图像处理依赖：

   ```bash
   pip install dlib face_recognition numpy opencv-python
   ```

6. 测试摄像头是否工作：

   ```bash
   libcamera-still -o test.jpg
   ```

## 性能优化与加速方案

对于实时人脸识别，树莓派的计算资源较为紧张，可通过以下方式提升运行速度：

1. **从源码编译并启用硬件指令集**：

   使用 NEON 与 OpenBLAS 优化 `dlib` 的矩阵运算，并开启多线程支持。

   ```bash
   # 获取源码
   git clone https://github.com/davisking/dlib.git
   cd dlib

   # 生成并编译（禁用 CUDA，启用 NEON/BLAS/LAPACK/OpenMP）
   mkdir build && cd build
   cmake .. \
       -DDLIB_USE_CUDA=OFF \
       -DUSE_NEON_INSTRUCTIONS=ON \
       -DDLIB_USE_BLAS=ON -DDLIB_USE_LAPACK=ON \
       -DUSE_OPENMP=ON
   cmake --build . --config Release -- -j"$(nproc)"
   sudo make install && sudo ldconfig

   # 安装 face_recognition（利用已安装的加速版 dlib）
   pip install face_recognition
   ```

2. **降低输入分辨率**：

   在不影响识别精度的前提下，将摄像头帧缩小到 `640x480` 或更低，可显著减轻处理负担。

3. **裁剪 ROI（感兴趣区域）**：

   若已知人脸大致出现位置，可先截取该区域再进行识别，减少多余计算。

4. **批量处理**：

   将多帧图像组合成批次处理，可更好地利用 `dlib` 的内部并行能力。

## 下一步

完成以上准备后，可在本仓库中添加或运行人脸识别示例代码。

