# Pi5 人脸识别工具

基于 Raspberry Pi 5 的实时人脸检测与识别系统，**完全使用 rpicam 原生命令**，避免所有 OpenCV 后端兼容性问题。本仓库提供完整的解决方案：从数据采集到实时识别，专为树莓派5优化。

> 🚨 **重要**：本项目已完全移除对 libcamera/V4L2/OpenCV VideoCapture 的依赖，使用纯 rpicam 方案，确保在树莓派5上100%兼容。

---

## 📋 项目概述

`pi5-face-recognition-tools` 专为 Raspberry Pi 5 设计，实现了完整的人脸识别工作流：

### 🎯 核心功能
- **数据采集** (`capture.py`)：使用 rpicam-still 进行高质量人脸图像采集
- **实时识别** (`recognize.py`)：基于 rpicam-still 的实时人脸检测与识别
- **树莓派5优化**：完美适配 Pi5 的新摄像头架构

### ✨ 技术特性
- **纯 rpicam 方案**：完全基于树莓派5原生摄像头命令
- **零依赖冲突**：不依赖 OpenCV VideoCapture、libcamera 或 V4L2
- **实时预览**：拍照时提供持续预览功能
- **多线程流**：后台帧捕获，8FPS 实时人脸识别
- **完美兼容**：专为树莓派5设计，避免所有后端问题

---

## ⚙️ 系统要求

- **硬件**：Raspberry Pi 5
- **摄像头**：CSI 摄像头或 USB 摄像头
- **操作系统**：Raspberry Pi OS (64位推荐)
- **Python**：3.8+

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
  libhdf5-dev python3-venv libcamera-apps rpicam-apps
```

**重要**：确保安装了 `rpicam-apps`，这是树莓派5摄像头的核心组件。

### 3. 编译并安装 dlib（同时启用 NEON 指令集与 OpenMP 并行）
- #### 说明：NEON 与 OpenMP 可以同时启用。NEON 提供 SIMD 向量化加速，而 OpenMP 利用多核并行，共同提升计算性能。

```bash
# 克隆 dlib 源码
git clone https://github.com/davisking/dlib.git
cd dlib
mkdir build && cd build
cmake .. \
  -DDLIB_USE_CUDA=OFF \
  -DCMAKE_BUILD_TYPE=RELEASE \
  -DCMAKE_CXX_FLAGS="-march=armv8-a+simd -O3 -fopenmp" \
  -DCMAKE_EXE_LINKER_FLAGS="-fopenmp"
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

## 📸 数据采集

### 🎥 使用 capture.py 采集人脸数据

`capture.py` 使用树莓派5的 `rpicam-still` 命令进行高质量图像采集，具有以下特性：

- **实时预览**：持续显示摄像头预览窗口
- **即拍即得**：在终端按回车键立即拍照
- **自动管理**：预览进程自动启停，避免资源冲突
- **优化参数**：针对Pi5调优的摄像头设置

#### 使用方法：

```bash
python capture.py
```

#### 操作流程：

1. **设置拍摄数量**：输入每人拍摄的照片数量（默认3张）
2. **输入姓名**：为要采集的人员输入姓名
3. **预览启动**：系统自动启动摄像头预览窗口
4. **拍照操作**：在终端按回车键拍照，预览窗口显示实时画面
5. **继续采集**：可以继续为其他人员采集数据

#### 数据存储结构：
```
dataset/
├── 张三/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── 3.jpg
├── 李四/
│   ├── 1.jpg
│   └── 2.jpg
└── ...
```

---

## 🧪 实时人脸识别

### 🎯 使用 recognize.py 进行实时识别

`recognize.py` 提供完整的实时人脸识别功能，专为树莓派5优化：

#### ✨ 主要特性：

- **纯 rpicam 方案**：使用 rpicam-still 多线程帧捕获
- **可视化界面**：绿框显示已知人员，红框显示未知人员  
- **性能监控**：实时显示 FPS 和检测到的人脸数量
- **自动记录**：未知人员自动保存图片和视频
- **零兼容问题**：完全避免 OpenCV 后端冲突

#### 使用方法：

```bash
python recognize.py
```

#### 系统界面说明：

- **绿色边框**：已识别的已知人员
- **红色边框**：未识别的陌生人员
- **状态信息**：显示实时FPS和检测人脸数量
- **按 'q' 退出**：安全退出识别系统

#### 自动记录功能：

- **未知人脸图片**：保存到 `unknown_images/` 目录
- **未知人脸视频**：保存到 `unknown_videos/` 目录  
- **冷却机制**：10秒内避免重复记录同一未知人员
- **文件命名**：使用时间戳确保文件唯一性

#### 数据集要求：

确保 `dataset/` 目录结构如下：
```
dataset/
├── 张三/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── 3.jpg
├── 李四/
│   ├── 1.jpg
│   └── 2.jpg
└── ...
```

---

## 🏗️ 技术架构

### 摄像头方案对比

| 方案 | 兼容性 | 性能 | 稳定性 | 本项目使用 |
|------|--------|------|--------|------------|
| **OpenCV + libcamera** | ❌ 经常失败 | 🟡 中等 | ❌ 不稳定 | ❌ 已移除 |
| **OpenCV + V4L2** | ❌ Pi5不支持 | 🟡 中等 | ❌ 不兼容 | ❌ 已移除 |
| **纯 rpicam 命令** | ✅ 完美支持 | ✅ 高效 | ✅ 稳定 | ✅ **当前方案** |

### 核心设计

- **capture.py**：使用 `rpicam-still` + 实时预览窗口
- **recognize.py**：使用 `RPiCameraStream` 类 + `rpicam-still` 多线程帧捕获
- **接口兼容**：完全模拟 OpenCV VideoCapture 接口，无需修改识别算法

---

## 🚀 快速开始

### 完整使用流程：

1. **环境准备**
   ```bash
   # 更新系统
   sudo apt update && sudo apt upgrade -y
   
   # 安装依赖
   sudo apt install -y rpicam-apps libcamera-apps
   
   # 测试摄像头
   rpicam-hello --timeout 5000
   ```

2. **安装项目**
   ```bash
   git clone https://github.com/SwartzMss/pi5-face-recognition-tools.git
   cd pi5-face-recognition-tools
   
   # 创建虚拟环境
   python3 -m venv venv
   source venv/bin/activate
   
   # 安装依赖
   pip install face_recognition opencv-python numpy
   ```

3. **数据采集**
   ```bash
   python capture.py
   # 为每个人员采集3-5张不同角度的照片
   ```

4. **开始识别**
   ```bash
   python recognize.py
   # 系统将自动加载训练数据并开始实时识别
   ```

---

## 🔧 故障排除

### 摄像头相关问题：

**问题：摄像头无法打开**
```bash
# 检查摄像头连接
rpicam-hello --timeout 3000

# 如果 rpicam-hello 工作，但 recognize.py 不工作：
# 检查 rpicam-apps 安装
sudo apt install rpicam-apps

# 检查权限
sudo usermod -a -G video $USER
# 重新登录后生效
```

**注意**：本项目已完全移除对 `/dev/video*` 设备和 V4L2 的依赖。如果 `rpicam-hello` 能正常工作，`recognize.py` 就应该能工作。

**问题：图像质量差或颜色异常**
- capture.py 中已优化参数，适合大多数环境
- 如需调整，可修改摄像头参数：亮度、对比度、白平衡等

**问题：性能较慢**
```bash
# 确保启用所有CPU核心
echo "检查CPU核心数："
nproc

# 监控系统资源
htop
```

### 识别精度问题：

- **采集更多样本**：每人至少3-5张不同角度、光线的照片
- **提高图像质量**：确保人脸清晰、光线充足
- **调整识别阈值**：在 recognize.py 中修改 `TOLERANCE` 值

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

