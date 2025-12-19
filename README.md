# ⚡ Object Detection on Snapdragon X Elite (NPU)

Run **YOLOv8** object detection at **50+ FPS** on the Windows ARM Hexagon NPU using ONNX Runtime QNN Execution Provider.

![NPU Ready](https://img.shields.io/badge/NPU-Ready-green) ![Platform](https://img.shields.io/badge/Platform-Snapdragon_X_Elite-blue) ![Model](https://img.shields.io/badge/Model-YOLOv8_INT8-orange)

## 📋 Prerequisites

- **Device**: Surface Laptop 7 / Surface Pro 11 / Any generic **Snapdragon X Elite** device.
- **OS**: Windows 11 on Arm.
- **Drivers**: Ensure you have the latest Qualcomm NPU drivers (Windows Update usually handles this).
- **Python**: Python 3.10+ (ARM64 Native).

## 🌟 For Beginners (The Easiest Way)

If you are new to command line:
1.  **Clone/Download** this repository.
2.  Double-click **`run.bat`**.
    - It will automatically install the necessary libraries (`onnxruntime-qnn`, etc.).
    - It will launch the NPU performance visualizer.
3.  Open **Task Manager** (`Ctrl+Shift+Esc`), go to **Performance** -> **NPU** to see the magic!

## 🚀 Manual Quick Start (For Developers)

1. **Install Dependencies**
   ```powershell
   pip install -r requirements.txt
   ```
   *Note: If `onnxruntime-qnn` fails to install or load, ensure you don't have conflicting `onnxruntime` packages installed.*

2. **Run the NPU Performance Test**
   The script runs a sustained inference visualizer so you can check Task Manager.
   ```powershell
   python npu_realtime_inference.py --duration 30
   ```
   **Expected Output:**
   ```text
   [SUCCESS] QNN HTP Backend Initialized! NPU is READY.
   [RUNNING] Time: 10.0s | Frames: 512 | FPS: 51.20
   ```
   Open **Task Manager** -> **Performance** -> **NPU** to see the compute graph active!

## 🛠️ How It Works

This project performs the following pipeline to unlock NPU performance:

1.  **Export**: YOLOv8 PyTorch model -> ONNX (Float32).
2.  **Quantize**: ONNX -> **INT8 QDQ ONNX** (using `quantize_yolov8.py` with COCO128 calibration data).
3.  **Inference**: Uses `onnxruntime-qnn` with the `QnnHtp.dll` backend for hardware acceleration.
    *   **CPU**: ~2 FPS (Slow)
    *   **NPU**: ~50-65 FPS (Real-time)

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `npu_realtime_inference.py` | Visual performance script for NPU usage verification |
| `technical_benchmark.py` | Professional benchmark for latency/throughput measurement |
| `quantize_yolov8.py` | Script to re-generate the quantized model locally |
| `download_assets.py` | Downloads COCO128 calibration dataset |
| `yolov8s_qdq.onnx` | Pre-quantized INT8 model (Ready to run) |
| `data_config.yaml` | Dataset configuration for YOLO validation |

## 🐛 Troubleshooting

**"QNN execution provider not found"**
- Ensure `onnxruntime-qnn` is the **only** installed ORT package.
- `pip uninstall onnxruntime onnxruntime-gpu`
- `pip install --force-reinstall onnxruntime-qnn`

**Low FPS (~2-3 FPS)**
- This means the NPU failed to initialize and it fell back to CPU. Check the console output for warnings.
- Ensure `QnnHtp.dll` can be found (the scripts handle this, but system PATH issues can occur).


https://github.com/user-attachments/assets/b14b2a68-00e0-45aa-a753-e13b5d37b89f


