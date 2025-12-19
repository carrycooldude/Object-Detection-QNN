# 🧠 The Art of Quantization: Deep Dive for NPU Acceleration

## 1. The Low-Precision Paradigm
Modern neural networks typically train using **32-bit Floating Point (FP32)** numbers. This offers massive dynamic range and precision.
*   **Range**: $\approx 1.18 \times 10^{-38}$ to $3.4 \times 10^{38}$
*   **Cost**: High memory bandwidth, complex silicon logic, high power consumption.

**Snapdragon X Elite's NPU (Hexagon HTP)** is a massive vector processor optimized for **8-bit Integer (INT8)** arithmetic.
*   **Range**: 0 to 255 (UINT8) or -128 to 127 (INT8).
*   **Benefit**: 4x compression, ~30x throughput vs CPU FP32.

The goal of quantization is to map the infinite FP32 real numbers to this tiny finite set of 256 integers without losing the "meaning" of the data (the signal).

---

## 2. The Quantization Math (Affine Transformation)
We typically use **Linear (Affine) Quantization**. This maps a range of real values $[min, max]$ to integer range $[0, 255]$.

The equation is:
$$ x_{real} = Scale \times (x_{quantized} - ZeroPoint) $$

or conversely:
$$ x_{quantized} = clamp(round(\frac{x_{real}}{Scale}) + ZeroPoint) $$

### Key Parameters
1.  **Scale ($S$)**: The step size. It represents how much "real value" one integer step corresponds to.
    $$ Scale = \frac{x_{max} - x_{min}}{q_{max} - q_{min}} $$
2.  **Zero Point ($Z$)**: The integer value that corresponds to the real value $0.0$. This ensures that zero-padding (common in CNNs) remains exactly zero integer, introducing no error during padding.

---

## 3. Calibration: Finding the Range
Because we cannot convert the *entire* universe of FP32 numbers, we must find the **subset** of values actually used by the model. This is called **Calibration**.

We run the model on a representative dataset (e.g., COCO128) and collect statistics for every layer's activiation.

### Methods
*   **MinMax (We used this)**:
    *   Records the absolute min and max seen during inference.
    *   *Pros*: Simple, guarantees no clipping (saturation).
    *   *Cons*: Sensitive to outliers. A single stuck neuron outputting $1000.0$ when everything else is $0-1.0$ will forcing the Scale to be huge, determining precision for the useful data.
*   **Entropy / KL Divergence**:
    *   Minimizes the information loss between the original FP32 distribution and the quantized INT8 histogram.
    *   *Pros*: Often better accuracy for heavy-tail distributions.
    *   *Cons*: Slower to calculate.

For YOLOv8 on QNN HTP, MinMax usually works well because Relu output is bounded and well-behaved.

---

## 4. Graph Mechanics: QDQ
In **ONNX**, quantization is represented explicitly using **QDQ (Quantize-Dequantize)** nodes.

### The "Fake" Graph
Conceptually, the graph looks like this:
1.  `Input (FP32)` -> `QuantizeLinear` -> **`Input (INT8)`**
2.  **`Conv2D (INT8)`** (This is the heavy op)
3.  `Output (INT8)` -> `DequantizeLinear` -> `Output (FP32)`

Even though the inputs and outputs are theoretically "Dequantized" back to Float, the QNN Execution Provider is smart.
*   It sees the `Q -> Conv -> DQ` pattern.
*   It fuses them into a single **NPU Kernel**.
*   It executes the Conv entirely in INT8 on the DSP.
*   It creates a hardware pipeline so data stays on high-speed SRAM (VTCM) in INT8 format as long as possible.

---

## 5. Why QNN HTP?
The Hexagon Tensor Processor (HTP) specifically accelerates **block-based vector math**.
*   **Tensors**: It works on 4D tensors (Batch, Height, Width, Channels).
*   **VTCM**: It has large on-chip Vector Tightly Coupled Memory.
*   **Throughput**: By forcing data to INT8, we can load 4x more data per clock cycle than FP32, effectively quadrupling the memory bandwidth utilization.

### Accuracy Trade-offs
Loss of precision can affect:
*   **Bounding Box Regression**: Small errors in coordinate prediction can lower IOU (Intersection Over Union).
*   **Confidence Scores**: Objectness scores might jitter effectively.
To mitigate this, advanced techniques like **Quantization Aware Training (QAT)** allow the network to "learn" the rounding errors during training, compensating for them.
