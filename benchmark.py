import time
import argparse
import numpy as np
import onnxruntime as ort
import cv2
from pathlib import Path

def preprocess(image_path, input_shape):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = input_shape[2], input_shape[3]
    img = cv2.resize(img, (w, h))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img

def benchmark(model_path, data_dir, backend='cpu', num_images=100):
    print(f"Benchmarking {model_path} on {backend}...")
    
    # Provider options
    providers = []
    if backend == 'qnn':
        # QNN EP specific options
        # We need to specify the backend path if it's not in default finding logic
        # For Windows ARM surface, we target HTP.
        qnn_options = {
            'backend_path': 'QnnHtp.dll', 
            # 'profiling_level': 'basic',
            # 'rpc_control_latency': 0 
        }
        providers = [('QNNExecutionProvider', qnn_options), 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        print(f"Failed to create session with {backend}: {e}")
        return

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    # Handle dynamic batch if present, usually [1, 3, 640, 640]
    if isinstance(input_shape[0], str): input_shape[0] = 1

    image_paths = list(Path(data_dir).glob('*.jpg'))[:num_images]
    if not image_paths:
        print("No images found for benchmarking.")
        return

    # Warmup
    print("Warming up...")
    dummy_input = np.zeros(input_shape, dtype=np.float32)
    for _ in range(5):
        session.run(None, {input_name: dummy_input})

    print(f"Running inference on {len(image_paths)} images...")
    start_time = time.time()
    
    for img_path in image_paths:
        img_data = preprocess(img_path, input_shape)
        session.run(None, {input_name: img_data})

    end_time = time.time()
    total_time = end_time - start_time
    avg_latency = (total_time / len(image_paths)) * 1000 # ms
    fps = len(image_paths) / total_time

    print(f"Results for {backend}:")
    print(f"  Total Time: {total_time:.4f}s")
    print(f"  Avg Latency: {avg_latency:.2f}ms")
    print(f"  Throughput: {fps:.2f} FPS")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--data_dir', type=str, default='datasets/coco128/images/train2017')
    parser.add_argument('--backend', type=str, default='qnn', choices=['cpu', 'qnn'])
    args = parser.parse_args()

    benchmark(args.model, args.data_dir, args.backend)
