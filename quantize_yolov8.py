import argparse
from pathlib import Path
import numpy as np
import cv2
import onnx
from ultralytics import YOLO
from onnxruntime.quantization import (
    quantize_static,
    QuantFormat,
    QuantType,
    CalibrationMethod,
    CalibrationDataReader
)

class YoloDataReader(CalibrationDataReader):
    def __init__(self, image_dir, input_name, input_shape, size=100):
        self.image_dir = Path(image_dir)
        self.input_name = input_name
        self.input_shape = input_shape
        self.image_paths = list(self.image_dir.glob('*.jpg'))[:size]
        self.enum_data = None
        
        print(f"found {len(self.image_paths)} images for calibration")

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(self.image_paths)
        
        try:
            image_path = next(self.enum_data)
            return {self.input_name: self.preprocess(image_path)}
        except StopIteration:
            return None

    def preprocess(self, image_path):
        # Resize and normalize image for YOLOv8
        # YOLOv8 expects RGB, 0-1 float32, usually (1, 3, 640, 640)
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h, w = self.input_shape[2], self.input_shape[3]
        img = cv2.resize(img, (w, h))
        
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1) # HWC to CHW
        img = np.expand_dims(img, axis=0) # Add batch dim
        return img

def export_to_onnx(model_name='yolov8s.pt'):
    print(f"Load {model_name}...")
    model = YOLO(model_name)
    print("Exporting to ONNX...")
    # Export with dynamic axes=False for QNN usually better to have static shapes for simpler quantization,
    # though dynamic is supported. Let's stick to static 640x640 for simplicity.
    success = model.export(format='onnx', opset=17, imgsz=640)
    return success

def quantize_model(onnx_path, calibration_data_dir):
    model = onnx.load(onnx_path)
    # Assume single input
    input_name = model.graph.input[0].name
    # Force shape if dynamic, or read from graph
    # YOLOv8 export usually is float32[1, 3, 640, 640]
    input_shape = (1, 3, 640, 640) 

    output_path = onnx_path.replace('.onnx', '_qdq.onnx')
    
    dr = YoloDataReader(calibration_data_dir, input_name, input_shape, size=128)
    
    print(f"Quantizing {onnx_path} to QDQ...")
    
    # QNN HTP configuration typically works well with QDQ, MinMax, and uint8 or int8.
    # We'll use QUInt8 for activations and defaults for weights.
    
    quantize_static(
        model_input=onnx_path,
        model_output=output_path,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={
            'ActivationSymmetric': False, # uint8 is asymmetric
            'WeightSymmetric': True 
        }
    )
    print(f"Quantized model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov8s.pt', help='YOLO model name')
    parser.add_argument('--data_dir', type=str, default='datasets/coco128/images/train2017', help='Path to calibration images')
    args = parser.parse_args()

    # 1. Export
    if not Path(args.model).name.endswith('.onnx'):
        onnx_file = args.model.replace('.pt', '.onnx')
        if not Path(onnx_file).exists():
           onnx_file = export_to_onnx(args.model)
        else:
           print(f"Found existing {onnx_file}")
    else:
        onnx_file = args.model

    # 2. Quantize
    # Note: 'export_to_onnx' returns the filename usually
    if isinstance(onnx_file, str):
         onnx_path = onnx_file
    else:
         # unexpected return
         onnx_path = 'yolov8s.onnx'

    quantize_model(onnx_path, args.data_dir)
