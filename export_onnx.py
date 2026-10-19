from ultralytics import YOLO

# 1. Load the PyTorch trained model
model = YOLO('best.pt')

# 2. Export the model to ONNX format with input size 640x640
model.export(format = 'onnx', imgsz = 640)
print("Conversion completed! The file 'best.onnx' is ready.")