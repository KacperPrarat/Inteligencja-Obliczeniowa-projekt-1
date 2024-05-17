from ultralytics import YOLO


model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)

model.train(data='config.yaml', epochs=100, imgsz=640)

# 718.03 181.83 2 673.93 251.23 2 700.84 231.42 2 699.88 328.49 2 736.45 271.59 2