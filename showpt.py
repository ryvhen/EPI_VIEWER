import torch
from ultralytics import YOLO

for nome in ['best.pt', 'bestbugado.pt']:
    print("=" * 40, nome)
    m = YOLO(nome)
    print("Classes:", m.names)

    ckpt = torch.load(nome, map_location='cpu', weights_only=False)
    args = ckpt.get('train_args', {})
    print("Dataset (data):", args.get('data'))
    print("Modelo base:", args.get('model'))
    print("Epochs:", args.get('epochs'), "| imgsz:", args.get('imgsz'))
    print("Data de treino:", ckpt.get('date'))