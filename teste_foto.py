from ultralytics import YOLO
import cv2

# (arquivo do modelo, imgsz usado no treino)
modelos = [
    ('best.pt', 320),
    ('bestbugado.pt', 640),
]

imagens = ['teste.png', 'teste.jpg', 'teste_1.jpg']

for nome_modelo, tamanho in modelos:
    model = YOLO(nome_modelo)
    print("=" * 40, nome_modelo, f"(imgsz={tamanho})")

    for img in imagens:
        results = model(img, conf=0.25, imgsz=tamanho, verbose=False)
        r = results[0]

        print(f"\n{img}: {len(r.boxes)} detecção(ões)")
        for box in r.boxes:
            cls_name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            print(f"  {cls_name} {conf:.2f}  caixa={x2-x1}x{y2-y1}px")

        # Salva a imagem anotada, com o nome do modelo para não sobrescrever
        anotada = r.plot()
        cv2.imwrite(f"resultado_{nome_modelo[:-3]}_{img}", anotada)

print("\nImagens anotadas salvas na pasta atual (resultado_*).")