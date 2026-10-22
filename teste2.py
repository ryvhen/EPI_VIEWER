import cv2
import numpy as np
import urllib.request
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

# ==========================================
# 1. CARREGAMENTO DOS MODELOS
# ==========================================
try:
    model = YOLO('best.pt')
except Exception as e:
    print("Erro ao carregar o modelo YOLO. Certifica-te de que o arquivo 'best.pt' esta na mesma pasta.")
    exit()

model_path = 'blaze_face_short_range.tflite'
if not os.path.exists(model_path):
    print("A descarregar o modelo de deteção facial do MediaPipe...")
    url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    urllib.request.urlretrieve(url, model_path)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
detector = vision.FaceDetector.create_from_options(options)

# ==========================================
# 2. INICIALIZAÇÃO DA WEBCAM
# ==========================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frames_sem_epi = 0
LIMITE_ALERTA = 15

print("Webcam aberta com sucesso! Pressiona 'q' para encerrar.")

# ==========================================
# 3. LOOP PRINCIPAL COM VALIDAÇÃO ANATÓMICA
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detection_result = detector.detect(mp_image)

    oculos_detectado = False
    abafador_detectado = False
    rosto_encontrado = False

    if detection_result.detections:
        rosto_encontrado = True
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            xmin = max(0, int(bbox.origin_x))
            ymin = max(0, int(bbox.origin_y))
            box_w = int(bbox.width)
            box_h = int(bbox.height)

            # Margens equilibradas: laterais para orelhas, topo para cabeça
            pad_x = int(box_w * 0.25)
            pad_y_top = int(box_h * 0.20)
            pad_y_bottom = int(box_h * 0.05)

            xmin_crop = max(0, xmin - pad_x)
            ymin_crop = max(0, ymin - pad_y_top)
            xmax_crop = min(w, xmin + box_w + pad_x)
            ymax_crop = min(h, ymin + box_h + pad_y_bottom)

            face_roi = frame[ymin_crop:ymax_crop, xmin_crop:xmax_crop]

            if face_roi.size > 0:
                roi_h, roi_w, _ = face_roi.shape

                # Inferência com threshold seguro em 0.35
                yolo_results = model(face_roi, conf=0.35, imgsz=320, verbose=False)

                for r in yolo_results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        cls_name = model.names[cls_id].lower()
                        conf = float(box.conf[0])

                        bx1, by1, bx2, by2 = map(int, box.xyxy[0])

                        # Centro relativo do objeto na ROI (de 0.0 a 1.0)
                        center_x_rel = ((bx1 + bx2) / 2.0) / roi_w
                        center_y_rel = ((by1 + by2) / 2.0) / roi_h

                        # Validação para Óculos: deve estar na metade superior da ROI (olhos)
                        is_glasses = any(k in cls_name for k in ['glass', 'oculos', 'glasses', '0'])
                        if is_glasses and (center_y_rel < 0.65):
                            oculos_detectado = True
                            abs_x1, abs_y1 = xmin_crop + bx1, ymin_crop + by1
                            abs_x2, abs_y2 = xmin_crop + bx2, ymin_crop + by2

                            cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"Oculos ({conf:.2f})", (abs_x1, max(15, abs_y1 - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Validação para Abafador: deve estar nas orelhas (laterais + metade superior)
                        is_earmuff = any(k in cls_name for k in ['earmuff', 'abafador', 'protetor', 'ear', '1'])
                        if is_earmuff and (center_y_rel < 0.70) and (center_x_rel < 0.35 or center_x_rel > 0.65):
                            abafador_detectado = True
                            abs_x1, abs_y1 = xmin_crop + bx1, ymin_crop + by1
                            abs_x2, abs_y2 = xmin_crop + bx2, ymin_crop + by2

                            cv2.rectangle(frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 255, 255), 2)
                            cv2.putText(frame, f"Abafador ({conf:.2f})", (abs_x1, max(15, abs_y1 - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # Caixa da ROI (MediaPipe)
                cv2.rectangle(frame, (xmin_crop, ymin_crop), (xmax_crop, ymax_crop), (255, 180, 0), 2)
                cv2.putText(frame, "ROI Operador (MediaPipe)", (xmin_crop, max(20, ymin_crop - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 2)

    # ==========================================
    # 4. PAINEL DE AUDITORIA E BLOQUEIO
    # ==========================================
    if rosto_encontrado:
        # Se precisares de exigir ambos os EPIs, usa: if not (oculos_detectado and abafador_detectado):
        if not oculos_detectado:
            frames_sem_epi += 1
        else:
            frames_sem_epi = 0

        if frames_sem_epi >= LIMITE_ALERTA:
            cv2.rectangle(frame, (0, 0), (w, 45), (0, 0, 220), -1)
            cv2.putText(frame, "SISTEMA BLOQUEADO - EPI INCOMPLETO", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.rectangle(frame, (0, 0), (w, 45), (0, 180, 0), -1)
            cv2.putText(frame, "MAQUINA LIBERADA (NR-06 CONFORME)", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.rectangle(frame, (0, 0), (w, 45), (50, 50, 50), -1)
        cv2.putText(frame, "AGUARDANDO OPERADOR NO POSTO...", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Auditoria NR-06 - Sistema Hibrido", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()