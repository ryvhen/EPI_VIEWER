import cv2
from ultralytics import YOLO

model = YOLO('best.pt')

# 2. Inicialização da webcam usando os parâmetros para otimização
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("Detecção de EPI Pressione 'q' para encerrar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Roda a inferência e confiança
    results = model(frame, conf=0.25, imgsz=320)
    
    # Flags lógicas para monitoramento simultâneo
    tem_oculos = False
    orelha_exposta = False
    tem_protetor = False

    # Varre as detecções do YOLO no frame
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # Mapeia as 3 classes do nosso trio facial
            if label == 'Glasses':
                tem_oculos = True
            elif label == 'Ear':
                orelha_exposta = True
            elif label == 'Earmuffs':
                tem_protetor = True

    #Se a flag continuar falsa após varrer a tela
    if not tem_oculos:
        cv2.putText(frame, "!!! SEM OCULOS DE PROTECAO !!!", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Oculos: OK", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    #Se achar orelha limpa OU não achar o abafador
    if orelha_exposta or not tem_protetor:
        cv2.putText(frame, "!!! SEM PROTETOR AURICULAR !!!", (30, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Protetor: OK", (30, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Plota as caixas coloridas automáticas do YOLOv8 na tela
    annotated_frame = results[0].plot()

    # Exibe a janela 
    cv2.imshow("Detecção de EPI", annotated_frame)

    # 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()