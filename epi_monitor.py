"""
Monitor de EPI (NR-06) - posto fixo com webcam

Fluxo por frame:
  1. MediaPipe BlazeFace acha o rosto (maior rosto = operador) e os keypoints (olhos e orelhas)
  2. YOLO detecta Glasses / Earmuffs no frame inteiro
  3. Cada deteccao do YOLO so vale se estiver na posicao anatomica correta do rosto
  4. Janela deslizante com votacao suaviza a decisao (evita piscar)
  5. Os DOIS EPIs sao exigidos; sem EPI por 5 s -> registra evento (CSV + foto)

Teclas: 'q' sai | 'd' liga/desliga modo debug
"""
import csv
import os
import time
from collections import deque
from datetime import datetime

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

# =====================================================================
# CONFIGURACAO (ajuste aqui, nao no meio do codigo)
# =====================================================================
MODELO_YOLO = "best.pt"
IMGSZ = 320                  # mesmo valor usado no treino (best.pt = 320; modelo novo = 640)
CONF_YOLO = 0.35
MODELO_ROSTO = "blaze_face_short_range.tflite"
CAMERA_INDEX = 0

CLASSE_OCULOS = "Glasses"
CLASSE_ABAFADOR = "Earmuffs"

# Janela deslizante com votacao
JANELA_FRAMES = 15           # quantos frames recentes entram na votacao
VOTOS_MINIMOS = 0.60         # fracao de frames com o EPI para considerar presente
MIN_FRAMES_DECISAO = JANELA_FRAMES // 2

# Registro de evento
SEGUNDOS_ATE_REGISTRAR = 5.0
TOLERANCIA_SEM_ROSTO = 1.0   # segundos sem rosto antes de encerrar o episodio
PASTA_LOGS = "logs"
PASTA_EVIDENCIAS = os.path.join(PASTA_LOGS, "evidencias")
ARQUIVO_CSV = os.path.join(PASTA_LOGS, "eventos.csv")

# Validacao geometrica (fracoes da caixa do rosto)
OCULOS_DESVIO_X_MAX = 0.30       # centro dos oculos vs ponto medio dos olhos (largura do rosto)
OCULOS_DESVIO_Y_MAX = 0.20       # idem na vertical (altura do rosto)
OCULOS_LARGURA = (0.60, 1.50)    # largura da caixa / largura do rosto
OCULOS_ALTURA_MAX = 0.40         # rejeita caixa que cobre o rosto inteiro
ABAFADOR_DIST_MAX = 0.45         # centro da caixa ate a orelha (em larguras do rosto)
ABAFADOR_LADOS_MIN = 1           # 1 = basta um lado visivel; 2 = exige os dois

# =====================================================================
# CORES (BGR) E TEXTOS - ASCII apenas, o putText do OpenCV nao desenha acentos
# =====================================================================
COR_ESTADO = {
    "AGUARDANDO": (80, 80, 80),
    "ANALISANDO": (150, 100, 0),
    "CONFORME": (0, 160, 0),
    "ATENCAO": (0, 140, 255),
    "BLOQUEADO": (0, 0, 200),
}
VERDE, VERMELHO, AZUL, BRANCO = (0, 220, 0), (0, 0, 255), (255, 160, 0), (255, 255, 255)


# =====================================================================
# VALIDACAO PELO ROSTO
# =====================================================================
def oculos_valido(caixa, rosto, olhos):
    """Caixa de oculos deve estar sobre os olhos e ter formato de oculos (nao de rosto)."""
    x1, y1, x2, y2 = caixa
    fx, fy, fw, fh = rosto
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = x2 - x1, y2 - y1
    mx = (olhos[0][0] + olhos[1][0]) / 2
    my = (olhos[0][1] + olhos[1][1]) / 2
    return (
        abs(cx - mx) <= OCULOS_DESVIO_X_MAX * fw
        and abs(cy - my) <= OCULOS_DESVIO_Y_MAX * fh
        and OCULOS_LARGURA[0] * fw <= bw <= OCULOS_LARGURA[1] * fw
        and bh <= OCULOS_ALTURA_MAX * fh
    )


def lado_abafador(caixa, rosto, orelhas):
    """Devolve o indice da orelha mais proxima se a caixa estiver perto dela, senao None."""
    x1, y1, x2, y2 = caixa
    fw = rosto[2]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    dists = [((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5 for ox, oy in orelhas]
    i = min(range(len(dists)), key=lambda k: dists[k])
    return i if dists[i] <= ABAFADOR_DIST_MAX * fw else None


# =====================================================================
# ANALISE DE UM FRAME
# =====================================================================
def criar_detector_rosto():
    opcoes = vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODELO_ROSTO),
        min_detection_confidence=0.5,
    )
    return vision.FaceDetector.create_from_options(opcoes)


def analisar(frame, modelo, detector):
    """Retorna None se nao ha rosto; senao um dict com o resultado do frame."""
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    deteccoes = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)).detections
    if not deteccoes:
        return None

    # operador = maior rosto do quadro (o mais proximo da camera)
    face = max(deteccoes, key=lambda d: d.bounding_box.width * d.bounding_box.height)
    bb = face.bounding_box
    rosto = (bb.origin_x, bb.origin_y, bb.width, bb.height)
    pontos = [(k.x * w, k.y * h) for k in (face.keypoints or [])]
    if len(pontos) < 6:
        return None
    # BlazeFace: 0-1 olhos, 2 nariz, 3 boca, 4-5 orelhas
    olhos, orelhas = pontos[0:2], pontos[4:6]

    resultado = modelo(frame, conf=CONF_YOLO, imgsz=IMGSZ, verbose=False)[0]
    oculos_ok = False
    lados = set()
    validas, rejeitadas = [], []

    for box in resultado.boxes:
        nome = modelo.names[int(box.cls[0])]
        conf = float(box.conf[0])
        caixa = tuple(float(v) for v in box.xyxy[0])

        if nome == CLASSE_OCULOS:
            ok = oculos_valido(caixa, rosto, olhos)
            oculos_ok = oculos_ok or ok
        elif nome == CLASSE_ABAFADOR:
            lado = lado_abafador(caixa, rosto, orelhas)
            ok = lado is not None
            if ok:
                lados.add(lado)
        else:
            continue  # classe 'Ear' e outras nao entram na decisao

        (validas if ok else rejeitadas).append((nome, conf, caixa))

    return {
        "oculos": oculos_ok,
        "abafador": len(lados) >= ABAFADOR_LADOS_MIN,
        "rosto": rosto,
        "pontos": pontos,
        "validas": validas,
        "rejeitadas": rejeitadas,
    }


# =====================================================================
# VOTACAO (janela deslizante)
# =====================================================================
class Votacao:
    def __init__(self):
        self.janela = deque(maxlen=JANELA_FRAMES)

    def adicionar(self, valor):
        self.janela.append(bool(valor))

    def pronta(self):
        return len(self.janela) >= MIN_FRAMES_DECISAO

    def presente(self):
        return sum(self.janela) / len(self.janela) >= VOTOS_MINIMOS

    def limpar(self):
        self.janela.clear()


# =====================================================================
# REGISTRO DE EVENTOS (CSV + foto)
# =====================================================================
class Registro:
    CABECALHO = ["data_hora", "evento", "oculos", "abafador", "duracao_s", "evidencia"]

    def __init__(self):
        os.makedirs(PASTA_EVIDENCIAS, exist_ok=True)
        if not os.path.exists(ARQUIVO_CSV):
            with open(ARQUIVO_CSV, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self.CABECALHO)

    def gravar(self, evento, oculos, abafador, duracao, frame=None):
        agora = datetime.now()
        caminho = ""
        if frame is not None:
            caminho = os.path.join(PASTA_EVIDENCIAS, f"{agora:%Y%m%d_%H%M%S}_{evento}.jpg")
            cv2.imwrite(caminho, frame)
        with open(ARQUIVO_CSV, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [agora.isoformat(timespec="seconds"), evento, int(oculos), int(abafador),
                 f"{duracao:.1f}", caminho]
            )
        print(f"[{agora:%H:%M:%S}] {evento} (oculos={oculos}, abafador={abafador}, {duracao:.1f}s)")


# =====================================================================
# DESENHO
# =====================================================================
def desenhar_caixa(frame, caixa, cor, texto, espessura=2):
    x1, y1, x2, y2 = map(int, caixa)
    cv2.rectangle(frame, (x1, y1), (x2, y2), cor, espessura)
    cv2.putText(frame, texto, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)


def desenhar_painel(frame, estado, linha1, linha2):
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, 55), COR_ESTADO[estado], -1)
    cv2.putText(frame, linha1, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, BRANCO, 2)
    cv2.putText(frame, linha2, (12, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BRANCO, 1)


# =====================================================================
# LOOP PRINCIPAL
# =====================================================================
def abrir_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def main():
    modelo = YOLO(MODELO_YOLO)
    detector = criar_detector_rosto()
    registro = Registro()
    cap = abrir_camera()
    if not cap.isOpened():
        print("Nao foi possivel abrir a webcam.")
        return

    vot_oculos, vot_abafador = Votacao(), Votacao()
    estado = "AGUARDANDO"
    linha1, linha2 = "AGUARDANDO OPERADOR NO POSTO...", ""
    t_inicio_sem_epi = None      # quando o operador ficou sem EPI
    registrada = False           # infracao ja registrada neste episodio
    t_ultimo_rosto = None
    ultimo_oculos = ultimo_abafador = False
    debug = True
    t_anterior, fps = time.monotonic(), 0.0

    print("Monitor iniciado. 'q' = sair | 'd' = debug")
    registro.gravar("INICIO_SESSAO", False, False, 0)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        bruto = frame.copy()  # evidencia sem desenhos
        agora = time.monotonic()
        fps = 0.9 * fps + 0.1 * (1.0 / max(agora - t_anterior, 1e-6))
        t_anterior = agora

        analise = analisar(frame, modelo, detector)

        if analise is None:
            # sem rosto: so encerra o episodio depois de uma pequena tolerancia
            if t_ultimo_rosto is None or agora - t_ultimo_rosto > TOLERANCIA_SEM_ROSTO:
                if registrada:
                    registro.gravar("OPERADOR_AUSENTE", ultimo_oculos, ultimo_abafador,
                                    agora - t_inicio_sem_epi)
                t_inicio_sem_epi, registrada = None, False
                vot_oculos.limpar()
                vot_abafador.limpar()
                estado = "AGUARDANDO"
                linha1, linha2 = "AGUARDANDO OPERADOR NO POSTO...", ""
        else:
            t_ultimo_rosto = agora
            vot_oculos.adicionar(analise["oculos"])
            vot_abafador.adicionar(analise["abafador"])

            if not vot_oculos.pronta():
                estado = "ANALISANDO"
                linha1, linha2 = "ANALISANDO...", ""
            else:
                oculos_ok = vot_oculos.presente()
                abafador_ok = vot_abafador.presente()
                ultimo_oculos, ultimo_abafador = oculos_ok, abafador_ok
                detalhe = (f"Oculos: {'OK' if oculos_ok else 'FALTANDO'}  |  "
                           f"Abafador: {'OK' if abafador_ok else 'FALTANDO'}")

                if oculos_ok and abafador_ok:  # os DOIS EPIs sao exigidos
                    if registrada:
                        registro.gravar("REGULARIZADO", oculos_ok, abafador_ok,
                                        agora - t_inicio_sem_epi)
                    t_inicio_sem_epi, registrada = None, False
                    estado = "CONFORME"
                    linha1, linha2 = "MAQUINA LIBERADA (NR-06 CONFORME)", detalhe
                else:
                    if t_inicio_sem_epi is None:
                        t_inicio_sem_epi = agora
                    decorrido = agora - t_inicio_sem_epi
                    if decorrido >= SEGUNDOS_ATE_REGISTRAR and not registrada:
                        registro.gravar("INFRACAO", oculos_ok, abafador_ok, decorrido, bruto)
                        registrada = True
                    if registrada:
                        estado = "BLOQUEADO"
                        linha1 = "SISTEMA BLOQUEADO - EPI INCOMPLETO"
                    else:
                        estado = "ATENCAO"
                        restante = max(0.0, SEGUNDOS_ATE_REGISTRAR - decorrido)
                        linha1 = f"EPI INCOMPLETO - REGISTRO EM {restante:.0f}s"
                    linha2 = detalhe

            # caixas validadas sempre aparecem; o resto so no debug
            for nome, conf, caixa in analise["validas"]:
                cor = VERDE if nome == CLASSE_OCULOS else (0, 255, 255)
                desenhar_caixa(frame, caixa, cor, f"{nome} {conf:.2f}")
            if debug:
                for nome, conf, caixa in analise["rejeitadas"]:
                    desenhar_caixa(frame, caixa, VERMELHO, f"REJ {nome} {conf:.2f}", 1)
                fx, fy, fw, fh = analise["rosto"]
                cv2.rectangle(frame, (int(fx), int(fy)), (int(fx + fw), int(fy + fh)), AZUL, 1)
                for px, py in analise["pontos"]:
                    cv2.circle(frame, (int(px), int(py)), 3, BRANCO, -1)

        desenhar_painel(frame, estado, linha1, linha2)
        if debug:
            cv2.putText(frame, f"FPS {fps:.1f}", (12, frame.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BRANCO, 1)

        cv2.imshow("Auditoria NR-06", frame)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        if tecla == ord("d"):
            debug = not debug

    if registrada:  # encerrou o programa no meio de uma infracao
        registro.gravar("SESSAO_ENCERRADA", ultimo_oculos, ultimo_abafador,
                        time.monotonic() - t_inicio_sem_epi)
    registro.gravar("FIM_SESSAO", False, False, 0)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
