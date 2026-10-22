"""
Captura de imagens para o dataset proprio (mesma camera e resolucao do posto).

Teclas:
  1-5     escolhe o cenario
  ESPACO  salva uma foto
  a       liga/desliga captura automatica (1 foto a cada INTERVALO_AUTO s)
  q       sai

As fotos vao para dataset_bruto/<cenario>/<pessoa>_<cenario>_<n>.jpg
O nome da pessoa vai no arquivo para depois separar treino/validacao/teste POR PESSOA.
"""
import os
import time

import cv2

PASTA = "dataset_bruto"
INTERVALO_AUTO = 0.8
CAMERA_INDEX = 0

CENARIOS = {
    ord("1"): "com_oculos_e_abafador",
    ord("2"): "so_oculos",
    ord("3"): "so_abafador",
    ord("4"): "sem_epi",
    ord("5"): "negativos_dificeis",  # oculos comum, fone de ouvido, mao no rosto, bone...
}


def contar(pasta, prefixo):
    if not os.path.isdir(pasta):
        return 0
    return sum(1 for f in os.listdir(pasta) if f.startswith(prefixo))


def main():
    pessoa = input("ID da pessoa (ex: renan, joao): ").strip().lower().replace(" ", "_") or "pessoa"

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cenario = CENARIOS[ord("1")]
    auto, ultimo = False, 0.0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        pasta = os.path.join(PASTA, cenario)
        prefixo = f"{pessoa}_{cenario}_"
        n = contar(pasta, prefixo)

        salvar = False
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        if tecla in CENARIOS:
            cenario = CENARIOS[tecla]
        elif tecla == ord("a"):
            auto = not auto
        elif tecla == 32:
            salvar = True
        if auto and time.monotonic() - ultimo >= INTERVALO_AUTO:
            salvar = True

        if salvar:
            os.makedirs(pasta, exist_ok=True)
            cv2.imwrite(os.path.join(pasta, f"{prefixo}{n:04d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            ultimo = time.monotonic()
            n += 1

        vista = frame.copy()
        cv2.rectangle(vista, (0, 0), (vista.shape[1], 50), (40, 40, 40), -1)
        cv2.putText(vista, f"{cenario}  |  {pessoa}: {n} fotos  |  AUTO {'ON' if auto else 'OFF'}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(vista, "1-5 cenario | ESPACO salva | a auto | q sai",
                    (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.imshow("Captura de dataset", vista)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
