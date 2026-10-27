"""
Avaliacao do sistema de EPI sobre um conjunto de fotos de teste.

Usa exatamente a mesma validacao do epi_monitor.py (MediaPipe + YOLO + filtro
pelo rosto) e compara COM e SEM a validacao pelo rosto, para varios limiares de
confianca. O YOLO roda uma vez por foto; os limiares sao aplicados depois.

Estrutura esperada da pasta de teste (o nome da subpasta define o gabarito):

    dataset_teste/
        com_oculos_e_abafador/   -> oculos=SIM, abafador=SIM
        so_oculos/               -> oculos=SIM, abafador=NAO
        so_abafador/             -> oculos=NAO, abafador=SIM
        sem_epi/                 -> oculos=NAO, abafador=NAO
        negativos_dificeis/      -> oculos=NAO, abafador=NAO (oculos comum, fone, etc.)

IMPORTANTE: as fotos de teste nao podem ter sido usadas no treino (idealmente,
pessoas que nao aparecem no treino). Senao as metricas ficam infladas.

Uso:
    python avaliar_sistema.py
    python avaliar_sistema.py --pasta dataset_teste --modelo best.pt --imgsz 640 --conf 0.35
    python avaliar_sistema.py --salvar-erros

Saidas (pasta avaliacao/):
    metricas_por_limiar.csv   precisao/recall/F1 por EPI e erros de decisao, com e sem validacao
    por_cenario.csv           acerto por cenario no limiar de referencia
    erros.csv                 fotos em que o sistema errou no limiar de referencia
    grafico_limiares.png      curvas por limiar (com x sem validacao pelo rosto)
    erros/                    (com --salvar-erros) fotos dos erros com as caixas desenhadas
"""
import argparse
import math
import os
import time

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
import pandas as pd
from ultralytics import YOLO

import epi_monitor as epi

# gabarito por nome de pasta: (oculos, abafador)
CENARIOS = {
    "com_oculos_e_abafador": (True, True),
    "so_oculos": (True, False),
    "so_abafador": (False, True),
    "sem_epi": (False, False),
    "negativos_dificeis": (False, False),
}
EXTENSOES = (".jpg", ".jpeg", ".png")
LIMIARES = [round(0.20 + 0.05 * i, 2) for i in range(13)]  # 0.20 ... 0.80


# =====================================================================
# LEITURA DAS FOTOS
# =====================================================================
def listar_imagens(pasta):
    itens = []
    if not os.path.isdir(pasta):
        return itens
    for nome in sorted(os.listdir(pasta)):
        sub = os.path.join(pasta, nome)
        if not os.path.isdir(sub):
            continue
        if nome not in CENARIOS:
            print(f"[aviso] subpasta ignorada (nome fora do padrao): {nome}")
            continue
        for arq in sorted(os.listdir(sub)):
            if arq.lower().endswith(EXTENSOES):
                itens.append((os.path.join(sub, arq), nome, CENARIOS[nome]))
    return itens


# =====================================================================
# INFERENCIA (uma vez por foto, com o menor limiar)
# =====================================================================
def inferir(img, modelo, detector, imgsz, conf_min):
    t0 = time.perf_counter()
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    achados = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)).detections

    face = None
    if achados:
        maior = max(achados, key=lambda d: d.bounding_box.width * d.bounding_box.height)
        bb = maior.bounding_box
        pontos = [(k.x * w, k.y * h) for k in (maior.keypoints or [])]
        if len(pontos) >= 6:  # mesmo criterio do monitor
            face = {
                "rosto": (bb.origin_x, bb.origin_y, bb.width, bb.height),
                "olhos": pontos[0:2],
                "orelhas": pontos[4:6],
            }

    resultado = modelo(img, conf=conf_min, imgsz=imgsz, verbose=False)[0]
    deteccoes = []
    for box in resultado.boxes:
        nome = modelo.names[int(box.cls[0])]
        if nome in (epi.CLASSE_OCULOS, epi.CLASSE_ABAFADOR):
            deteccoes.append((nome, float(box.conf[0]), tuple(float(v) for v in box.xyxy[0])))

    return face, deteccoes, (time.perf_counter() - t0) * 1000


# =====================================================================
# DECISAO POR LIMIAR
# =====================================================================
def caixas_validadas(face, deteccoes, conf):
    """Aplica o limiar e a validacao pelo rosto. Devolve (nome, conf, caixa, valido, lado)."""
    saida = []
    for nome, c, caixa in deteccoes:
        if c < conf:
            continue
        lado = None
        if face is None:
            valido = False
        elif nome == epi.CLASSE_OCULOS:
            valido = epi.oculos_valido(caixa, face["rosto"], face["olhos"])
        else:
            lado = epi.lado_abafador(caixa, face["rosto"], face["orelhas"])
            valido = lado is not None
        saida.append((nome, c, caixa, valido, lado))
    return saida


def decidir(face, deteccoes, conf, validar):
    """Devolve (oculos_previsto, abafador_previsto)."""
    itens = caixas_validadas(face, deteccoes, conf)
    if validar:
        oculos = any(v and n == epi.CLASSE_OCULOS for n, _, _, v, _ in itens)
        lados = {l for n, _, _, v, l in itens if v and n == epi.CLASSE_ABAFADOR}
        return oculos, len(lados) >= epi.ABAFADOR_LADOS_MIN
    # sem validacao: qualquer caixa do YOLO acima do limiar conta
    return (any(n == epi.CLASSE_OCULOS for n, *_ in itens),
            any(n == epi.CLASSE_ABAFADOR for n, *_ in itens))


# =====================================================================
# METRICAS
# =====================================================================
def razao(a, b):
    return a / b if b else float("nan")


def prf(tp, fp, fn):
    p, r = razao(tp, tp + fp), razao(tp, tp + fn)
    f1 = float("nan") if (math.isnan(p) or math.isnan(r)) else razao(2 * p * r, p + r)
    return p, r, f1


def metricas(amostras, limiar, validar):
    cont = {"oculos": [0, 0, 0, 0], "abafador": [0, 0, 0, 0]}  # tp, fp, fn, tn
    n_conf = n_nao_conf = bloqueio = liberacao = acertos = 0

    for a in amostras:
        o, ab = decidir(a["face"], a["deteccoes"], limiar, validar)
        eo, ea = a["esperado"]
        for chave, esp, prev in (("oculos", eo, o), ("abafador", ea, ab)):
            idx = 0 if (esp and prev) else 1 if prev else 2 if esp else 3
            cont[chave][idx] += 1

        prev_conforme, esp_conforme = (o and ab), (eo and ea)
        if esp_conforme:
            n_conf += 1
            bloqueio += not prev_conforme      # bloqueou quem estava certo
        else:
            n_nao_conf += 1
            liberacao += prev_conforme         # liberou quem estava sem EPI
        acertos += prev_conforme == esp_conforme

    po, ro, fo = prf(cont["oculos"][0], cont["oculos"][1], cont["oculos"][2])
    pa, ra, fa = prf(cont["abafador"][0], cont["abafador"][1], cont["abafador"][2])
    return {
        "validacao": "com" if validar else "sem",
        "limiar": limiar,
        "oculos_precisao": po, "oculos_recall": ro, "oculos_f1": fo,
        "abafador_precisao": pa, "abafador_recall": ra, "abafador_f1": fa,
        "liberacao_indevida": razao(liberacao, n_nao_conf),   # liberou sem EPI completo (grave)
        "bloqueio_indevido": razao(bloqueio, n_conf),         # bloqueou com EPI completo
        "acuracia_decisao": razao(acertos, len(amostras)),
    }


def por_cenario(amostras, limiar):
    linhas = []
    for nome in CENARIOS:
        grupo = [a for a in amostras if a["cenario"] == nome]
        if not grupo:
            continue
        ok_o = ok_a = ok_t = sem_rosto = 0
        for a in grupo:
            o, ab = decidir(a["face"], a["deteccoes"], limiar, True)
            eo, ea = a["esperado"]
            ok_o += o == eo
            ok_a += ab == ea
            ok_t += (o == eo) and (ab == ea)
            sem_rosto += a["face"] is None
        n = len(grupo)
        linhas.append({
            "cenario": nome, "n_fotos": n,
            "acerto_oculos": ok_o / n, "acerto_abafador": ok_a / n,
            "acerto_total": ok_t / n, "rosto_nao_detectado": sem_rosto / n,
        })
    return pd.DataFrame(linhas)


def listar_erros(amostras, limiar):
    erros = []
    for a in amostras:
        o, ab = decidir(a["face"], a["deteccoes"], limiar, True)
        if (o, ab) != a["esperado"]:
            erros.append({
                "arquivo": a["caminho"], "cenario": a["cenario"],
                "esperado_oculos": int(a["esperado"][0]), "esperado_abafador": int(a["esperado"][1]),
                "previsto_oculos": int(o), "previsto_abafador": int(ab),
                "rosto_detectado": int(a["face"] is not None),
            })
    return pd.DataFrame(erros)


# =====================================================================
# SAIDAS
# =====================================================================
def grafico(df, limiar_ref, caminho):
    paineis = [
        ("Oculos", [("oculos_precisao", "precisao", "#2874a6"), ("oculos_recall", "recall", "#c0392b")]),
        ("Abafador", [("abafador_precisao", "precisao", "#2874a6"), ("abafador_recall", "recall", "#c0392b")]),
        ("Decisao do sistema", [("liberacao_indevida", "liberacao indevida", "#8e44ad"),
                                ("bloqueio_indevido", "bloqueio indevido", "#e67e22")]),
    ]
    fig, eixos = plt.subplots(1, 3, figsize=(16, 4.3))
    for ax, (titulo, series) in zip(eixos, paineis):
        for validacao, estilo in (("com", "-"), ("sem", "--")):
            d = df[df["validacao"] == validacao]
            for col, rotulo, cor in series:
                ax.plot(d["limiar"], d[col], estilo, color=cor, marker="o", markersize=3,
                        label=f"{rotulo} ({validacao} validacao)")
        ax.axvline(limiar_ref, color="gray", lw=0.8, ls=":")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("Limiar de confianca")
        ax.set_title(titulo)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(caminho, dpi=130)
    plt.close(fig)


def desenhar_erro(img, face, deteccoes, limiar, caminho):
    quadro = img.copy()
    if face is not None:
        fx, fy, fw, fh = face["rosto"]
        cv2.rectangle(quadro, (int(fx), int(fy)), (int(fx + fw), int(fy + fh)), (255, 160, 0), 1)
    for nome, c, caixa, valido, _ in caixas_validadas(face, deteccoes, limiar):
        x1, y1, x2, y2 = map(int, caixa)
        cor = (0, 200, 0) if valido else (0, 0, 255)
        cv2.rectangle(quadro, (x1, y1), (x2, y2), cor, 2)
        cv2.putText(quadro, f"{nome} {c:.2f}{'' if valido else ' REJ'}", (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)
    cv2.imwrite(caminho, quadro)


def mostrar(df, limiar_ref):
    cols = ["validacao", "limiar", "oculos_precisao", "oculos_recall", "abafador_precisao",
            "abafador_recall", "liberacao_indevida", "bloqueio_indevido", "acuracia_decisao"]
    ref = df[df["limiar"] == limiar_ref][cols]
    print(f"\n=== Resultado no limiar de referencia ({limiar_ref}) ===")
    print(ref.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nliberacao_indevida = fracao de fotos SEM EPI completo que o sistema liberou (erro grave)")
    print("bloqueio_indevido  = fracao de fotos COM EPI completo que o sistema bloqueou")


# =====================================================================
# MAIN
# =====================================================================
def main():
    ap = argparse.ArgumentParser(description="Avaliacao do sistema de EPI")
    ap.add_argument("--pasta", default="dataset_teste")
    ap.add_argument("--modelo", default=epi.MODELO_YOLO)
    ap.add_argument("--imgsz", type=int, default=epi.IMGSZ)
    ap.add_argument("--conf", type=float, default=epi.CONF_YOLO, help="limiar de referencia")
    ap.add_argument("--saida", default="avaliacao")
    ap.add_argument("--salvar-erros", action="store_true")
    args = ap.parse_args()

    imagens = listar_imagens(args.pasta)
    if not imagens:
        print(f"Nenhuma foto encontrada em '{args.pasta}'. Veja a estrutura esperada no topo do arquivo.")
        return

    limiares = sorted(set(LIMIARES + [round(args.conf, 2)]))
    modelo = YOLO(args.modelo)
    detector = epi.criar_detector_rosto()
    os.makedirs(args.saida, exist_ok=True)

    amostras, tempos = [], []
    for i, (caminho, cenario, esperado) in enumerate(imagens, 1):
        img = cv2.imread(caminho)
        if img is None:
            print(f"[aviso] nao consegui ler: {caminho}")
            continue
        face, deteccoes, ms = inferir(img, modelo, detector, args.imgsz, min(limiares))
        tempos.append(ms)
        amostras.append({"caminho": caminho, "cenario": cenario, "esperado": esperado,
                         "face": face, "deteccoes": deteccoes, "img": img if args.salvar_erros else None})
        if i % 25 == 0 or i == len(imagens):
            print(f"  {i}/{len(imagens)} fotos processadas")

    # --- resumo da amostra
    print("\nFotos por cenario:")
    for nome in CENARIOS:
        n = sum(a["cenario"] == nome for a in amostras)
        if n:
            print(f"  {nome:24s} {n}")
    n_conf = sum(a["esperado"] == (True, True) for a in amostras)
    if n_conf == 0 or n_conf == len(amostras):
        print("[aviso] a amostra nao tem fotos dos dois tipos (com e sem EPI completo): "
              "as taxas de decisao ficam indefinidas.")
    if min(sum(a["cenario"] == c for a in amostras) for c in {a["cenario"] for a in amostras}) < 30:
        print("[aviso] algum cenario tem menos de 30 fotos: interprete as taxas com cautela.")

    # --- metricas
    linhas = [metricas(amostras, lim, validar) for validar in (True, False) for lim in limiares]
    df = pd.DataFrame(linhas)
    df.to_csv(os.path.join(args.saida, "metricas_por_limiar.csv"), index=False, encoding="utf-8-sig")

    cen = por_cenario(amostras, round(args.conf, 2))
    cen.to_csv(os.path.join(args.saida, "por_cenario.csv"), index=False, encoding="utf-8-sig")
    print(f"\n=== Acerto por cenario (limiar {args.conf}, com validacao pelo rosto) ===")
    print(cen.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    erros = listar_erros(amostras, round(args.conf, 2))
    erros.to_csv(os.path.join(args.saida, "erros.csv"), index=False, encoding="utf-8-sig")
    print(f"\nErros no limiar de referencia: {len(erros)} de {len(amostras)} fotos")

    if args.salvar_erros and not erros.empty:
        pasta_erros = os.path.join(args.saida, "erros")
        os.makedirs(pasta_erros, exist_ok=True)
        por_caminho = {a["caminho"]: a for a in amostras}
        for _, e in erros.iterrows():
            a = por_caminho[e["arquivo"]]
            nome = f"{a['cenario']}__{os.path.basename(a['caminho'])}"
            desenhar_erro(a["img"], a["face"], a["deteccoes"], args.conf, os.path.join(pasta_erros, nome))
        print(f"Fotos dos erros salvas em {pasta_erros} (contem rostos: nao versionar)")

    mostrar(df, round(args.conf, 2))
    grafico(df, args.conf, os.path.join(args.saida, "grafico_limiares.png"))

    # --- desempenho (a primeira foto inclui aquecimento do modelo, por isso e descartada)
    uteis = tempos[1:] if len(tempos) > 1 else tempos
    ms = sum(uteis) / len(uteis)
    print(f"\nTempo medio por foto (MediaPipe + YOLO): {ms:.0f} ms  (~{1000 / ms:.1f} FPS) "
          f"nesta maquina, imgsz={args.imgsz}")
    print(f"Arquivos gerados em '{args.saida}'.")


if __name__ == "__main__":
    main()
