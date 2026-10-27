# Monitor de EPI - NR-06

Sistema de visão computacional que verifica, em tempo real, se o operador de um posto de trabalho fixo está usando os EPIs obrigatórios (óculos de proteção e protetor auricular), usando uma webcam comum.

## Como funciona

1. **MediaPipe (BlazeFace)** localiza o rosto do operador e seus pontos de referência (olhos e orelhas).
2. **YOLOv8** detecta `Glasses` e `Earmuffs` no frame.
3. **Validação pelo rosto:** cada detecção só é aceita se estiver na posição anatômica correta (óculos sobre os olhos, abafador junto às orelhas), o que reduz falsos positivos.
4. **Votação em janela deslizante** suaviza a decisão entre frames.
5. **Os dois EPIs são exigidos** para liberar a máquina.
6. **Registro de auditoria:** se o operador ficar 5 segundos sem EPI completo, o evento é gravado em CSV junto com uma foto de evidência.

## Estrutura

| Arquivo | Função |
|---|---|
| `epi_monitor.py` | Aplicação principal (detecção, regras e registro) |
| `capturar_dataset.py` | Captura de imagens para o dataset próprio |
| `preanotar.py` | Gera pré-anotações (formato YOLO) com o modelo atual, para revisão manual |
| `relatorio_logs.py` | Lê `logs/eventos.csv` e gera relatório HTML com gráficos e `infracoes.csv` para gestores |
| `avaliar_sistema.py` | Avalia o sistema em fotos de teste: precisão, recall e erros de decisão por limiar, com e sem validação pelo rosto |
| `best.pt` | Pesos do modelo YOLO treinado |
| `blaze_face_short_range.tflite` | Modelo de detecção facial do MediaPipe |

## Instalação

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

O arquivo `blaze_face_short_range.tflite` pode ser baixado em:
https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite

## Uso

```bash
python epi_monitor.py
```

Teclas: `q` encerra, `d` liga/desliga o modo debug.

Os parâmetros (thresholds, janela de votação, tempo de registro, geometria de validação) ficam no bloco de configuração no topo de `epi_monitor.py`. Se trocar o modelo, ajuste também `MODELO_YOLO` e `IMGSZ` (use o mesmo tamanho do treino).

## Saídas

- `logs/eventos.csv`: data/hora, evento (`INFRACAO`, `REGULARIZADO`, `OPERADOR_AUSENTE`), estado de cada EPI e duração.
- `logs/evidencias/`: fotos das infrações registradas.

## Dataset

As imagens de pessoas não são versionadas (veja `.gitignore`). Para montar o dataset próprio:

1. `python capturar_dataset.py` para capturar fotos no posto.
2. `python preanotar.py` (opcional) para gerar anotações iniciais.
3. Revisar e anotar no Roboflow com as classes `Glasses` e `Earmuffs`.
4. Treinar com YOLOv8 (Colab) e substituir o `best.pt`.

## Tecnologias

Python, OpenCV, Ultralytics YOLOv8, MediaPipe.

## Contexto

Trabalho de Conclusão de Curso - Engenharia da Computação, UNISAL.
