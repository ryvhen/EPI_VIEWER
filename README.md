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
| `best.pt` | Pesos do modelo YOLO treinado |
| `blaze_face_short_range.tflite` | Modelo de detecção facial do MediaPipe |

## Instalação

```bash
pip install ultralytics mediapipe opencv-python
```

## Uso

```bash
python epi_monitor.py
```

Teclas: `q` encerra, `d` liga/desliga o modo debug.

Os parâmetros (thresholds, janela de votação, tempo de registro, geometria de validação) ficam no bloco de configuração no topo de `epi_monitor.py`.

## Saídas

- `logs/eventos.csv`: data/hora, evento (`INFRACAO`, `REGULARIZADO`, `OPERADOR_AUSENTE`), estado de cada EPI e duração.
- `logs/evidencias/`: fotos das infrações registradas.

## Tecnologias

Python, OpenCV, Ultralytics YOLOv8, MediaPipe.

## Contexto

Trabalho de Conclusão de Curso - Engenharia da Computação, UNISAL.
