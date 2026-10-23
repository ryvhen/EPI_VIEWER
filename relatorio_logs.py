"""
Relatorio gerencial dos eventos do monitor de EPI.

Uso:
    python relatorio_logs.py
    python relatorio_logs.py --de 2026-10-20 --ate 2026-10-27
    python relatorio_logs.py --csv logs/eventos.csv --saida relatorio

Gera:
    relatorio/relatorio.html   relatorio autocontido (graficos embutidos)
    relatorio/infracoes.csv    uma linha por infracao (abre no Excel para tabelas dinamicas)

Rode a partir da mesma pasta onde o epi_monitor.py roda, para os links das
fotos de evidencia funcionarem.
"""
import argparse
import base64
import html
import io
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ENCERRAMENTOS = {"REGULARIZADO", "OPERADOR_AUSENTE", "SESSAO_ENCERRADA"}
DIAS = ["segunda", "terca", "quarta", "quinta", "sexta", "sabado", "domingo"]
COR = "#c0392b"


# ---------------------------------------------------------------- leitura
def carregar(caminho):
    df = pd.read_csv(caminho)
    df["data_hora"] = pd.to_datetime(df["data_hora"])
    return df.sort_values("data_hora").reset_index(drop=True)


def rotulo_falta(oculos, abafador):
    if not oculos and not abafador:
        return "Oculos e abafador"
    return "Oculos" if not oculos else "Abafador"


def montar_infracoes(df):
    """Junta cada INFRACAO com o evento que a encerrou (duracao total do episodio)."""
    itens, aberta = [], None

    def fechar(linha, motivo, duracao):
        nonlocal aberta
        # a INFRACAO e gravada 5 s depois do inicio do episodio
        inicio = aberta["data_hora"] - pd.Timedelta(seconds=float(aberta["duracao_s"]))
        itens.append({
            "inicio": inicio,
            "falta": rotulo_falta(aberta["oculos"], aberta["abafador"]),
            "duracao_s": float(duracao),
            "desfecho": motivo,
            "evidencia": aberta["evidencia"] if isinstance(aberta["evidencia"], str) else "",
        })
        aberta = None

    for _, linha in df.iterrows():
        if linha["evento"] == "INFRACAO":
            if aberta is not None:
                fechar(None, "EM_ABERTO", aberta["duracao_s"])
            aberta = linha
        elif linha["evento"] in ENCERRAMENTOS and aberta is not None:
            fechar(linha, linha["evento"], linha["duracao_s"])
    if aberta is not None:
        fechar(None, "EM_ABERTO", aberta["duracao_s"])
    return pd.DataFrame(itens)


def segundos_monitorados(df):
    """Soma INICIO_SESSAO -> FIM_SESSAO (se o monitor gravou esses eventos)."""
    total, inicio = 0.0, None
    for _, linha in df.iterrows():
        if linha["evento"] == "INICIO_SESSAO":
            inicio = linha["data_hora"]
        elif linha["evento"] == "FIM_SESSAO" and inicio is not None:
            total += (linha["data_hora"] - inicio).total_seconds()
            inicio = None
    return total


# ---------------------------------------------------------------- graficos
def para_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def grafico_por_hora(inf):
    horas = inf["inicio"].dt.hour
    lo, hi = max(0, horas.min() - 1), min(23, horas.max() + 1)
    contagem = horas.value_counts().reindex(range(lo, hi + 1), fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar([f"{h:02d}h" for h in contagem.index], contagem.values, color=COR)
    ax.set_title("Infracoes por hora do dia")
    ax.set_ylabel("Ocorrencias")
    ax.yaxis.get_major_locator().set_params(integer=True)
    return para_base64(fig)


def grafico_por_dia(inf):
    contagem = inf["inicio"].dt.date.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar([d.strftime("%d/%m") for d in contagem.index], contagem.values, color=COR)
    ax.set_title("Infracoes por dia")
    ax.set_ylabel("Ocorrencias")
    ax.yaxis.get_major_locator().set_params(integer=True)
    return para_base64(fig)


def grafico_falta(inf):
    contagem = inf["falta"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ax.barh(contagem.index[::-1], contagem.values[::-1], color=COR)
    ax.set_title("EPI que faltava no momento do registro")
    ax.xaxis.get_major_locator().set_params(integer=True)
    return para_base64(fig)


def grafico_duracao(inf):
    faixas = pd.cut(inf["duracao_s"], [0, 15, 30, 60, 300, float("inf")],
                    labels=["ate 15 s", "15-30 s", "30-60 s", "1-5 min", "mais de 5 min"])
    contagem = faixas.value_counts().reindex(faixas.cat.categories, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 3.0))
    ax.bar(contagem.index.astype(str), contagem.values, color=COR)
    ax.set_title("Duracao das infracoes")
    ax.yaxis.get_major_locator().set_params(integer=True)
    return para_base64(fig)


# ---------------------------------------------------------------- texto
def fmt_duracao(s):
    s = int(round(s))
    return f"{s // 60}min {s % 60:02d}s" if s >= 60 else f"{s}s"


def resumo(inf, seg_monitorados):
    hora_pico = inf["inicio"].dt.hour.value_counts().idxmax()
    n_pico = int((inf["inicio"].dt.hour == hora_pico).sum())
    dia_pico = DIAS[inf["inicio"].dt.dayofweek.value_counts().idxmax()]
    falta_top = inf["falta"].value_counts().idxmax()
    linhas = [
        ("Periodo", f"{inf['inicio'].min():%d/%m/%Y} a {inf['inicio'].max():%d/%m/%Y}"),
        ("Total de infracoes (5 s ou mais sem EPI completo)", str(len(inf))),
        ("Duracao media / maxima", f"{fmt_duracao(inf['duracao_s'].mean())} / {fmt_duracao(inf['duracao_s'].max())}"),
        ("Tempo total sem EPI", fmt_duracao(inf["duracao_s"].sum())),
        ("Horario com mais ocorrencias", f"{hora_pico:02d}h as {hora_pico + 1:02d}h ({n_pico} ocorrencias)"),
        ("Dia da semana com mais ocorrencias", dia_pico),
        ("EPI mais ausente", falta_top),
    ]
    if seg_monitorados > 0:
        horas = seg_monitorados / 3600
        linhas.append(("Tempo monitorado", f"{horas:.1f} h"))
        linhas.append(("Infracoes por hora monitorada", f"{len(inf) / horas:.2f}"))
    return linhas


def linha_tabela(r, pasta_saida):
    ev = ""
    if r["evidencia"]:
        rel = os.path.relpath(os.path.abspath(r["evidencia"]), os.path.abspath(pasta_saida))
        ev = f'<a href="{html.escape(rel.replace(os.sep, "/"))}">foto</a>'
    return (f"<tr><td>{r['inicio']:%d/%m/%Y}</td><td>{r['inicio']:%H:%M:%S}</td>"
            f"<td>{r['falta']}</td><td>{fmt_duracao(r['duracao_s'])}</td>"
            f"<td>{r['desfecho'].replace('_', ' ').lower()}</td><td>{ev}</td></tr>")


CSS = """
body{font-family:Segoe UI,Arial,sans-serif;max-width:900px;margin:30px auto;padding:0 16px;color:#222}
h1{margin-bottom:4px}h2{margin-top:32px;border-bottom:2px solid #eee;padding-bottom:4px}
table{border-collapse:collapse;width:100%;font-size:14px}
td,th{border:1px solid #ddd;padding:6px 8px;text-align:left}th{background:#f4f4f4}
.kv td:first-child{width:55%;background:#fafafa}
img{max-width:100%}.nota{color:#666;font-size:13px}
"""


def gerar_html(inf, linhas, imagens, pasta_saida):
    kv = "".join(f"<tr><td>{k}</td><td><b>{v}</b></td></tr>" for k, v in linhas)
    figs = "".join(f'<p><img src="data:image/png;base64,{b}"></p>' for b in imagens)
    tabela = "".join(linha_tabela(r, pasta_saida) for _, r in inf.sort_values("inicio").iterrows())
    return f"""<!DOCTYPE html><html lang="pt-BR"><head><meta charset="utf-8">
<title>Relatorio de EPI</title><style>{CSS}</style></head><body>
<h1>Relatorio de conformidade de EPI</h1>
<p class="nota">Gerado em {pd.Timestamp.now():%d/%m/%Y %H:%M}. Infracao = operador no posto sem os dois EPIs por 5 s ou mais.</p>
<h2>Resumo</h2><table class="kv">{kv}</table>
<h2>Graficos</h2>{figs}
<h2>Ocorrencias</h2>
<table><tr><th>Data</th><th>Inicio</th><th>EPI ausente</th><th>Duracao</th><th>Desfecho</th><th>Evidencia</th></tr>{tabela}</table>
<p class="nota">As fotos de evidencia contem imagens de pessoas: acesso restrito a quem precisa (LGPD).</p>
</body></html>"""


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Relatorio gerencial do monitor de EPI")
    ap.add_argument("--csv", default=os.path.join("logs", "eventos.csv"))
    ap.add_argument("--saida", default="relatorio")
    ap.add_argument("--de", help="data inicial AAAA-MM-DD")
    ap.add_argument("--ate", help="data final AAAA-MM-DD (inclusive)")
    args = ap.parse_args()

    df = carregar(args.csv)
    inf = montar_infracoes(df)
    if inf.empty:
        print("Nenhuma infracao registrada.")
        return

    if args.de:
        inf = inf[inf["inicio"] >= pd.Timestamp(args.de)]
        df = df[df["data_hora"] >= pd.Timestamp(args.de)]
    if args.ate:
        fim = pd.Timestamp(args.ate) + pd.Timedelta(days=1)
        inf = inf[inf["inicio"] < fim]
        df = df[df["data_hora"] < fim]
    if inf.empty:
        print("Nenhuma infracao no periodo informado.")
        return

    os.makedirs(args.saida, exist_ok=True)
    imagens = [grafico_por_hora(inf), grafico_por_dia(inf), grafico_falta(inf), grafico_duracao(inf)]
    pagina = gerar_html(inf, resumo(inf, segundos_monitorados(df)), imagens, args.saida)

    with open(os.path.join(args.saida, "relatorio.html"), "w", encoding="utf-8") as f:
        f.write(pagina)
    inf.to_csv(os.path.join(args.saida, "infracoes.csv"), index=False, encoding="utf-8-sig")
    print(f"Relatorio gerado em {args.saida}{os.sep}relatorio.html ({len(inf)} infracoes)")


if __name__ == "__main__":
    main()
