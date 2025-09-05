# app.py — Offline / Estático: Vento & Maré para esportes aquáticos
from __future__ import annotations

import io
import math
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime, timedelta

# =========================================
# CONFIG & ESTILO (sem assets remotos)
# =========================================
st.set_page_config(page_title="Vento & Maré — Watersports", page_icon="🌊", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 0.8rem; padding-bottom: 2rem; }
h1, h2, h3, h4 { line-height: 1.2; }
.dataframe tbody tr th { vertical-align: middle; }
@media (max-width: 640px) { .modebar { display: none !important; } 
</style>
""", unsafe_allow_html=True)

st.title("🌊 Vento & Maré — Watersports (offline)")
st.caption("App estático para exploração de vento e maré, sem chamadas de rede. Ideal para Render gratuito.")

# =========================================
# UTILIDADES / CONVERSÕES
# =========================================
def ms_to_knots(x): return x * 1.9438445
def ms_to_kmh(x):   return x * 3.6
def knots_to_ms(x): return x / 1.9438445

def wrap180(angle):
    """Converte ângulo em graus para faixa [-180, 180]."""
    a = (angle + 180) % 360 - 180
    return a

def cardinal_from_deg(deg):
    """Converte direção em graus para ponto cardeal (16 setores)."""
    dirs = ['N','NNE','NE','ENE','E','ESE','SE','SSE',
            'S','SSW','SW','WSW','W','WNW','NW','NNW']
    ix = int((deg % 360) / 22.5)
    return dirs[ix]

# =========================================
# DADOS: AMOSTRA EMBUTIDA (sem rede)
# =========================================
def build_sample_data(start="2024-02-01 00:00", days=7, freq="H"):
    """
    Gera um dataset sintético coerente com padrões típicos:
    - Maré semi-diurna (~12.42 h) com harmônicos simples
    - Vento com pico diurno e variação direcional suave
    - Temperatura variando ao longo do dia
    """
    idx = pd.date_range(start, periods=int(pd.Timedelta(days=days)/pd.Timedelta(freq))+1, freq=freq, tz=None)
    n = len(idx)
    t = np.arange(n)

    rng = np.random.default_rng(42)

    # Maré (m): baseline 1.2 m, harmônicos, ruído leve
    tide = 1.2 + 0.7*np.sin(2*np.pi*t/12.42) + 0.15*np.sin(2*np.pi*t/6.21 + 0.5)
    tide += rng.normal(0, 0.03, size=n)
    tide = tide.clip(min=0.2)  # evita negativos

    # Derivada ~ variação por hora (m/h)
    tide_rate = np.gradient(tide)

    # Vento (knots) com pico vespertino e leve modulação semidiurna
    hour = np.array([ts.hour for ts in idx])
    diurnal = 10 + 8*np.sin((hour-14)/24*2*np.pi)           # pico ~14h local
    semidiurnal = 2*np.sin(2*np.pi*t/12.42 + 0.8)
    noise = rng.normal(0, 1.5, size=n)
    wind_kt = np.clip(diurnal + semidiurnal + noise, 2, 30)

    gust_extra = np.clip(rng.normal(3.5, 2.0, size=n), 0.5, 12)
    wind_gust_kt = np.clip(wind_kt + gust_extra, wind_kt, wind_kt + 12)

    # Direção (graus): predominante E-SE com oscilação
    wind_dir = (115 + 20*np.sin(2*np.pi*t/24 + 0.3) + rng.normal(0, 10, size=n)) % 360

    # Temperatura do ar (°C)
    air_temp = 22 + 5*np.sin((hour-15)/24*2*np.pi) + rng.normal(0, 0.8, size=n)

    wind = pd.DataFrame({
        "timestamp": idx,
        "wind_speed_ms": knots_to_ms(wind_kt),
        "wind_gust_ms":  knots_to_ms(wind_gust_kt),
        "wind_dir_deg":  wind_dir,
        "air_temp_c":    air_temp
    })

    tide_df = pd.DataFrame({
        "timestamp": idx,
        "tide_height_m": tide,
    })

    return wind, tide_df

# =========================================
# CARREGAMENTO DE DADOS (SEM REDE)
# =========================================
@st.cache_data(show_spinner=False)
def load_wind_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    # Normaliza nomes, exige colunas mínimas
    df.columns = [c.strip().lower() for c in df.columns]
    req = {"timestamp","wind_speed_ms","wind_gust_ms","wind_dir_deg"}
    if not req.issubset(set(df.columns)):
        raise ValueError(f"CSV de vento precisa das colunas: {sorted(req)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    # Opcional: temperatura
    if "air_temp_c" not in df.columns:
        df["air_temp_c"] = np.nan
    return df

@st.cache_data(show_spinner=False)
def load_tide_csv(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = [c.strip().lower() for c in df.columns]
    req = {"timestamp","tide_height_m"}
    if not req.issubset(set(df.columns)):
        raise ValueError(f"CSV de maré precisa das colunas: {sorted(req)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df

def try_local_files() -> tuple[pd.DataFrame|None, pd.DataFrame|None]:
    """Tenta ler data/wind.csv e data/tide.csv se existirem (sem internet)."""
    wind_path = "data/wind.csv"
    tide_path = "data/tide.csv"
    wdf = tdf = None
    try:
        if st.toggle("Usar CSVs locais da pasta data/ (se existirem)", value=False, help="Procura data/wind.csv e data/tide.csv"):
            if os.path.exists(wind_path):
                wdf = pd.read_csv(wind_path)
                wdf.columns = [c.strip().lower() for c in wdf.columns]
                wdf["timestamp"] = pd.to_datetime(wdf["timestamp"], errors="coerce")
                wdf = wdf.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            if os.path.exists(tide_path):
                tdf = pd.read_csv(tide_path)
                tdf.columns = [c.strip().lower() for c in tdf.columns]
                tdf["timestamp"] = pd.to_datetime(tdf["timestamp"], errors="coerce")
                tdf = tdf.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    except Exception:
        pass
    return wdf, tdf

# =========================================
# SIDEBAR — FONTE DE DADOS & CONTROLES
# =========================================
st.sidebar.header("📦 Fonte de dados (offline)")
data_mode = st.sidebar.radio("Escolha a fonte", ["Amostra embutida", "Arquivos (upload)"], index=0)

wind_df = tide_df = None

if data_mode == "Arquivos (upload)":
    uwind = st.sidebar.file_uploader("Wind CSV (timestamp, wind_speed_ms, wind_gust_ms, wind_dir_deg, [air_temp_c])", type=["csv"])
    utide = st.sidebar.file_uploader("Tide CSV (timestamp, tide_height_m)", type=["csv"])
    if uwind and utide:
        try:
            wind_df = load_wind_csv(uwind.getvalue())
            tide_df = load_tide_csv(utide.getvalue())
        except Exception as e:
            st.sidebar.error(f"Erro ao ler CSVs: {e}")
    else:
        # Tenta CSVs locais
        lw, lt = try_local_files()
        if lw is not None and lt is not None:
            wind_df, tide_df = lw, lt
else:
    wind_df, tide_df = build_sample_data()

# Se não carregou nada, avisa e para
if wind_df is None or tide_df is None:
    st.warning("Nenhum dado disponível ainda. Use a **amostra embutida** ou **faça upload** dos CSVs.")
    st.stop()

# =========================================
# PROCESSAMENTO / FEATURE ENGINEERING
# =========================================
# Converte para mesmas frequências e faz asof join por horário
wind_df = wind_df.sort_values("timestamp").reset_index(drop=True)
tide_df = tide_df.sort_values("timestamp").reset_index(drop=True)

# Rolling para suavizar um pouco (opcional)
wind_df["wind_speed_ms"] = wind_df["wind_speed_ms"].astype(float).rolling(2, min_periods=1).mean()
wind_df["wind_gust_ms"]  = wind_df["wind_gust_ms"].astype(float).rolling(2, min_periods=1).max()

# Derivada de maré (m/h) e tendência
tide_df["tide_rate_mph"] = tide_df["tide_height_m"].astype(float).diff()
rate_thr = 0.02  # ~2 cm/h ~ "slack" (quase parada)
tide_df["tide_trend"] = np.where(tide_df["tide_rate_mph"] >  rate_thr, "Subindo",
                          np.where(tide_df["tide_rate_mph"] < -rate_thr, "Descendo", "Quase parada"))

# Merge asof (une valores de vento ao timestamp de maré mais próximo)
df = pd.merge_asof(
    left=tide_df.sort_values("timestamp"),
    right=wind_df.sort_values("timestamp"),
    on="timestamp",
    direction="nearest",
    tolerance=pd.Timedelta("30min")
)

# Conversões & colunas derivadas
df["wind_speed_kt"] = ms_to_knots(df["wind_speed_ms"].astype(float))
df["wind_gust_kt"]  = ms_to_knots(df["wind_gust_ms"].astype(float))
df["gustiness_kt"]  = (df["wind_gust_kt"] - df["wind_speed_kt"]).clip(lower=0)
df["wind_speed_kmh"] = ms_to_kmh(df["wind_speed_ms"].astype(float))

# Direção cardinal
df["wind_dir_cardinal"] = df["wind_dir_deg"].apply(lambda d: cardinal_from_deg(float(d)) if pd.notnull(d) else np.nan)

# Filtros de período
min_ts, max_ts = df["timestamp"].min(), df["timestamp"].max()
st.sidebar.header("🕒 Janela temporal")
start, end = st.sidebar.slider(
    "Selecione o intervalo",
    min_value=min_ts.to_pydatetime(), max_value=max_ts.to_pydatetime(),
    value=(min_ts.to_pydatetime(), max_ts.to_pydatetime()),
    format="DD/MM/YYYY - HH:mm"
)
mask = (df["timestamp"] >= pd.to_datetime(start)) & (df["timestamp"] <= pd.to_datetime(end))
df_view = df.loc[mask].reset_index(drop=True)

# =========================================
# PERFIL DO ESPORTE & ORIENTAÇÃO DA PRAIA
# =========================================
st.sidebar.header("🏄 Configurações do spot")
sport = st.sidebar.selectbox("Perfil do esporte", ["Kitesurf/Windsurf", "Wingfoil", "Surf", "SUP"], index=0)
beach_face_deg = st.sidebar.slider(
    "Direção do mar visto da praia (0°=N, 90°=E, 180°=S, 270°=O)",
    min_value=0, max_value=359, value=90, step=1,
    help="Aponte para onde você olha quando está de frente para o mar."
)

# Classificação Onshore/Side/Offshore (com base no vetor PARA onde o vento sopra)
# Direção meteorológica é "de onde vem". Então para onde vai = dir+180.
df_view["wind_to_deg"] = (df_view["wind_dir_deg"].astype(float) + 180) % 360
delta = wrap180(df_view["wind_to_deg"] - beach_face_deg)
cond = np.where(np.abs(delta) <= 30, "Onshore",
         np.where(np.abs(delta) >= 150, "Offshore", "Side"))
df_view["wind_relative"] = cond

# =========================================
# SCORING simples por perfil (0 a 100)
# =========================================
def score_row(row, sport: str):
    spd = float(row["wind_speed_kt"]) if pd.notnull(row["wind_speed_kt"]) else np.nan
    gust = float(row["gustiness_kt"]) if pd.notnull(row["gustiness_kt"]) else 0.0
    temp = float(row["air_temp_c"]) if pd.notnull(row.get("air_temp_c", np.nan)) else 22.0
    trend = row["tide_trend"]
    rel = row["wind_relative"]

    if np.isnan(spd):
        return np.nan

    score = 0.0

    if sport in ["Kitesurf/Windsurf", "Wingfoil"]:
        # alvo de velocidade
        target = 18 if sport == "Kitesurf/Windsurf" else 14
        width  = 7  if sport == "Kitesurf/Windsurf" else 6
        base = math.exp(-((spd - target)/width)**2)  # 0..1
        # rajadas penalizam
        gust_pen = max(0.0, (gust - 6)/10)  # acima de ~6 kt perde pontos
        # direção relativa
        dir_bonus = {"Onshore": 0.15, "Side": 0.25, "Offshore": -0.6}.get(rel, 0.0)
        # maré: leve bônus quando está "Subindo" (correntes previstas)
        tide_bonus = 0.1 if trend == "Subindo" else (0.05 if trend == "Quase parada" else 0.0)
        # temperatura (conforto)
        temp_bonus = 0.0 if temp >= 18 else -0.2
        score = (0.65*base) + dir_bonus + tide_bonus - 0.2*gust_pen + temp_bonus

    elif sport in ["Surf", "SUP"]:
        # Preferência por vento fraco; pico ideal ~5 kt (SUP) e ~8 kt (Surf)
        target = 5 if sport == "SUP" else 8
        width  = 5
        base = math.exp(-((spd - target)/width)**2)  # mais perto do alvo, melhor
        # direção: offshore leve bônus para surf (segura a onda), mas com cautela
        if sport == "Surf":
            dir_bonus = {"Offshore": 0.25, "Side": 0.05, "Onshore": -0.2}.get(rel, 0.0)
        else:
            dir_bonus = {"Offshore": -0.2, "Side": 0.0, "Onshore": 0.1}.get(rel, 0.0)
        # maré: bônus em meia‑maré a maré cheia (aproximação usando altura > mediana)
        tide_med = df_view["tide_height_m"].median()
        tide_bonus = 0.2 if row["tide_height_m"] >= tide_med else 0.05
        score = (0.75*base) + dir_bonus + tide_bonus

    # Normaliza para 0..100
    return max(0, min(100, round(100*score, 1)))

df_view["score"] = df_view.apply(lambda r: score_row(r, sport), axis=1)

# =========================================
# LAYOUT: TABS
# =========================================
tab_resumo, tab_series, tab_rose, tab_relacao, tab_correl, tab_dados = st.tabs(
    ["Resumo", "Séries", "Rosa do vento", "Vento × Maré", "Correlação/Lag", "Dados"]
)

# =============== RESUMO ===============
with tab_resumo:
    st.subheader("Resumo do período")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Vento médio (kt)", f"{df_view['wind_speed_kt'].mean():.1f}")
    c2.metric("Rajada média (kt)", f"{df_view['wind_gust_kt'].mean():.1f}")
    c3.metric("Temp média (°C)", f"{df_view['air_temp_c'].mean():.1f}")
    c4.metric("Altura maré média (m)", f"{df_view['tide_height_m'].mean():.2f}")
    c5.metric("Score médio", f"{df_view['score'].mean():.1f}")

    st.markdown("### Melhores janelas (top 5 por score)")
    # Encontrar blocos consecutivos bons (score alto)
    thr = st.slider("Limite de score", 0, 100, 60, 1, help="Mínimo para destacar como 'bom'")
    good = df_view[df_view["score"] >= thr].copy()
    if good.empty:
        st.info("Nenhum intervalo acima do limite. Tente reduzir o limiar.")
    else:
        # Agrupa intervalos consecutivos por diferença de timestamp de 1 hora
        good["group"] = (good["timestamp"].diff() > pd.Timedelta("65min")).cumsum()
        agg = good.groupby("group").agg(
            inicio=("timestamp", "first"),
            fim=("timestamp", "last"),
            duracao_h=("timestamp", lambda s: (s.max() - s.min())/pd.Timedelta("1H") + 1),
            score_medio=("score", "mean"),
            vento_med=("wind_speed_kt", "mean"),
            rajada_med=("wind_gust_kt", "mean"),
            direcao_mais=("wind_dir_cardinal", lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan),
            mare_med=("tide_height_m", "mean"),
            tendencia_mais=("tide_trend", lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan),
        ).sort_values("score_medio", ascending=False).head(5).reset_index(drop=True)
        st.dataframe(agg, use_container_width=True)

# =============== SÉRIES ===============
with tab_series:
    st.subheader("Séries temporais (vento, maré, temperatura)")
    # Vento
    fig_w = px.line(df_view, x="timestamp", y=["wind_speed_kt","wind_gust_kt"],
                    labels={"value":"kt", "timestamp":"Tempo"}, title="Vento (kt)")
    fig_w.update_traces(mode="lines+markers")
    fig_w.update_layout(legend_title="", hovermode="x unified")
    st.plotly_chart(fig_w, use_container_width=True)

    # Maré (altura)
    fig_t = px.line(df_view, x="timestamp", y="tide_height_m",
                    labels={"tide_height_m":"m","timestamp":"Tempo"},
                    title="Altura da maré (m)")
    fig_t.update_traces(mode="lines+markers")
    st.plotly_chart(fig_t, use_container_width=True)

    # Taxa de variação da maré
    fig_r = px.bar(df_view, x="timestamp", y="tide_rate_mph",
                   labels={"tide_rate_mph":"m/h","timestamp":"Tempo"},
                   title="Variação da maré (m/h)")
    st.plotly_chart(fig_r, use_container_width=True)

    # Temperatura
    if df_view["air_temp_c"].notna().any():
        fig_temp = px.line(df_view, x="timestamp", y="air_temp_c",
                           labels={"air_temp_c":"°C","timestamp":"Tempo"},
                           title="Temperatura do ar (°C)")
        fig_temp.update_traces(mode="lines+markers")
        st.plotly_chart(fig_temp, use_container_width=True)

# =============== ROSA DO VENTO ===============
with tab_rose:
    st.subheader("Rosa do vento")
    # Bin de direção (16 setores) e velocidade média
    df_rose = df_view.dropna(subset=["wind_dir_deg","wind_speed_kt"]).copy()
    if df_rose.empty:
        st.info("Sem dados de vento para rosa.")
    else:
        bins = np.arange(-11.25, 372, 22.5)  # centraliza nos pontos cardeais
        labels = ['N','NNE','NE','ENE','E','ESE','SE','SSE',
                  'S','SSW','SW','WSW','W','WNW','NW','NNW']
        df_rose["dir_bin"] = pd.cut(df_rose["wind_dir_deg"] % 360, bins=bins, labels=labels, include_lowest=True)
        rose = df_rose.groupby("dir_bin", as_index=False).agg(
            media_kt=("wind_speed_kt", "mean"),
            freq=("wind_speed_kt", "size")
        )
        fig_rose = px.bar_polar(
            rose, r="media_kt", theta="dir_bin",
            title="Rosa do vento — velocidade média por direção (kt)",
            hover_data={"freq":True, "media_kt":":.1f"}
        )
        st.plotly_chart(fig_rose, use_container_width=True)

# =============== RELAÇÃO: VENTO × MARÉ ===============
with tab_relacao:
    st.subheader("Relação entre vento e maré")
    df_sc = df_view.dropna(subset=["wind_speed_kt","tide_height_m"]).copy()
    if df_sc.empty:
        st.info("Sem dados suficientes.")
    else:
        df_sc["Gustiness (kt)"] = df_sc["gustiness_kt"]
        fig_sc = px.scatter(
            df_sc, x="tide_height_m", y="wind_speed_kt",
            color="tide_trend", size="Gustiness (kt)",
            hover_data=["timestamp","wind_dir_cardinal","wind_relative","air_temp_c"],
            labels={"tide_height_m":"Maré (m)", "wind_speed_kt":"Vento (kt)"},
            title="Dispersão: Vento (kt) × Maré (m) — cor por tendência, tamanho por rajada"
        )
        st.plotly_chart(fig_sc, use_container_width=True)

# =============== CORRELAÇÃO / LAG ===============
with tab_correl:
    st.subheader("Correlação com defasagem (lag)")
    # Correlação entre velocidade do vento e taxa de variação da maré (melhor proxy de corrente)
    a = df_view["wind_speed_kt"].astype(float)
    b = df_view["tide_rate_mph"].astype(float)
    lags = range(-6, 7)  # -6h .. +6h
    corr = []
    for L in lags:
        if L < 0:
            c = a.corr(b.shift(-L))
        else:
            c = a.corr(b.shift(L))
        corr.append(c)
    corr_df = pd.DataFrame({"lag_h": list(lags), "correl": corr})
    fig_cc = px.bar(corr_df, x="lag_h", y="correl",
                    labels={"lag_h":"Lag (h)", "correl":"Correlação"},
                    title="Correlação (vento vs variação da maré) por lag")
    st.plotly_chart(fig_cc, use_container_width=True)
    st.caption("Lags positivos: maré antecede o vento; negativos: vento antecede a variação da maré.")

# =============== DADOS / DOWNLOAD ===============
with tab_dados:
    st.subheader("Dados processados")
    st.dataframe(df_view, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Baixar CSV (janela selecionada)",
        df_view.to_csv(index=False).encode("utf-8"),
        file_name="vento_mare_processado.csv",
        mime="text/csv"
    )

# Rodapé
st.markdown("---")
st.caption("""
Este app evita chamadas a APIs para funcionar bem em ambientes de baixo recurso (ex.: Render gratuito). 
Use seus próprios CSVs de **vento** e **maré** (ex.: NOAA Tides & Currents, estações anemométricas locais/DHN) para análises reais.
As recomendações/`score` são heurísticas gerais e **não substituem** avaliação local de segurança e condições.
""")
