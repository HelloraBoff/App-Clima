# app.py
from __future__ import annotations

import math
import requests
import pandas as pd
import plotly.express as px
import streamlit as st

from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional

# ----------------------------
# CONFIG DA PÁGINA
# ----------------------------
st.set_page_config(
    page_title="Clima Brasil — UX Case",
    page_icon="⛅",
    layout="wide"
)

# ----------------------------
# ESTILOS FINOS DE UX
# ----------------------------
st.markdown("""
<style>
/* Espaçamento e legibilidade */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
h1, h2, h3, h4 { line-height: 1.2; }

/* Métricas mais legíveis */
.big-metric .stMetric-value { font-size: 2.0rem; }
.small-note { color: var(--text-color-secondary, #6b7280); font-size: 0.9rem; }

/* Tabela compacta */
.dataframe tbody tr th { vertical-align: middle; }

/* Quebra de linha nos tooltips do Plotly em telas menores */
@media (max-width: 640px) {
  .modebar { display: none !important; }
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# HEADER COM LOTTIE (opcional)
# ----------------------------
try:
    from streamlit_lottie import st_lottie  # pip install streamlit-lottie
    def load_lottie(url: str) -> Optional[dict]:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None
    with st.container():
        left, right = st.columns([1, 5])
        with left:
            # Use um JSON de animação público. Se offline, apenas ignora.
            anim = load_lottie("https://assets9.lottiefiles.com/packages/lf20_jmBauD.json")
            if anim:
                st_lottie(anim, height=120, key="lottie_header")
        with right:
            st.title("Clima nas Capitais — UX Case")
            st.caption("Previsões horárias e diárias com insights úteis para planejamento no Brasil.")
except Exception:
    # Fallback sem Lottie
    st.title("Clima nas Capitais — UX Case")
    st.caption("Previsões horárias e diárias com insights úteis para planejamento no Brasil.")

# ----------------------------
# LISTA DE CAPITAIS (amostra enxuta e realista)
# ----------------------------
CAPITAIS: Dict[str, Dict[str, float]] = {
    "Brasília": {"lat": -15.7939, "lon": -47.8828},
    "São Paulo": {"lat": -23.5505, "lon": -46.6333},
    "Rio de Janeiro": {"lat": -22.9068, "lon": -43.1729},
    "Belo Horizonte": {"lat": -19.9167, "lon": -43.9345},
    "Curitiba": {"lat": -25.4284, "lon": -49.2733},
    "Fortaleza": {"lat": -3.7172, "lon": -38.5433},
    "Salvador": {"lat": -12.9714, "lon": -38.5014},
    "Manaus": {"lat": -3.1190, "lon": -60.0217},
    "Porto Alegre": {"lat": -30.0331, "lon": -51.2300},
    "Recife": {"lat": -8.0476, "lon": -34.8770},
    "Belém": {"lat": -1.4558, "lon": -48.4902},
}

# ----------------------------
# UTILS
# ----------------------------
@st.cache_data(ttl=600, show_spinner=False)
def geocode(query: str, count: int = 5) -> List[Dict[str, Any]]:
    """Busca coordenadas por nome de cidade via Open-Meteo Geocoding API."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": query, "count": count, "language": "pt", "format": "json"}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json().get("results", [])

@st.cache_data(ttl=600, show_spinner=False)
def fetch_forecast(coords: List[Tuple[float, float]], forecast_days: int, wind_unit: str = "kmh") -> Any:
    """
    Busca previsão horária/diária para uma ou mais coordenadas.
    A Open-Meteo permite múltiplas coordenadas na mesma chamada.
    """
    base = "https://api.open-meteo.com/v1/forecast"
    latitudes = ",".join(str(c[0]) for c in coords)
    longitudes = ",".join(str(c[1]) for c in coords)
    hourly_vars = [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation_probability",
        "precipitation",
        "wind_speed_10m",
        "apparent_temperature",
        "uv_index",
        "weather_code",
    ]
    daily_vars = [
        "temperature_2m_max",
        "temperature_2m_min",
        "uv_index_max",
        "precipitation_sum",
        "precipitation_probability_max",
        "wind_speed_10m_max",
        "sunrise",
        "sunset",
    ]
    params = {
        "latitude": latitudes,
        "longitude": longitudes,
        "hourly": ",".join(hourly_vars),
        "daily": ",".join(daily_vars),
        "forecast_days": max(1, min(int(forecast_days), 16)),
        "timezone": "auto",
        "temperature_unit": "celsius",
        "wind_speed_unit": wind_unit,  # kmh/ms/mph/kn
        "precipitation_unit": "mm",
    }
    r = requests.get(base, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def nearest_hour_index(times: pd.Series, now_local: datetime) -> int:
    """Índice da hora mais próxima a 'agora' (na zona local da cidade)."""
    diffs = (pd.to_datetime(times) - now_local).abs()
    return int(diffs.idxmin())

WMO = {
    0: ("Céu limpo", "☀️"),
    1: ("Poucas nuvens", "🌤️"),
    2: ("Parcialmente nublado", "⛅"),
    3: ("Nublado", "☁️"),
    45: ("Nevoeiro", "🌫️"),
    48: ("Nevoeiro gelado", "🌫️"),
    51: ("Garoa leve", "🌦️"),
    53: ("Garoa moderada", "🌦️"),
    55: ("Garoa forte", "🌧️"),
    61: ("Chuva fraca", "🌧️"),
    63: ("Chuva moderada", "🌧️"),
    65: ("Chuva forte", "🌧️"),
    80: ("Pancadas leves", "🌦️"),
    81: ("Pancadas moderadas", "🌧️"),
    82: ("Pancadas fortes", "⛈️"),
    95: ("Trovoadas", "⛈️"),
    96: ("Trovoadas com granizo", "⛈️"),
    99: ("Trovoadas fortes com granizo", "⛈️"),
}

def dicas_rapidas(temp: float, app_temp: float, prob_chuva: float, precip: float, uv: float, vento: float) -> List[str]:
    tips = []
    if prob_chuva >= 60 or precip >= 1:
        tips.append("🌂 **Leve guarda-chuva** (chance de chuva elevada).")
    if temp >= 32 or app_temp >= 34 or uv >= 7:
        tips.append("🧴 **Protetor solar e hidratação** recomendados.")
    if vento >= 30:
        tips.append("💨 **Rajadas de vento**: cuidado com objetos soltos.")
    if app_temp <= 12:
        tips.append("🧥 **Agasalho** vai bem (sensação térmica baixa).")
    if not tips:
        tips.append("✅ Condições confortáveis na maior parte do dia.")
    return tips

def kpi_delta(curr: float, prev: Optional[float]) -> str:
    if prev is None or (isinstance(prev, float) and math.isnan(prev)):
        return "—"
    delta = curr - prev
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.1f}"

# ----------------------------
# SIDEBAR CONTROLS
# ----------------------------
st.sidebar.header("⚙️ Controles")
modo = st.sidebar.radio("Modo", ["Cidade única", "Comparar capitais"], index=0)
dias = st.sidebar.slider("Dias de previsão", 1, 5, 2, help="Até 16 dias disponíveis na API.")
unidade_vento = st.sidebar.selectbox("Unidade do vento", ["kmh", "ms", "mph", "kn"], index=0)

# Seleção de capital e busca por cidade
cidade_escolhida = st.sidebar.selectbox("Capitais do Brasil", list(CAPITAIS.keys()), index=1)
st.sidebar.markdown("**Ou pesquise uma cidade (qualquer lugar):**")
consulta = st.sidebar.text_input("Ex.: Lisboa, Buenos Aires, Goiânia…")
cidade_custom = None
if consulta:
    with st.sidebar.status("🔎 Buscando cidades…", expanded=False):
        try:
            resultados = geocode(consulta, count=6)
            if resultados:
                opcoes = {
                    f"{r['name']} — {r.get('admin1', '')}, {r.get('country', '')} (lat {r['latitude']:.2f}, lon {r['longitude']:.2f})": r
                    for r in resultados
                }
                escolha = st.sidebar.selectbox("Resultados", list(opcoes.keys()))
                if st.sidebar.button("Usar cidade pesquisada"):
                    cidade_custom = opcoes[escolha]
                    st.sidebar.success(f"Cidade selecionada: {cidade_custom['name']}")
            else:
                st.sidebar.warning("Nenhum resultado encontrado.")
        except Exception as e:
            st.sidebar.error(f"Falha na busca: {e}")

# Define coordenadas conforme o modo
if modo == "Cidade única":
    if cidade_custom:
        coords = [(cidade_custom["latitude"], cidade_custom["longitude"])]
        nome_principal = cidade_custom["name"]
    else:
        coords = [(CAPITAIS[cidade_escolhida]["lat"], CAPITAIS[cidade_escolhida]["lon"])]
        nome_principal = cidade_escolhida
else:
    # Comparar: selecione múltiplas capitais ou inclua a custom
    selecao = st.sidebar.multiselect(
        "Escolha capitais para comparar",
        list(CAPITAIS.keys()),
        default=[cidade_escolhida, "Rio de Janeiro", "Belo Horizonte"]
    )
    coords = [(CAPITAIS[c]["lat"], CAPITAIS[c]["lon"]) for c in selecao]
    nomes_multiplos = list(selecao)
    if cidade_custom:
        coords.append((cidade_custom["latitude"], cidade_custom["longitude"]))
        nomes_multiplos.append(cidade_custom["name"])

# ----------------------------
# CARREGAMENTO DE DADOS
# ----------------------------
with st.status("☁️ Buscando previsão e montando painéis…", expanded=False) as status:
    try:
        payload = fetch_forecast(coords, dias, unidade_vento)
        status.update(label="☁️ Dados recebidos!", state="complete")
    except Exception as e:
        st.error(f"Erro ao consultar API: {e}")
        st.stop()

# A API pode retornar dict (1 local) ou lista (N locais)
items = payload if isinstance(payload, list) else [payload]

# ----------------------------
# TABS PRINCIPAIS
# ----------------------------
tab_resumo, tab_horas, tab_comparar, tab_sobre = st.tabs(
    ["Resumo", "Horas", "Comparar", "Sobre"]
)

# ----------------------------
# CONTEÚDO: CIDADE ÚNICA
# ----------------------------
def painel_cidade(item: Dict[str, Any], nome_cidade: str):
    hourly = item["hourly"]
    daily = item.get("daily", {})
    tz_offset = int(item.get("utc_offset_seconds", 0))
    now_local = datetime.utcnow() + timedelta(seconds=tz_offset)

    # DataFrame horário
    df = pd.DataFrame({
        "Hora": pd.to_datetime(hourly["time"]),
        "Temperatura (°C)": hourly.get("temperature_2m", []),
        "Sensação (°C)": hourly.get("apparent_temperature", []),
        "Umidade (%)": hourly.get("relative_humidity_2m", []),
        "Prob. Chuva (%)": hourly.get("precipitation_probability", []),
        "Precipitação (mm)": hourly.get("precipitation", []),
        f"Vento ({unidade_vento})": hourly.get("wind_speed_10m", []),
        "UV": hourly.get("uv_index", []),
        "WMO": hourly.get("weather_code", []),
    })

    if df.empty:
        st.warning("Sem dados horários para exibir.")
        return

    idx_agora = nearest_hour_index(df["Hora"], now_local)
    idx_prev = idx_agora - 1 if idx_agora > 0 else None

    # Estado do tempo (WMO)
    wmo_code = int(df.loc[idx_agora, "WMO"]) if not math.isnan(df.loc[idx_agora, "WMO"]) else 0
    wmo_desc, wmo_emoji = WMO.get(wmo_code, ("—", "❔"))

    with tab_resumo:
        st.subheader(f"📍 {nome_cidade} — Agora")
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("🌡️ Temperatura", f"{df.loc[idx_agora, 'Temperatura (°C)']:.1f} °C",
                  delta=kpi_delta(df.loc[idx_agora, 'Temperatura (°C)'], df.loc[idx_prev, 'Temperatura (°C)'] if idx_prev is not None else None))
        k2.metric("🤔 Sensação", f"{df.loc[idx_agora, 'Sensação (°C)']:.1f} °C",
                  delta=kpi_delta(df.loc[idx_agora, 'Sensação (°C)'], df.loc[idx_prev, 'Sensação (°C)'] if idx_prev is not None else None))
        k3.metric("💧 Umidade", f"{df.loc[idx_agora, 'Umidade (%)']:.0f} %",
                  delta=kpi_delta(df.loc[idx_agora, 'Umidade (%)'], df.loc[idx_prev, 'Umidade (%)'] if idx_prev is not None else None))
        k4.metric("☔ Prob. chuva", f"{df.loc[idx_agora, 'Prob. Chuva (%)']:.0f} %",
                  delta=kpi_delta(df.loc[idx_agora, 'Prob. Chuva (%)'], df.loc[idx_prev, 'Prob. Chuva (%)'] if idx_prev is not None else None))
        k5.metric(f"💨 Vento ({unidade_vento})", f"{df.loc[idx_agora, f'Vento ({unidade_vento})']:.0f}",
                  delta=kpi_delta(df.loc[idx_agora, f'Vento ({unidade_vento})'], df.loc[idx_prev, f'Vento ({unidade_vento})'] if idx_prev is not None else None))

        st.markdown(f"**{wmo_emoji} {wmo_desc}** na hora mais próxima de agora.")
        st.caption("Deltas comparam a hora atual com a hora anterior.")

        # Dicas
        st.markdown("### 💡 Dicas rápidas")
        tips = dicas_rapidas(
            temp=float(df.loc[idx_agora, "Temperatura (°C)"]),
            app_temp=float(df.loc[idx_agora, "Sensação (°C)"]),
            prob_chuva=float(df.loc[idx_agora, "Prob. Chuva (%)"]),
            precip=float(df.loc[idx_agora, "Precipitação (mm)"]),
            uv=float(df.loc[idx_agora, "UV"]),
            vento=float(df.loc[idx_agora, f"Vento ({unidade_vento})"]),
        )
        for t in tips:
            st.markdown(f"- {t}")

        # Resumo diário (máx/mín/chuva)
        if daily:
            dd = pd.DataFrame({
                "Data": pd.to_datetime(daily["time"]),
                "Tmax (°C)": daily.get("temperature_2m_max", []),
                "Tmin (°C)": daily.get("temperature_2m_min", []),
                "UV máx": daily.get("uv_index_max", []),
                "Precip. (mm)": daily.get("precipitation_sum", []),
                "Prob. chuva máx (%)": daily.get("precipitation_probability_max", []),
                f"Vento máx ({unidade_vento})": daily.get("wind_speed_10m_max", []),
                "Nascer do sol": pd.to_datetime(daily.get("sunrise", [])),
                "Pôr do sol": pd.to_datetime(daily.get("sunset", [])),
            })
            st.markdown("### 📅 Próximos dias")
            st.dataframe(dd, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Baixar diário (CSV)",
                dd.to_csv(index=False).encode("utf-8"),
                file_name=f"{nome_cidade}_diario.csv",
                mime="text/csv"
            )

    with tab_horas:
        st.subheader(f"📊 Evolução por hora — {nome_cidade}")
        # Temperatura
        fig_temp = px.line(df, x="Hora", y=["Temperatura (°C)", "Sensação (°C)"], markers=True,
                           title="Temperatura & sensação ao longo do período")
        fig_temp.update_layout(hovermode="x unified")
        st.plotly_chart(fig_temp, use_container_width=True)
        # Umidade
        fig_umid = px.line(df, x="Hora", y="Umidade (%)", markers=True, title="Umidade relativa")
        fig_umid.update_layout(hovermode="x unified")
        st.plotly_chart(fig_umid, use_container_width=True)
        # Chuva (probabilidade e mm)
        fig_prob = px.bar(df, x="Hora", y="Prob. Chuva (%)", title="Probabilidade de chuva")
        st.plotly_chart(fig_prob, use_container_width=True)
        fig_chuva = px.bar(df, x="Hora", y="Precipitação (mm)", title="Precipitação prevista")
        st.plotly_chart(fig_chuva, use_container_width=True)
        # Vento
        fig_vento = px.line(df, x="Hora", y=f"Vento ({unidade_vento})", markers=True, title="Velocidade do vento")
        fig_vento.update_layout(hovermode="x unified")
        st.plotly_chart(fig_vento, use_container_width=True)
        # UV
        fig_uv = px.line(df, x="Hora", y="UV", markers=True, title="Índice UV")
        fig_uv.update_layout(hovermode="x unified")
        st.plotly_chart(fig_uv, use_container_width=True)

        st.download_button(
            "⬇️ Baixar horário (CSV)",
            df.drop(columns=["WMO"]).to_csv(index=False).encode("utf-8"),
            file_name=f"{nome_cidade}_horario.csv",
            mime="text/csv"
        )

# Render “Cidade única”
if modo == "Cidade única":
    painel_cidade(items[0], nome_principal)

# ----------------------------
# CONTEÚDO: COMPARAR CAPITAIS
# ----------------------------
with tab_comparar:
    if modo != "Comparar capitais":
        st.info("Use o modo **Comparar capitais** na barra lateral.")
    else:
        st.subheader("🗺️ Comparativo entre cidades")
        if not items:
            st.warning("Sem dados para comparar.")
        else:
            res_rows = []
            for i, it in enumerate(items):
                hourly = it["hourly"]
                tz_offset = int(it.get("utc_offset_seconds", 0))
                now_local = datetime.utcnow() + timedelta(seconds=tz_offset)
                times = pd.Series(pd.to_datetime(hourly["time"]))
                idx_now = nearest_hour_index(times, now_local)

                # Nome (coerente com a ordem enviada)
                nome = nomes_multiplos[i] if i < len(nomes_multiplos) else f"Cidade {i+1}"

                temp = float(hourly["temperature_2m"][idx_now])
                prob = float(hourly.get("precipitation_probability", [0])[idx_now] or 0)
                precip = float(hourly.get("precipitation", [0])[idx_now] or 0)
                vento = float(hourly.get("wind_speed_10m", [0])[idx_now] or 0)
                uv = float(hourly.get("uv_index", [0])[idx_now] or 0)
                wmo_code = int(hourly.get("weather_code", [0])[idx_now] or 0)
                desc, emo = WMO.get(wmo_code, ("—", "❔"))

                lat = float(it["latitude"]) if "latitude" in it else coords[i][0]
                lon = float(it["longitude"]) if "longitude" in it else coords[i][1]

                res_rows.append({
                    "Cidade": nome,
                    "lat": lat, "lon": lon,
                    "Temp (°C)": temp,
                    "Prob. chuva (%)": prob,
                    "Precip. (mm)": precip,
                    f"Vento ({unidade_vento})": vento,
                    "UV": uv,
                    "Condição": desc,
                    "Ícone": emo
                })

            df_comp = pd.DataFrame(res_rows).sort_values(by=["Temp (°C)"], ascending=False)
            c1, c2 = st.columns([1.2, 1])
            with c1:
                st.dataframe(df_comp.drop(columns=["lat", "lon"]), use_container_width=True, hide_index=True)
            with c2:
                st.markdown("**Destaques**")
                quente = df_comp.iloc[0]
                frio = df_comp.iloc[-1]
                st.markdown(f"• Mais quente agora: **{quente['Cidade']}** ({quente['Temp (°C)']:.1f} °C) {quente['Ícone']}")
                st.markdown(f"• Mais fria agora: **{frio['Cidade']}** ({frio['Temp (°C)']:.1f} °C) {frio['Ícone']}")
                chuv = df_comp.sort_values("Prob. chuva (%)", ascending=False).iloc[0]
                st.markdown(f"• Maior chance de chuva: **{chuv['Cidade']}** ({chuv['Prob. chuva (%)']:.0f}%)")

            # Mapa
            fig_map = px.scatter_geo(
                df_comp,
                lat="lat", lon="lon",
                color="Temp (°C)",
                size="Prob. chuva (%)",
                hover_name="Cidade",
                hover_data=["Condição", "Temp (°C)", "Prob. chuva (%)", f"Vento ({unidade_vento})", "UV"],
                projection="natural earth",
                title="Mapa — Temperatura (cor) e probabilidade de chuva (tamanho)"
            )
            st.plotly_chart(fig_map, use_container_width=True)

# ----------------------------
# SOBRE / REFERÊNCIAS
# ----------------------------
with tab_sobre:
    st.markdown("""
**Fonte de dados:** [Open‑Meteo](https://open-meteo.com/) (gratuita para uso não comercial; sem API key).  
**Boas práticas aplicadas:** caching com `st.cache_data`, tabs para organizar conteúdo, status de carregamento, tema via config.toml, mapa interativo e dicas orientadas por variáveis de risco (probabilidade de chuva, UV, vento).  
**Observação:** previsões são sujeitas a modelos e podem variar por região.
""")

