# app.py — versão otimizada para evitar 429 em ambientes gratuitos
from __future__ import annotations

import os
import json
import math
import time
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, timedelta

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import pandas as pd
import plotly.express as px
import streamlit as st

# =========================
# CONFIG DA PÁGINA & ESTILO
# =========================
st.set_page_config(page_title="Clima Brasil — UX Case", page_icon="⛅", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 1.0rem; padding-bottom: 2rem; }
h1, h2, h3, h4 { line-height: 1.2; }
.big-metric .stMetric-value { font-size: 2.0rem; }
.small-note { color: var(--text-color-secondary, #6b7280); font-size: 0.9rem; }
.dataframe tbody tr th { vertical-align: middle; }
@media (max-width: 640px) { .modebar { display: none !important; }
</style>
""", unsafe_allow_html=True)

# =========================
# ESTADO GLOBAL (session)
# =========================
if "payload" not in st.session_state: st.session_state["payload"] = None
if "param_key" not in st.session_state: st.session_state["param_key"] = None
if "last_fetch_ts" not in st.session_state: st.session_state["last_fetch_ts"] = 0.0
if "names_for_payload" not in st.session_state: st.session_state["names_for_payload"] = []
if "cidade_custom" not in st.session_state: st.session_state["cidade_custom"] = None
if "geocode_results" not in st.session_state: st.session_state["geocode_results"] = []

# =========================
# HTTP SESSION + RETRY/BACKOFF
# =========================
@st.cache_resource(show_spinner=False)
def _http_session() -> requests.Session:
    s = requests.Session()
    try:
        # urllib3 >= 1.26
        retries = Retry(
            total=4,
            backoff_factor=0.8,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET"]),
            respect_retry_after_header=True,
            raise_on_status=False,
        )
    except TypeError:
        # compat com versões antigas
        retries = Retry(
            total=4,
            backoff_factor=0.8,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=frozenset(["GET"]),
            raise_on_status=False,
        )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s

_SESSION = _http_session()

def http_get(url: str, **kwargs) -> requests.Response:
    # timeout=(conexão, leitura). Leitura mais folgada p/ Render gratuito.
    timeout = kwargs.pop("timeout", (5, 45))
    return _SESSION.get(url, timeout=timeout, **kwargs)

# =========================
# LOTTIE (opcional, cacheado)
# =========================
def _load_lottie_url(url: str):
    try:
        r = http_get(url, timeout=(3, 10))
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

@st.cache_data(ttl=86400, show_spinner=False)
def load_lottie(urls: List[str]) -> Optional[dict]:
    for u in urls:
        anim = _load_lottie_url(u)
        if anim:
            return anim
    # local opcional
    try:
        if os.path.exists("assets/weather.json"):
            with open("assets/weather.json", "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None

DEFAULT_LOTTIE_URLS = [
    "https://assets10.lottiefiles.com/packages/lf20_qp1q7mct.json",
    "https://assets8.lottiefiles.com/private_files/lf30_weather_icon.json",
]

try:
    from streamlit_lottie import st_lottie
    cols = st.columns([1, 5])
    with cols[0]:
        anim = load_lottie(DEFAULT_LOTTIE_URLS)
        if anim:
            st_lottie(anim, height=110, key="lottie_header")
        elif os.path.exists("assets/weather.gif"):
            st.image("assets/weather.gif", width=110)
        else:
            st.markdown("⛅")
    with cols[1]:
        st.title("Clima nas Capitais — UX Case")
        st.caption("Previsões horárias e diárias com insights úteis para planejamento no Brasil.")
except Exception:
    st.title("Clima nas Capitais — UX Case")
    st.caption("Previsões horárias e diárias com insights úteis para planejamento no Brasil.")

# =========================
# DADOS FIXOS: CAPITAIS
# =========================
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

# =========================
# HELPERS & CACHES
# =========================
def _make_key(coords: List[Tuple[float, float]], dias: int, wind_unit: str) -> Tuple:
    # chave estável para rate-limit local e identificação de cache
    lat_t = tuple(round(c[0], 4) for c in coords)
    lon_t = tuple(round(c[1], 4) for c in coords)
    return (lat_t, lon_t, int(dias), str(wind_unit))

@st.cache_data(ttl=86400, show_spinner=False)
def geocode(query: str, count: int = 6) -> List[Dict[str, Any]]:
    """Busca coordenadas por nome de cidade via Open‑Meteo Geocoding API (cache 24h)."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": query, "count": count, "language": "pt", "format": "json"}
    r = http_get(url, params=params)
    r.raise_for_status()
    return r.json().get("results", [])

@st.cache_data(ttl=1800, show_spinner=False)  # 30 min
def fetch_forecast_cached(lat_t: Tuple[float, ...], lon_t: Tuple[float, ...], forecast_days: int, wind_unit: str) -> Any:
    """Busca previsão horária/diária (cache 30 min). Usa multi‑coordenadas em 1 chamada."""
    base = "https://api.open-meteo.com/v1/forecast"
    hourly_vars = [
        "temperature_2m","apparent_temperature","relative_humidity_2m",
        "precipitation_probability","precipitation","wind_speed_10m",
        "uv_index","weather_code",
    ]
    daily_vars = [
        "temperature_2m_max","temperature_2m_min","uv_index_max",
        "precipitation_sum","precipitation_probability_max","wind_speed_10m_max",
        "sunrise","sunset",
    ]
    params = {
        "latitude": ",".join(str(x) for x in lat_t),
        "longitude": ",".join(str(x) for x in lon_t),
        "hourly": ",".join(hourly_vars),
        "daily": ",".join(daily_vars),
        "forecast_days": max(1, min(int(forecast_days), 16)),
        "timezone": "auto",
        "temperature_unit": "celsius",
        "wind_speed_unit": wind_unit,
        "precipitation_unit": "mm",
    }
    r = http_get(base, params=params)
    # se 429, http_get respeita Retry-After, mas pode ainda retornar 429; deixamos o caller tratar
    r.raise_for_status()
    return r.json()

def nearest_hour_index(times: pd.Series, now_local: datetime) -> int:
    ts = pd.to_datetime(times)
    diffs = (ts - now_local).abs()
    return int(diffs.reset_index(drop=True).idxmin())

WMO = {
    0: ("Céu limpo", "☀️"), 1: ("Poucas nuvens", "🌤️"), 2: ("Parcialmente nublado", "⛅"),
    3: ("Nublado", "☁️"), 45: ("Nevoeiro", "🌫️"), 48: ("Nevoeiro gelado", "🌫️"),
    51: ("Garoa leve", "🌦️"), 53: ("Garoa moderada", "🌦️"), 55: ("Garoa forte", "🌧️"),
    61: ("Chuva fraca", "🌧️"), 63: ("Chuva moderada", "🌧️"), 65: ("Chuva forte", "🌧️"),
    80: ("Pancadas leves", "🌦️"), 81: ("Pancadas moderadas", "🌧️"), 82: ("Pancadas fortes", "⛈️"),
    95: ("Trovoadas", "⛈️"), 96: ("Trovoadas com granizo", "⛈️"), 99: ("Trovoadas fortes com granizo", "⛈️"),
}

def dicas_rapidas(temp: float, app_temp: float, prob_chuva: float, precip: float, uv: float, vento: float) -> List[str]:
    tips = []
    if prob_chuva >= 60 or precip >= 1: tips.append("🌂 **Leve guarda-chuva** (chance de chuva elevada).")
    if temp >= 32 or app_temp >= 34 or uv >= 7: tips.append("🧴 **Protetor solar e hidratação** recomendados.")
    if vento >= 30: tips.append("💨 **Rajadas de vento**: cuidado com objetos soltos.")
    if app_temp <= 12: tips.append("🧥 **Agasalho** vai bem (sensação térmica baixa).")
    if not tips: tips.append("✅ Condições confortáveis na maior parte do dia.")
    return tips

def kpi_delta(curr: float, prev: Optional[float]) -> str:
    if prev is None or (isinstance(prev, float) and (math.isnan(prev) if not isinstance(prev, bool) else False)):
        return "—"
    delta = curr - prev
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.1f}"

# =========================
# SIDEBAR — CONTROLES
# =========================
st.sidebar.header("⚙️ Controles")
modo = st.sidebar.radio("Modo", ["Cidade única", "Comparar capitais"], index=0)
dias = st.sidebar.slider("Dias de previsão", 1, 5, 2, help="Até 16 dias disponíveis na API.")
unidade_vento = st.sidebar.selectbox("Unidade do vento", ["kmh", "ms", "mph", "kn"], index=0)

# Seleção de capital (não consulta API)
cidade_escolhida = st.sidebar.selectbox("Capitais do Brasil", list(CAPITAIS.keys()), index=1)

# Geocoding só quando o usuário clicar em "Buscar"
st.sidebar.markdown("**Ou pesquise uma cidade (qualquer lugar):**")
consulta = st.sidebar.text_input("Ex.: Lisboa, Buenos Aires, Goiânia…")
colb1, colb2 = st.sidebar.columns([1, 1])
buscar = colb1.button("Buscar")
limpar = colb2.button("Limpar")

if limpar:
    st.session_state["cidade_custom"] = None
    st.session_state["geocode_results"] = []

if buscar:
    if not consulta or len(consulta.strip()) < 3:
        st.sidebar.warning("Digite pelo menos 3 caracteres para buscar.")
    else:
        with st.sidebar.status("🔎 Buscando cidades…", expanded=False):
            try:
                st.session_state["geocode_results"] = geocode(consulta.strip(), count=6)
                if not st.session_state["geocode_results"]:
                    st.sidebar.warning("Nenhum resultado encontrado.")
            except Exception as e:
                st.sidebar.error(f"Falha na busca: {e}")

if st.session_state["geocode_results"]:
    opcoes = {
        f"{r['name']} — {r.get('admin1', '')}, {r.get('country', '')} (lat {r['latitude']:.2f}, lon {r['longitude']:.2f})": r
        for r in st.session_state["geocode_results"]
    }
    escolha = st.sidebar.selectbox("Resultados", list(opcoes.keys()))
    if st.sidebar.button("Usar cidade pesquisada"):
        st.session_state["cidade_custom"] = opcoes[escolha]
        st.sidebar.success(f"Cidade: {st.session_state['cidade_custom']['name']}")

# Monta lista de coordenadas (sem chamar a API ainda)
nomes_multiplos: List[str] = []
if modo == "Cidade única":
    if st.session_state["cidade_custom"]:
        coords = [(st.session_state["cidade_custom"]["latitude"], st.session_state["cidade_custom"]["longitude"])]
        nomes_multiplos = [st.session_state["cidade_custom"]["name"]]
    else:
        coords = [(CAPITAIS[cidade_escolhida]["lat"], CAPITAIS[cidade_escolhida]["lon"])]
        nomes_multiplos = [cidade_escolhida]
else:
    selecao = st.sidebar.multiselect(
        "Escolha capitais para comparar", list(CAPITAIS.keys()),
        default=[cidade_escolhida, "Rio de Janeiro", "Belo Horizonte"]
    )
    coords = [(CAPITAIS[c]["lat"], CAPITAIS[c]["lon"]) for c in selecao]
    nomes_multiplos = list(selecao)
    if st.session_state["cidade_custom"]:
        coords.append((st.session_state["cidade_custom"]["latitude"], st.session_state["cidade_custom"]["longitude"]))
        nomes_multiplos.append(st.session_state["cidade_custom"]["name"])

# =========================
# RATE-LIMIT LOCAL + BOTÃO DE ATUALIZAÇÃO
# =========================
MIN_INTERVAL_S = 30  # mínimo entre chamadas reais (por sessão)
param_key_now = _make_key(coords, dias, unidade_vento)

st.sidebar.markdown("---")
colu1, colu2 = st.sidebar.columns([1.2, 1])
update_clicked = colu1.button("🔄 Atualizar dados", type="primary")
last_ts = st.session_state["last_fetch_ts"]
if st.session_state["param_key"] and st.session_state["param_key"] != param_key_now:
    colu2.caption("Parâmetros mudaram — clique **Atualizar dados** para buscar.")

# Decide se vai buscar agora (evita chamadas automáticas)
need_fetch = False
if st.session_state["payload"] is None:
    need_fetch = True  # primeiro carregamento
elif update_clicked:
    # só permite nova busca se passar do intervalo mínimo OU se mudaram os parâmetros
    if (time.time() - last_ts) >= MIN_INTERVAL_S or (st.session_state["param_key"] != param_key_now):
        need_fetch = True
    else:
        st.sidebar.info(f"Aguarde {int(MIN_INTERVAL_S - (time.time()-last_ts))}s para nova atualização.")

# =========================
# BUSCA DE DADOS (apenas quando necessário)
# =========================
if need_fetch:
    lat_t = tuple(round(c[0], 4) for c in coords)
    lon_t = tuple(round(c[1], 4) for c in coords)
    with st.status("☁️ Buscando previsão…", expanded=False) as status:
        try:
            payload = fetch_forecast_cached(lat_t, lon_t, dias, unidade_vento)
            st.session_state["payload"] = payload
            st.session_state["param_key"] = param_key_now
            st.session_state["last_fetch_ts"] = time.time()
            st.session_state["names_for_payload"] = list(nomes_multiplos)
            status.update(label="☁️ Dados recebidos!", state="complete")
        except requests.HTTPError as e:
            code = getattr(e.response, "status_code", None)
            if code == 429:
                status.update(label="⚠️ Limite de taxa atingido (429). Mantendo dados anteriores.", state="error")
                st.warning("A API atingiu o limite de requisições. Mostrando **dados anteriores**. Tente atualizar mais tarde.")
            else:
                status.update(label=f"Erro ao consultar API: {e}", state="error")
                st.error(f"Erro ao consultar API: {e}")
                st.stop()
        except Exception as e:
            status.update(label=f"Erro ao consultar API: {e}", state="error")
            st.error(f"Erro ao consultar API: {e}")
            st.stop()

# Se ainda não temos payload (ex.: erro e primeira carga), interrompe
if st.session_state["payload"] is None:
    st.info("Sem dados em cache para exibir ainda.")
    st.stop()

# A API pode retornar dict (1 local) ou lista (N locais)
payload = st.session_state["payload"]
items = payload if isinstance(payload, list) else [payload]
names_for_payload = st.session_state.get("names_for_payload", [])

# =========================
# TABS
# =========================
tab_resumo, tab_horas, tab_comparar, tab_sobre = st.tabs(["Resumo", "Horas", "Comparar", "Sobre"])

# =========================
# CIDADE ÚNICA
# =========================
def painel_cidade(item: Dict[str, Any], nome_cidade: str):
    hourly = item["hourly"]
    daily = item.get("daily", {})
    tz_offset = int(item.get("utc_offset_seconds", 0))
    now_local = datetime.utcnow() + timedelta(seconds=tz_offset)

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
    }).reset_index(drop=True)

    if df.empty:
        st.warning("Sem dados horários para exibir.")
        return

    idx_agora = nearest_hour_index(df["Hora"], now_local)
    idx_prev = idx_agora - 1 if idx_agora > 0 else None

    try:
        wmo_val = df.loc[idx_agora, "WMO"]
        wmo_code = int(wmo_val) if pd.notna(wmo_val) else 0
    except Exception:
        wmo_code = 0
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
        for t in tips: st.markdown(f"- {t}")

        # Resumo diário
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
            st.download_button("⬇️ Baixar diário (CSV)", dd.to_csv(index=False).encode("utf-8"),
                               file_name=f"{nome_cidade}_diario.csv", mime="text/csv")

    with tab_horas:
        st.subheader(f"📊 Evolução por hora — {nome_cidade}")
        fig_temp = px.line(df, x="Hora", y=["Temperatura (°C)", "Sensação (°C)"], markers=True,
                           title="Temperatura & sensação")
        fig_temp.update_layout(hovermode="x unified")
        st.plotly_chart(fig_temp, use_container_width=True)

        fig_umid = px.line(df, x="Hora", y="Umidade (%)", markers=True, title="Umidade relativa")
        fig_umid.update_layout(hovermode="x unified")
        st.plotly_chart(fig_umid, use_container_width=True)

        st.plotly_chart(px.bar(df, x="Hora", y="Prob. Chuva (%)", title="Probabilidade de chuva"), use_container_width=True)
        st.plotly_chart(px.bar(df, x="Hora", y="Precipitação (mm)", title="Precipitação prevista"), use_container_width=True)

        fig_vento = px.line(df, x="Hora", y=f"Vento ({unidade_vento})", markers=True, title="Velocidade do vento")
        fig_vento.update_layout(hovermode="x unified")
        st.plotly_chart(fig_vento, use_container_width=True)

        fig_uv = px.line(df, x="Hora", y="UV", markers=True, title="Índice UV")
        fig_uv.update_layout(hovermode="x unified")
        st.plotly_chart(fig_uv, use_container_width=True)

        st.download_button("⬇️ Baixar horário (CSV)", df.drop(columns=["WMO"]).to_csv(index=False).encode("utf-8"),
                           file_name=f"{nome_cidade}_horario.csv", mime="text/csv")

# Render cidade única com os dados da última busca (evita nova chamada)
if names_for_payload:
    painel_cidade(items[0], names_for_payload[0])

# =========================
# COMPARAR CAPITAIS
# =========================
with tab_comparar:
    st.subheader("🗺️ Comparativo entre cidades")
    if len(items) <= 1:
        st.info("Use o modo **Comparar capitais** na barra lateral e clique **Atualizar dados**.")
    else:
        res_rows = []
        for i, it in enumerate(items):
            hourly = it["hourly"]
            tz_offset = int(it.get("utc_offset_seconds", 0))
            now_local = datetime.utcnow() + timedelta(seconds=tz_offset)
            times = pd.Series(pd.to_datetime(hourly["time"]))
            idx_now = nearest_hour_index(times, now_local)

            nome = names_for_payload[i] if i < len(names_for_payload) else f"Cidade {i+1}"

            def _get(arr, idx, default=0.0):
                try:
                    v = arr[idx]
                    return float(v) if v is not None else default
                except Exception:
                    return default

            temp = _get(hourly.get("temperature_2m", []), idx_now)
            prob = _get(hourly.get("precipitation_probability", []), idx_now)
            precip = _get(hourly.get("precipitation", []), idx_now)
            vento = _get(hourly.get("wind_speed_10m", []), idx_now)
            uv = _get(hourly.get("uv_index", []), idx_now)

            try:
                wmo_code = int(_get(hourly.get("weather_code", []), idx_now, 0))
            except Exception:
                wmo_code = 0
            desc, emo = WMO.get(wmo_code, ("—", "❔"))

            lat = float(it.get("latitude"))
            lon = float(it.get("longitude"))

            res_rows.append({
                "Cidade": nome, "lat": lat, "lon": lon,
                "Temp (°C)": temp, "Prob. chuva (%)": prob, "Precip. (mm)": precip,
                f"Vento ({unidade_vento})": vento, "UV": uv,
                "Condição": desc, "Ícone": emo
            })

        df_comp = pd.DataFrame(res_rows)
        for col in ["Temp (°C)", "Prob. chuva (%)", "Precip. (mm)", f"Vento ({unidade_vento})", "UV"]:
            df_comp[col] = pd.to_numeric(df_comp[col], errors="coerce")
        df_comp = df_comp.sort_values(by=["Temp (°C)"], ascending=False)

        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.dataframe(df_comp.drop(columns=["lat", "lon"]), use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**Destaques**")
            if not df_comp.empty:
                quente = df_comp.iloc[0]; frio = df_comp.iloc[-1]
                st.markdown(f"• Mais quente agora: **{quente['Cidade']}** ({quente['Temp (°C)']:.1f} °C) {quente['Ícone']}")
                st.markdown(f"• Mais fria agora: **{frio['Cidade']}** ({frio['Temp (°C)']:.1f} °C) {frio['Ícone']}")
                chuv = df_comp.sort_values("Prob. chuva (%)", ascending=False).iloc[0]
                st.markdown(f"• Maior chance de chuva: **{chuv['Cidade']}** ({chuv['Prob. chuva (%)']:.0f}%)")

        # Mapa com bolha mínima + auto-zoom
        df_plot = df_comp.copy()
        df_plot["Bolha"] = df_plot["Prob. chuva (%)"].fillna(0).clip(lower=6)
        fig_map = px.scatter_geo(
            df_plot, lat="lat", lon="lon",
            color="Temp (°C)", size="Bolha", size_max=24, opacity=0.85,
            hover_name="Cidade",
            hover_data={
                "Condição": True, "Temp (°C)": ":.1f", "Prob. chuva (%)": ":.0f",
                f"Vento ({unidade_vento})": ":.0f", "UV": ":.0f", "lat": False, "lon": False
            },
            title="Mapa — Temperatura (cor) e probabilidade de chuva (tamanho)"
        )
        fig_map.update_geos(fitbounds="locations", showland=True, landcolor="rgb(245,245,245)",
                            showcoastlines=True, showcountries=True)
        fig_map.update_traces(marker=dict(line=dict(color="white", width=0.5)))
        st.plotly_chart(fig_map, use_container_width=True)

# =========================
# SOBRE
# =========================
with tab_sobre:
    st.markdown(f"""
**Fonte de dados:** Open‑Meteo (gratuita; sem API key).  
**Anti‑429 implementado:**  
- Cache de previsão **30 min** (`st.cache_data`);  
- **Botão de atualização** (sem chamadas automáticas por rerun);  
- **Rate‑limit local** ({MIN_INTERVAL_S}s por sessão);  
- **Retry/Backoff** respeitando `Retry-After` em 429;  
- Geocoding só com **botão “Buscar”** (sem disparo a cada tecla).
""")
