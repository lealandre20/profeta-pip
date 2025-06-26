import streamlit as st
from streamlit import session_state as ss
import pandas as pd
import plotly.graph_objects as go
import ta
import io
import xlsxwriter
# === Fonte dos Dados: CSV ou Simulado ===
st.subheader("📂 Fonte dos Dados")

fonte_dados = st.radio(
    "Escolha a origem dos dados:",
    ["📂 Arquivo CSV", "🧪 Simulação de Teste"],
    index=0
)
# === CARREGAMENTO DE DADOS ===
@st.cache_data
def load_data_csv(nome_arquivo):
    return pd.read_csv(f"data/{nome_arquivo}", parse_dates=["timestamp"])

@st.cache_data
def load_data_simulada():
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    base_time = pd.to_datetime("2024-01-01 09:00")
    timestamps = [base_time + pd.Timedelta(minutes=15*i) for i in range(300)]
    close = np.cumsum(np.random.randn(300)) + 1900

    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': close + np.random.uniform(-1, 1, size=300),
        'high': close + np.random.uniform(0, 2, size=300),
        'low': close - np.random.uniform(0, 2, size=300),
        'close': close
    })

    return df
# === AVALIAÇÃO DE ACERTOS HIPOTÉTICOS ===
def avaliar_acertos(df, n_candles=3):
    df = df.copy()
    df['Acerto_RSI'] = None
    df['Acerto_MACD'] = None
    df['Acerto_Duplo'] = None

    for i in range(len(df) - n_candles):
        futuro = df.loc[i + n_candles, 'close']
        atual = df.loc[i, 'close']

        # RSI
        if df.loc[i, 'Sinal'] == '🟢 Compra':
            df.loc[i, 'Acerto_RSI'] = futuro > atual
        elif df.loc[i, 'Sinal'] == '🔴 Venda':
            df.loc[i, 'Acerto_RSI'] = futuro < atual

        # MACD
        if df.loc[i, 'Sinal_MACD'] == '🟢 Compra MACD':
            df.loc[i, 'Acerto_MACD'] = futuro > atual
        elif df.loc[i, 'Sinal_MACD'] == '🔴 Venda MACD':
            df.loc[i, 'Acerto_MACD'] = futuro < atual

        # Confirmação
        if df.loc[i, 'Confirmacao_Dupla'] == '🟢 Alvo Duplo':
            df.loc[i, 'Acerto_Duplo'] = futuro > atual
        elif df.loc[i, 'Confirmacao_Dupla'] == '🔴 Alvo Duplo':
            df.loc[i, 'Acerto_Duplo'] = futuro < atual

    return df

# === Estratégia Principal: RSI + EMA ===
def calculate_signals(df):
    df['EMA_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
    df['RSI_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

    df['Sinal'] = '🔍 Neutro'
    df.loc[(df['RSI_14'] < 30) & (df['close'] > df['EMA_21']), 'Sinal'] = '🟢 Compra'
    df.loc[(df['RSI_14'] > 70) & (df['close'] < df['EMA_21']), 'Sinal'] = '🔴 Venda'

    return df


# === Estratégia Alternativa: MACD ===
def calcular_macd_signals(df):
    macd = ta.trend.MACD(df['close'])

    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()

    df['Sinal_MACD'] = '🔍 Neutro'
    df.loc[
        (df['MACD'].shift(1) < df['MACD_signal'].shift(1)) & (df['MACD'] > df['MACD_signal']),
        'Sinal_MACD'
    ] = '🟢 Compra MACD'
    df.loc[
        (df['MACD'].shift(1) > df['MACD_signal'].shift(1)) & (df['MACD'] < df['MACD_signal']),
        'Sinal_MACD'
    ] = '🔴 Venda MACD'

    return df


# === Confirmação Dupla (RSI + MACD) ===
def identificar_confirmacao_dupla(df):
    df['Confirmacao_Dupla'] = ''

    cond_compra = (df['Sinal'] == '🟢 Compra') & (df['Sinal_MACD'] == '🟢 Compra MACD')
    cond_venda = (df['Sinal'] == '🔴 Venda') & (df['Sinal_MACD'] == '🔴 Venda MACD')

    df.loc[cond_compra, 'Confirmacao_Dupla'] = '🟢 Alvo Duplo'
    df.loc[cond_venda, 'Confirmacao_Dupla'] = '🔴 Alvo Duplo'

    return df


# === PLOTAGEM DO GRÁFICO ===
def plot_chart(df, cor_fundo, cor_texto, opcoes_plot):
    fig = go.Figure(data=[go.Candlestick(
        x=df['timestamp'],
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name='Candles'
    )])

    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['EMA_21'],
        mode='lines', name='EMA 21',
        line=dict(color='orange', width=1)
    ))

    if "🟢 RSI/EMA" in opcoes_plot:
        sinais_compra = df[df['Sinal'] == '🟢 Compra']
        sinais_venda = df[df['Sinal'] == '🔴 Venda']
        fig.add_trace(go.Scatter(
            x=sinais_compra['timestamp'],
            y=sinais_compra['low'] * 0.995,
            mode='markers', name='🟢 Compra (RSI)',
            marker=dict(symbol='arrow-up', size=10, color='green')
        ))
        fig.add_trace(go.Scatter(
            x=sinais_venda['timestamp'],
            y=sinais_venda['high'] * 1.005,
            mode='markers', name='🔴 Venda (RSI)',
            marker=dict(symbol='arrow-down', size=10, color='red')
        ))

    if "📘 MACD" in opcoes_plot:
        sinais_macd_compra = df[df['Sinal_MACD'] == '🟢 Compra MACD']
        sinais_macd_venda = df[df['Sinal_MACD'] == '🔴 Venda MACD']
        fig.add_trace(go.Scatter(
            x=sinais_macd_compra['timestamp'],
            y=sinais_macd_compra['low'] * 0.997,
            mode='markers', name='📘 Compra MACD',
            marker=dict(symbol='triangle-up', size=10, color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=sinais_macd_venda['timestamp'],
            y=sinais_macd_venda['high'] * 1.003,
            mode='markers', name='📕 Venda MACD',
            marker=dict(symbol='triangle-down', size=10, color='darkred')
        ))

    if "🟡 Confirmação Dupla" in opcoes_plot:
        duplos = df[df['Confirmacao_Dupla'].isin(['🟢 Alvo Duplo', '🔴 Alvo Duplo'])]
        fig.add_trace(go.Scatter(
            x=duplos['timestamp'],
            y=duplos['close'],
            mode='markers', name='🟡 Confirmação Dupla',
            marker=dict(symbol='star-diamond', size=14, color='gold')
        ))

    fig.update_layout(
        title=f"Gráfico {ativo} – {timeframe}",
        xaxis_title="Horário", yaxis_title="Preço",
        height=500,
        plot_bgcolor=cor_fundo,
        paper_bgcolor=cor_fundo,
        font=dict(color=cor_texto),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


# === INTERFACE PRINCIPAL ===

st.title("🤖 Profeta Pip – Painel de Sinais Técnicos")

if st.button("🔄 Recarregar dados"):
    st.cache_data.clear()
    st.rerun()

ativo = "XAU/USD"
timeframe = "15min"
nome_arquivo = "xauusd_15min.csv"
cor_fundo = "#111111"
cor_texto = "#FFFFFF"

if fonte_dados == "📂 Arquivo CSV":
    df_raw = load_data_csv(nome_arquivo)
else:
    df_raw = load_data_simulada()
df_signals = calculate_signals(df_raw)
df_signals = calcular_macd_signals(df_signals)
df_signals = identificar_confirmacao_dupla(df_signals)
df_signals = avaliar_acertos(df_signals, n_candles=3)

# FILTRO VISUAL
st.subheader("👁️ Filtrar estratégias no gráfico")
opcoes_plot = st.multiselect(
    "Selecione os sinais a exibir:",
    ["🟢 RSI/EMA", "📘 MACD", "🟡 Confirmação Dupla"],
    default=["🟢 RSI/EMA", "📘 MACD", "🟡 Confirmação Dupla"]
)

# GRÁFICO FINAL
st.plotly_chart(
    plot_chart(df_signals, cor_fundo, cor_texto, opcoes_plot),
    use_container_width=True
)

# TABELA
st.subheader("📋 Últimos sinais")
st.dataframe(
    df_signals[['timestamp', 'close', 'Sinal', 'Sinal_MACD', 'Confirmacao_Dupla']].iloc[::-1].head(20),
    use_container_width=True
)

# EXPORTAÇÃO
st.subheader("📥 Exportar sinais para Excel")
def exportar_para_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sinais')
        for i, col in enumerate(df.columns):
            largura = max(df[col].astype(str).map(len).max(), len(col)) + 2
            writer.sheets['Sinais'].set_column(i, i, largura)
    return output.getvalue()

excel_data = exportar_para_excel(df_signals)
st.download_button(
    label="📥 Baixar planilha .xlsx",
    data=excel_data,
    file_name="sinais_profeta.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# REPLAY
st.subheader("🎬 Replay de Sinais")
df_replay = df_signals[
    df_signals['Sinal'].isin(['🟢 Compra', '🔴 Venda']) |
    df_signals['Sinal_MACD'].isin(['🟢 Compra MACD', '🔴 Venda MACD']) |
    df_signals['Confirmacao_Dupla'].isin(['🟢 Alvo Duplo', '🔴 Alvo Duplo'])
].reset_index(drop=True)

if 'replay_index' not in ss:
    ss.replay_index = 0
    # === DIAGNÓSTICO DE DADOS ===
with st.expander("🩺 Diagnóstico de Sinais nos Dados"):
    total = len(df_signals)
    sinais_rsi = df_signals['Sinal'].isin(['🟢 Compra', '🔴 Venda']).sum()
    sinais_macd = df_signals['Sinal_MACD'].isin(['🟢 Compra MACD', '🔴 Venda MACD']).sum()
    sinais_duplos = df_signals['Confirmacao_Dupla'].isin(['🟢 Alvo Duplo', '🔴 Alvo Duplo']).sum()

    st.markdown(f"""
    - **Total de candles carregados:** `{total}`  
    - 🟢 RSI/EMA sinalizados: `{sinais_rsi}`  
    - 📘 MACD sinalizados: `{sinais_macd}`  
    - ✨ Confirmações duplas: `{sinais_duplos}`
    """)

    if sinais_rsi == 0 or sinais_macd == 0:
        st.warning("⚠️ Alguns sinais estão ausentes! Verifique se o CSV possui candles suficientes e se as colunas 'close' e 'timestamp' estão corretas.")
    else:
        st.success("✅ Sinais calculados com sucesso! Todas as colunas estão preenchidas.")
    # === MAPA TÉRMICO DE FREQUÊNCIA POR HORA ===
st.subheader("🧊 Mapa Térmico de Horários mais Ativos")

df_sinais_horario = df_signals.copy()
df_sinais_horario['hora'] = df_sinais_horario['timestamp'].dt.hour

contagem = pd.DataFrame({
    'RSI+EMA': df_sinais_horario[df_sinais_horario['Sinal'].isin(['🟢 Compra', '🔴 Venda'])]['hora'].value_counts(),
    'MACD': df_sinais_horario[df_sinais_horario['Sinal_MACD'].isin(['🟢 Compra MACD', '🔴 Venda MACD'])]['hora'].value_counts(),
    'Duplos': df_sinais_horario[df_sinais_horario['Confirmacao_Dupla'].isin(['🟢 Alvo Duplo', '🔴 Alvo Duplo'])]['hora'].value_counts(),
}).fillna(0).astype(int).sort_index()

if not contagem.empty:
    import seaborn as sns
    import matplotlib.pyplot as plt

    fig_mapa, ax = plt.subplots(figsize=(8, 3))
    sns.heatmap(contagem.T, annot=True, fmt="d", cmap="YlOrRd", ax=ax, cbar=False)
    ax.set_xlabel("Hora do Dia")
    ax.set_ylabel("Estratégia")
    ax.set_title("Sinais por Horário (0h–23h)")
    st.pyplot(fig_mapa)
else:
    st.info("📭 Nenhum sinal foi encontrado para compor o mapa térmico.")
    # === ESTATÍSTICA DE ACERTOS POR ESTRATÉGIA ===
st.subheader("🎯 Taxa de Acertos Hipotética")

def taxa_acerto(col):
    validos = df_signals[col].dropna()
    if len(validos) == 0:
        return "—"
    return f"{(validos.sum() / len(validos)) * 100:.1f}%"

col1, col2, col3 = st.columns(3)
col1.metric("🟢 RSI/EMA", taxa_acerto("Acerto_RSI"))
col2.metric("📘 MACD", taxa_acerto("Acerto_MACD"))
col3.metric("✨ Duplo", taxa_acerto("Acerto_Duplo"))
# === GRÁFICO DE SINAIS POR DATA ===
st.subheader("📅 Distribuição de Sinais por Dia")

df_datas = df_signals.copy()
df_datas['data'] = df_datas['timestamp'].dt.date

sinais_por_dia = pd.DataFrame({
    'RSI+EMA': df_datas[df_datas['Sinal'].isin(['🟢 Compra', '🔴 Venda'])]['data'].value_counts(),
    'MACD': df_datas[df_datas['Sinal_MACD'].isin(['🟢 Compra MACD', '🔴 Venda MACD'])]['data'].value_counts(),
    'Duplos': df_datas[df_datas['Confirmacao_Dupla'].isin(['🟢 Alvo Duplo', '🔴 Alvo Duplo'])]['data'].value_counts(),
}).fillna(0).sort_index()

fig_timeline = go.Figure()
fig_timeline.add_trace(go.Scatter(x=sinais_por_dia.index, y=sinais_por_dia['RSI+EMA'], name='🟢 RSI/EMA', fill='tozeroy'))
fig_timeline.add_trace(go.Scatter(x=sinais_por_dia.index, y=sinais_por_dia['MACD'], name='📘 MACD', fill='tozeroy'))
fig_timeline.add_trace(go.Scatter(x=sinais_por_dia.index, y=sinais_por_dia['Duplos'], name='✨ Duplos', fill='tozeroy'))

fig_timeline.update_layout(
    xaxis_title="Data",
    yaxis_title="Qtde de Sinais",
    height=300,
    template="plotly_dark"
)

st.plotly_chart(fig_timeline, use_container_width=True)
