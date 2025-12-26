import streamlit as st
import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt
import io
import soundfile as sf
import whisper

# ======================================================
# FUNÇÕES (SEMPRE NO TOPO)
# ======================================================
def cortar_audio(y, sr, inicio, fim):
    """Corta o áudio entre inicio e fim (em segundos, sem arredondar)"""
    i_ini = int(inicio * sr)
    i_fim = int(fim * sr)
    return y[i_ini:i_fim]


@st.cache_resource
def carregar_modelo():
    return whisper.load_model("tiny")


# ======================================================
# INTERFACE
# ======================================================
st.header("🎧 Leitor de som + transcrição por frases")

filer = st.file_uploader(
    "Escolha o arquivo de som",
    type=["wav", "mp3"]
)

# ======================================================
# PROCESSAMENTO
# ======================================================
if filer is not None:

    # ---------- ÁUDIO ORIGINAL ----------
    y, sr = librosa.load(filer, sr=16000)


    st.success(f"Sample rate: {sr} Hz")
    st.success(f"Número de amostras: {len(y)}")

    # ---------- CONTROLE DE VELOCIDADE (APENAS PLAYER GERAL) ----------
    velocidade = st.slider(
        "Velocidade do áudio (player geral)",
        0.5, 2.0, 1.0, 0.1
    )

    if velocidade != 1.0:
        y_proc = librosa.effects.time_stretch(y, rate=velocidade)
    else:
        y_proc = y

    # ---------- PLAYER DO ÁUDIO COMPLETO ----------
    buffer = io.BytesIO()
    sf.write(buffer, y_proc, sr, format="WAV")
    buffer.seek(0)
    st.audio(buffer, format="audio/wav")

    # ---------- GRÁFICO ----------
    tempo = np.linspace(0, len(y_proc) / sr, len(y_proc))
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(tempo, y_proc)
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Forma de onda do áudio")
    st.pyplot(fig)

    # ---------- WHISPER (USANDO ÁUDIO ORIGINAL) ----------
    with st.spinner("Transcrevendo áudio..."):
        model = carregar_modelo()
        resultado = model.transcribe(
        y,
        task="transcribe")

    # ---------- FRASES (SEM ROUND!) ----------
    dados = [{
        "inicio": seg["start"],   # tempo real
        "fim": seg["end"],        # tempo real
        "frase": seg["text"].strip()
    } for seg in resultado["segments"]]

    df_frases = pd.DataFrame(dados)

    st.subheader("📝 Frases extraídas do áudio")
    st.dataframe(
        df_frases.assign(
            **{
                "inicio (s)": df_frases["inicio"].round(2),
                "fim (s)": df_frases["fim"].round(2)
            }
        )[["inicio (s)", "fim (s)", "frase"]],
        use_container_width=True
    )

    # ---------- ÁUDIO POR FRASE (ALINHADO) ----------
    st.subheader("🎧 Frases com áudio individual")

    for _, row in df_frases.iterrows():
        st.markdown(
            f"**{row['inicio']:.2f} – {row['fim']:.2f} s**"
        )
        st.markdown(row["frase"])

        y_seg = cortar_audio(
            y,          # 🔥 áudio original
            sr,
            row["inicio"],
            row["fim"]
        )

        buffer_seg = io.BytesIO()
        sf.write(buffer_seg, y_seg, sr, format="WAV")
        buffer_seg.seek(0)

        st.audio(buffer_seg, format="audio/wav")
