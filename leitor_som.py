
import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt

st.header('Leitor de som')
filer = st.file_uploader(label = 'Escolha o arquivo de som ', type=['wav', 'mp3'])

if filer is not None:
    #st.write(filer)
    y, sr = librosa.load(filer, sr=22050)  # sr is the sample rate, 
    st.success(f'Sample rate:{sr} hz')
    st.success(f'Número de amostras: {len(y)}')

    #gráfico
     # eixo do tempo
    tempo = np.linspace(0, len(y) / sr, num=len(y))

    # plot da onda
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(tempo, y)
    ax.set_xlabel("Tempo (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Forma de onda do áudio")

    st.pyplot(fig)



