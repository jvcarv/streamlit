import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd

# Configuração da página
st.set_page_config(layout="wide", page_title="Validação de Segmentação")
st.title("🔍 Validação de Segmentação com Ajuste de Transparência")

# Diretórios
DIR_IMAGENS = "data/G1020/Images_Square"  # Altere para o caminho correto
DIR_MASCARAS = "data/G1020/Masks_RGB"  # Altere para o caminho correto
CSV_FILE = "classificacoes.csv"  # Arquivo para salvar classificações

# Garantir que os diretórios existem
if not os.path.exists(DIR_IMAGENS) or not os.path.exists(DIR_MASCARAS):
    st.error("🚨 Os diretórios especificados não existem. Verifique os caminhos e tente novamente.")
    st.stop()

# Listar imagens disponíveis (somente arquivos .jpg)
imagens_disponiveis = sorted([f for f in os.listdir(DIR_IMAGENS) if f.lower().endswith('.jpg')])

# Estado persistente para index da imagem no carrossel
if "img_index" not in st.session_state:
    st.session_state.img_index = 0

# Botões de navegação
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    if st.button("⬅ Anterior") and st.session_state.img_index > 0:
        st.session_state.img_index -= 1
with col3:
    if st.button("Próximo ➡") and st.session_state.img_index < len(imagens_disponiveis) - 1:
        st.session_state.img_index += 1

# Obter a imagem selecionada
imagem_selecionada = imagens_disponiveis[st.session_state.img_index]

# Ajuste da transparência
alpha = st.slider("Ajuste a transparência da máscara", 0.0, 1.0, 0.5, 0.05)

# Pegar o nome do arquivo sem extensão e buscar a máscara correspondente
nome_base = os.path.splitext(imagem_selecionada)[0]
caminho_img = os.path.join(DIR_IMAGENS, imagem_selecionada)
caminho_mask = os.path.join(DIR_MASCARAS, nome_base + ".png")

# Função para salvar classificação no CSV
def salvar_classificacao(imagem, classificacao):
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
    else:
        df = pd.DataFrame(columns=["Imagem", "Classificacao"])

    df = df[df["Imagem"] != imagem]  # Remover entrada anterior se existir
    df = pd.concat([df, pd.DataFrame([[imagem, classificacao]], columns=["Imagem", "Classificacao"])], ignore_index=True)
    
    df.to_csv(CSV_FILE, index=False)
    st.success(f"✅ Classificação '{classificacao}' salva para {imagem}!")

# Carregar e exibir imagens
if os.path.exists(caminho_img) and os.path.exists(caminho_mask):
    img = cv2.imread(caminho_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converter para RGB

    mask = cv2.imread(caminho_mask, cv2.IMREAD_UNCHANGED)

    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    elif mask.shape[2] == 4:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2BGR)

    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask = np.clip(mask, 0, 255).astype(np.uint8)

    overlay = cv2.addWeighted(img, 1, mask, alpha, 0)

    # Exibir imagem sobreposta com tamanho reduzido e centralizado
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.image(overlay, caption=f"Sobreposição - {imagem_selecionada}", use_column_width=False, width=600)
    st.markdown("</div>", unsafe_allow_html=True)

    # Botões para classificar
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Boa", key="boa"):
            salvar_classificacao(imagem_selecionada, "Boa")
    with col2:
        if st.button("❌ Ruim", key="ruim"):
            salvar_classificacao(imagem_selecionada, "Ruim")

    # Exibir classificações
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        st.dataframe(df)
else:
    st.warning(f"⚠️ Arquivo não encontrado: {caminho_mask}")

st.markdown("---")
st.info("Use o slider para ajustar a transparência e classifique a imagem.")