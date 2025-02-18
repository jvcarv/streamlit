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
CSV_FILE = "classificacoes.csv"  # Arquivo onde será salva a classificação

# Garantir que os diretórios existem
if not os.path.exists(DIR_IMAGENS) or not os.path.exists(DIR_MASCARAS):
    st.error("🚨 Os diretórios especificados não existem. Verifique os caminhos e tente novamente.")
    st.stop()

# Listar imagens disponíveis (somente arquivos .jpg)
imagens_disponiveis = sorted([f for f in os.listdir(DIR_IMAGENS) if f.lower().endswith('.jpg')])

# Criar uma seleção de imagem
imagem_selecionada = st.selectbox("Escolha uma imagem para visualizar:", imagens_disponiveis)

# Ajuste da transparência
alpha = st.slider("Ajuste a transparência da máscara", 0.0, 1.0, 0.5, 0.05)

# Pegar o nome do arquivo sem extensão e buscar a máscara correspondente
nome_base = os.path.splitext(imagem_selecionada)[0]
caminho_img = os.path.join(DIR_IMAGENS, imagem_selecionada)
caminho_mask = os.path.join(DIR_MASCARAS, nome_base + ".png")

# Função para salvar classificação no CSV
def salvar_classificacao(imagem, classificacao):
    # Verifica se o arquivo já existe
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
    else:
        df = pd.DataFrame(columns=["Imagem", "Classificacao"])
    
    # Remove a entrada anterior (se existir) e adiciona a nova
    df = df[df["Imagem"] != imagem]
    df = pd.concat([df, pd.DataFrame([[imagem, classificacao]], columns=["Imagem", "Classificacao"])], ignore_index=True)
    
    # Salva o arquivo atualizado
    df.to_csv(CSV_FILE, index=False)

    st.success(f"✅ Classificação '{classificacao}' salva para {imagem}!")

# Carregar e exibir imagens
if os.path.exists(caminho_img) and os.path.exists(caminho_mask):
    # Carregar a imagem
    img = cv2.imread(caminho_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converter para RGB

    # Carregar a máscara (pode ter 1, 3 ou 4 canais)
    mask = cv2.imread(caminho_mask, cv2.IMREAD_UNCHANGED)

    # Se a máscara for grayscale (1 canal), converter para RGB
    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

    # Se a máscara tiver 4 canais (RGBA), remover o canal alpha e converter para RGB
    elif mask.shape[2] == 4:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2BGR)

    # Garantir que a máscara tem o mesmo tamanho da imagem original
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Normalizar a máscara para o intervalo correto (0 a 255)
    mask = np.clip(mask, 0, 255).astype(np.uint8)

    # Criar a sobreposição
    overlay = cv2.addWeighted(img, 1, mask, alpha, 0)

    # Exibir a imagem sobreposta
    st.image(overlay, caption="Sobreposição", use_column_width=True)

    # Botões para classificar
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Boa"):
            salvar_classificacao(imagem_selecionada, "Boa")
    with col2:
        if st.button("❌ Ruim"):
            salvar_classificacao(imagem_selecionada, "Ruim")

    # Exibir o arquivo CSV atualizado
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        st.dataframe(df)
else:
    st.warning(f"⚠️ Arquivo não encontrado: {caminho_mask}")

st.markdown("---")
st.info("Use o slider para ajustar a transparência, classifique a imagem e salve os resultados.")
