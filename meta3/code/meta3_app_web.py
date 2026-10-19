# Objetivo:
#   - Carregar o melhor modelo treinado (meta3_default_mobilenetv2.h5).
#   - Disponibilizar uma interface web para:
#         * Fazer upload de uma imagem
#         * Ou usar a webcam
#         * Obter a classe prevista e as probabilidades por emoção

import os
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

import gradio as gr

# ============================================================
# 1. Caminhos e parâmetros gerais
# ============================================================
BASE_TP_DIR = r"C:\Users\rpess\OneDrive\Área de Trabalho\Curso\3ºANO\1_Semestre\IC\TP"

BASE_META3_DIR = os.path.join(BASE_TP_DIR, "Meta3")
MODELOS_DIR    = os.path.join(BASE_META3_DIR, "Modelos")

# Melhor modelo encontrado na fase de otimização (default)
BEST_MODEL_PATH = os.path.join(MODELOS_DIR, "meta3_default_mobilenetv2.h5")

# Tamanho de imagem usado em todos os modelos MobileNetV2
IMG_SIZE = 224

# Classes na ordem usada no treino:
# {'anger': 0, 'fear': 1, 'happy': 2, 'sad': 3, 'surprise': 4}
CLASS_NAMES = ["anger", "fear", "happy", "sad", "surprise"]

# ============================================================
# 2. Carregar o modelo treinado
# ============================================================

if not os.path.exists(BEST_MODEL_PATH):
    raise FileNotFoundError(
        f"Modelo não encontrado em: {BEST_MODEL_PATH}\n"
        f"Verifica se já os códigos de treino e otimização foram executados "
        f"e se o ficheiro .h5 foi gerado."
    )

print(f"A carregar modelo a partir de: {BEST_MODEL_PATH}")
model = load_model(BEST_MODEL_PATH)
print("Modelo carregado com sucesso.")

# ============================================================
# 3. Funções auxiliares de pré-processamento e previsão
# ============================================================

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Recebe uma imagem PIL, aplica os mesmos passos usados no treino:
      - conversão para RGB
      - resize para 224x224
      - conversão para array
      - expansão de dimensão (batch size 1)
      - preprocess_input da MobileNetV2
    Devolve um array pronto a ser enviado para model.predict.
    """
    # garantir que está em RGB
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def predict_image(image):
    """
    Função chamada pela interface Gradio.
    Recebe uma imagem (PIL ou numpy), faz pré-processamento, corre o modelo
    e devolve:
      - classe prevista (string)
      - probabilidades por classe
    """

    # caso de não haja captura pela webcam
    if image is None:
        msg = (
            "Nenhuma imagem capturada. "
            "Clica no ícone da câmara para tirar a foto e só depois em Submit."
        )
        probs_zero = {cls: 0.0 for cls in CLASS_NAMES}
        return msg, probs_zero

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # pré-processamento
    img_batch = preprocess_image(image)

    # previsão
    preds = model.predict(img_batch)
    preds = preds[0]  

    # índice da classe mais provável
    best_idx = int(np.argmax(preds))
    best_class = CLASS_NAMES[best_idx]

    # probabilidades das classes
    probs = {cls: float(preds[i]) for i, cls in enumerate(CLASS_NAMES)}
    label_text = f"Classe prevista: {best_class} ({preds[best_idx] * 100:.2f}%)"

    return label_text, probs

# ============================================================
# 4. Definição da interface Gradio 
# ============================================================

title = "Meta 3 - Classificação de Expressões Faciais"
description = (
    "Escolhe o método de entrada nas abas abaixo:\n"
    "- **Upload**: Para carregar imagens do computador.\n"
    "- **Webcam**: Para tirar uma foto."
)

with gr.Blocks(title=title) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)
    
    with gr.Tabs():
        # --- ABA 1: UPLOAD ---
        with gr.TabItem("📁 Upload de Imagem"):
            with gr.Row():
                # Coluna da Esquerda: Input
                with gr.Column():
                    img_upload = gr.Image(
                        sources=["upload"], 
                        type="pil", # garante formato consistente e compatibilidade com o pré-processamento do treino
                        label="Carrega a tua imagem aqui"
                    )
                    btn_upload = gr.Button("Classificar Upload", variant="primary")
                
                # Coluna da Direita: Output
                with gr.Column():
                    out_text_up = gr.Textbox(label="Resultado")
                    out_probs_up = gr.Label(label="Probabilidades")
            
            # Ação do botão
            btn_upload.click(
                fn=predict_image, 
                inputs=img_upload, 
                outputs=[out_text_up, out_probs_up]
            )

        # --- ABA 2: WEBCAM ---
        with gr.TabItem("📷 Usar Webcam"):
            gr.Markdown("Instruções: Clicar no ícone da câmara e depois clicar no botão abaixo.")
            with gr.Row():
                # Coluna da Esquerda: Input
                with gr.Column():
                    # streaming=False garante que ele espera uma FOTO e não vídeo
                    img_webcam = gr.Image(
                        sources=["webcam"], 
                        type="pil", 
                        label="Câmara", 
                        streaming=False
                    )
                    btn_webcam = gr.Button("Tirar Foto e Classificar", variant="primary")
                
                # Coluna da Direita: Output
                with gr.Column():
                    out_text_cam = gr.Textbox(label="Resultado")
                    out_probs_cam = gr.Label(label="Probabilidades")

            # Ação do botão
            btn_webcam.click(
                fn=predict_image, 
                inputs=img_webcam, 
                outputs=[out_text_cam, out_probs_cam]
            )

# ============================================================
# 5. Ponto de entrada
# ============================================================

if __name__ == "__main__":
    demo.launch()
