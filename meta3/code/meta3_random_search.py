# ============================================================
# Meta 3 - Random Search com MobileNetV2
# ============================================================
# Objetivo:
#   - Explorar o espaço de hiperparâmetros de forma aleatória
#   - Comparar depois os melhores resultados com os outros modelos
#   - Guardar:
#       * melhor modelo da Random Search em ficheiro .h5
#       * tabela de resultados em ficheiro .xlsx

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import random

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# ============================================================
# 1. Caminhos 
# ============================================================

BASE_TP_DIR = r"C:\Users\rpess\OneDrive\Área de Trabalho\Curso\3ºANO\1_Semestre\IC\TP"

DATASET_DIR = os.path.join(BASE_TP_DIR, "dataset")
TRAIN_DIR   = os.path.join(DATASET_DIR, "train")
VAL_DIR     = os.path.join(DATASET_DIR, "validation")
TEST_DIR    = os.path.join(DATASET_DIR, "test")

BASE_META3_DIR = os.path.join(BASE_TP_DIR, "Meta3")
MODELOS_DIR    = os.path.join(BASE_META3_DIR, "Modelos")
FIG_DIR        = os.path.join(BASE_META3_DIR, "Figuras")
EXCEL_DIR      = os.path.join(BASE_META3_DIR, "Excel")

os.makedirs(MODELOS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(EXCEL_DIR, exist_ok=True)

print("Train dir:     ", TRAIN_DIR)
print("Validation dir:", VAL_DIR)
print("Test dir:      ", TEST_DIR)

# ============================================================
# 2. Parâmetros gerais
# ============================================================

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5          
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# ============================================================
# 3. Geradores de imagens 
# ============================================================

# normalização
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=SEED
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=SEED
)

NUM_CLASSES = train_gen.num_classes
class_indices = train_gen.class_indices
print("Classes detetadas:", class_indices)

# ============================================================
# 4. Função para construir modelo com hiperparâmetros variáveis
# ============================================================

def build_model(hidden_units=256, dropout_rate=0.5, lr=1e-4):
    """
    Cria um novo modelo MobileNetV2 + top layer com os
    hiperparâmetros indicados.
    """

    base_model = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    for layer in base_model.layers:
        layer.trainable = False  # transfer learning com feature extractor congelado

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(hidden_units, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ============================================================
# 5. Espaço de hiperparâmetros para Random Search
# ============================================================

N_TRIALS = 10  # número de experiências aleatórias

hidden_units_range = (128, 512)       
dropout_range      = (0.3, 0.6)       
lr_log_range       = (1e-4, 1e-2)     

def sample_hyperparameters():
    """
    Amostra aleatoriamente uma combinação de hiperparâmetros.
    """
    # nº de unidades
    hu = random.randint(2, 8) * 64     
    hu = max(hidden_units_range[0], min(hu, hidden_units_range[1]))

    # dropout contínuo 
    dr = random.uniform(dropout_range[0], dropout_range[1])

    # learning rate 
    log_lr_min = np.log10(lr_log_range[0])
    log_lr_max = np.log10(lr_log_range[1])
    log_lr = random.uniform(log_lr_min, log_lr_max)
    lr = 10 ** log_lr

    return hu, dr, lr

# ============================================================
# 6. Loop da Random Search
# ============================================================

results = []

best_val_acc = -1
best_config = None
best_model_path = os.path.join(MODELOS_DIR, "meta3_random_best.h5")

for trial in range(1, N_TRIALS + 1):
    print("\n")
    print("=" * 60)
    print(f"Experiência Random Search {trial}/{N_TRIALS}")
    print("=" * 60)

    # Amostrar hiperparâmetros
    hidden_units, dropout_rate, lr = sample_hyperparameters()

    print(f"Hiperparâmetros amostrados:")
    print(f"  hidden_units = {hidden_units}")
    print(f"  dropout      = {dropout_rate:.3f}")
    print(f"  learning_rate= {lr:.6f}")

    # limpar sessão para não acumular modelos na memória
    K.clear_session()
    tf.random.set_seed(SEED + trial)   # ligeira variação de seed por experiência
    np.random.seed(SEED + trial)
    random.seed(SEED + trial)

    # construir modelo com estes hiperparâmetros
    model = build_model(
        hidden_units=hidden_units,
        dropout_rate=dropout_rate,
        lr=lr
    )

    # treino
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        verbose=1
    )

    # métricas da última época
    train_acc  = history.history["accuracy"][-1]
    val_acc    = history.history["val_accuracy"][-1]
    train_loss = history.history["loss"][-1]
    val_loss   = history.history["val_loss"][-1]

    print("\nResultados finais desta experiência:")
    print(f"  train_acc  = {train_acc:.4f}")
    print(f"  val_acc    = {val_acc:.4f}")
    print(f"  train_loss = {train_loss:.4f}")
    print(f"  val_loss   = {val_loss:.4f}")

    # guardar nas listas de resultados
    results.append({
        "trial": trial,
        "hidden_units": hidden_units,
        "dropout": float(dropout_rate),
        "learning_rate": float(lr),
        "epochs": EPOCHS,
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss)
    })

    # atualizar melhor modelo
    if val_acc > best_val_acc:
        print("Novo melhor modelo (Random Search) encontrado, a guardar...")
        best_val_acc = val_acc
        best_config = {
            "hidden_units": hidden_units,
            "dropout": float(dropout_rate),
            "learning_rate": float(lr)
        }
        model.save(best_model_path)
        print("Modelo guardado em:", best_model_path)

# ============================================================
# 7. Guardar resultados em Excel
# ============================================================

df_results = pd.DataFrame(results)
excel_path = os.path.join(EXCEL_DIR, "meta3_random_results.xlsx")
df_results.to_excel(excel_path, index=False)

print("\nResultados completos da Random Search guardados em:")
print(excel_path)

print("\nMelhor configuração (Random Search) encontrada:")
print(best_config)
print(f"Melhor val_accuracy (Random Search): {best_val_acc:.4f}")
print("Melhor modelo Random Search guardado em:", best_model_path)
