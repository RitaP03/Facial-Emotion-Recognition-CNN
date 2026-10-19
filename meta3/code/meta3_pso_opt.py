# ============================================================
# Meta 3 - PSO (Particle Swarm Optimization) com MobileNetV2
# ============================================================
# Objetivo:
#   - Utilizar PSO para otimizar hiperparâmetros do topo da MobileNetV2.
#   - Comparar posteriormente com os outros modelos
#   - Guardar:
#       * melhor modelo PSO em ficheiro .h5
#       * tabela de resultados em ficheiro .xlsx
# ============================================================

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

# normaliação
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

    # garantir tipos corretos para o Keras
    hidden_units = int(hidden_units)
    dropout_rate = float(dropout_rate)
    lr = float(lr)

    base_model = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    for layer in base_model.layers:
        layer.trainable = False

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
# 5. Espaço de hiperparâmetros para PSO
# ============================================================
HU_MIN, HU_MAX = 128, 512 
D_MIN, D_MAX = 0.3, 0.6
LR_MIN, LR_MAX = 1e-4, 1e-2
LOG_LR_MIN, LOG_LR_MAX = np.log10(LR_MIN), np.log10(LR_MAX) # explorar learning rates igualmente distribuídos por ordens de grandeza.

def clip(value, vmin, vmax): # valorees dentro dos limites
    return max(vmin, min(value, vmax))

def position_to_hyperparams(pos):
    """
    Converte posição contínua da partícula em hiperparâmetros discretos/contínuos.
    pos[0] - hidden_units (arredondado ao múltiplo de 64 mais próximo)
    pos[1] - dropout
    pos[2] - log10(lr)
    """
    hu_cont = clip(pos[0], HU_MIN, HU_MAX)
    # arredondar ao múltiplo de 64 mais próximo
    hu = int(round(hu_cont / 64.0) * 64)
    hu = clip(hu, HU_MIN, HU_MAX)

    dr = clip(pos[1], D_MIN, D_MAX)

    log_lr = clip(pos[2], LOG_LR_MIN, LOG_LR_MAX)
    lr = 10 ** log_lr

    return hu, dr, lr

def random_position():
    """
    Gera uma posição inicial aleatória no espaço contínuo.
    """
    hu = random.uniform(HU_MIN, HU_MAX)
    dr = random.uniform(D_MIN, D_MAX)
    log_lr = random.uniform(LOG_LR_MIN, LOG_LR_MAX)
    return np.array([hu, dr, log_lr], dtype=np.float32)

# ============================================================
# 6. Parâmetros do PSO
# ============================================================
NUM_PARTICLES = 5
NUM_ITERATIONS = 2

w  = 0.7   # peso de inércia
c1 = 1.5   # peso da melhor posição individual
c2 = 1.5   # peso da melhor posição global

# Estruturas para PSO
particles_pos = []
particles_vel = []
particles_best_pos = []
particles_best_score = []  # melhor val_accuracy de cada partícula

global_best_pos = None
global_best_score = -1.0

# Resultados para Excel
results = []

best_model_path = os.path.join(MODELOS_DIR, "meta3_pso_best.h5")

# ============================================================
# 7. Inicialização das partículas
# ============================================================

for i in range(NUM_PARTICLES):
    pos = random_position()
    vel = np.zeros_like(pos)
    particles_pos.append(pos) # posição atual
    particles_vel.append(vel) # velocidade atual
    particles_best_pos.append(pos.copy())
    particles_best_score.append(-1.0)

print("\nSwarm inicializado com", NUM_PARTICLES, "partículas.")

# ============================================================
# 8. Loop principal do PSO
# ============================================================

eval_counter = 0 

for it in range(1, NUM_ITERATIONS + 1):
    print("\n")
    print("=" * 60)
    print(f"Iteração PSO {it}/{NUM_ITERATIONS}")
    print("=" * 60)

    # 8.1 Avaliar cada partícula na posição atual
    for p in range(NUM_PARTICLES):
        print(f"\nPartícula {p + 1}/{NUM_PARTICLES} na iteração {it}")

        pos = particles_pos[p]
        hu, dr, lr = position_to_hyperparams(pos)

        print("Hiperparâmetros da partícula:")
        print(f"  hidden_units = {hu}")
        print(f"  dropout      = {dr:.3f}")
        print(f"  learning_rate= {lr:.6f}")

        # limpar sessão para não acumular modelos na memória
        K.clear_session()
        # seeds ligeiramente diferentes por iteração e partícula
        tf.random.set_seed(SEED + it * 10 + p)
        np.random.seed(SEED + it * 10 + p)
        random.seed(SEED + it * 10 + p)

        # construir modelo
        model = build_model(
            hidden_units=hu,
            dropout_rate=dr,
            lr=lr
        )

        # treino
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS,
            verbose=1
        )

        eval_counter += 1

        # métricas da última época
        train_acc  = history.history["accuracy"][-1]
        val_acc    = history.history["val_accuracy"][-1]
        train_loss = history.history["loss"][-1]
        val_loss   = history.history["val_loss"][-1]

        print("\nResultados finais desta avaliação:")
        print(f"  train_acc  = {train_acc:.4f}")
        print(f"  val_acc    = {val_acc:.4f}")
        print(f"  train_loss = {train_loss:.4f}")
        print(f"  val_loss   = {val_loss:.4f}")

        # guardar para Excel
        results.append({
            "evaluation": eval_counter,
            "iteration": it,
            "particle": p + 1,
            "hidden_units": hu,
            "dropout": float(dr),
            "learning_rate": float(lr),
            "epochs": EPOCHS,
            "train_accuracy": float(train_acc),
            "val_accuracy": float(val_acc),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss)
        })

        # atualizar best individual (pbest)
        if val_acc > particles_best_score[p]:
            particles_best_score[p] = val_acc
            particles_best_pos[p] = pos.copy()

        # atualizar best global (gbest)
        global global_best_pos, global_best_score
        if val_acc > global_best_score:
            print("Novo melhor modelo global (PSO) encontrado, a guardar...")
            global_best_score = val_acc
            global_best_pos = pos.copy()

            # guardar modelo como melhor da PSO
            model.save(best_model_path)
            print("Modelo PSO guardado em:", best_model_path)

    # 8.2 Atualizar velocidades e posições das partículas
    print("\nAtualização de velocidades e posições das partículas...")

    for p in range(NUM_PARTICLES):
        r1 = np.random.rand(3)
        r2 = np.random.rand(3)

        pos = particles_pos[p]
        vel = particles_vel[p]
        pbest = particles_best_pos[p]
        gbest = global_best_pos

        # fórmula standard do PSO
        vel_new = (
            w * vel
            + c1 * r1 * (pbest - pos)
            + c2 * r2 * (gbest - pos)
        )
        pos_new = pos + vel_new

        # garantir que as novas posições estão dentro dos limites
        pos_new[0] = clip(pos_new[0], HU_MIN, HU_MAX)
        pos_new[1] = clip(pos_new[1], D_MIN, D_MAX)
        pos_new[2] = clip(pos_new[2], LOG_LR_MIN, LOG_LR_MAX)

        particles_vel[p] = vel_new
        particles_pos[p] = pos_new

    print("Fim da iteração", it)

# ============================================================
# 9. Resumo final e guardar resultados em Excel
# ============================================================

df_results = pd.DataFrame(results)
excel_path = os.path.join(EXCEL_DIR, "meta3_pso_results.xlsx")
df_results.to_excel(excel_path, index=False)

print("\nResultados completos do PSO guardados em:")
print(excel_path)

best_hu, best_dr, best_lr = position_to_hyperparams(global_best_pos)
print("\nMelhor configuração PSO encontrada (a partir da global_best_pos):")
print(f"  hidden_units = {best_hu}")
print(f"  dropout      = {best_dr:.3f}")
print(f"  learning_rate= {best_lr:.6f}")
print(f"Melhor val_accuracy (PSO): {global_best_score:.4f}")
print("Melhor modelo PSO guardado em:", best_model_path)
