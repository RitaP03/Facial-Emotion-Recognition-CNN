# ============================================================
# Meta 3 - Grid Search com MobileNetV2
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

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
    class_mode="categorical", # labels one-hot
    shuffle=True, # baralhar treino
    seed=SEED # mantém consistência
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True, # baralhar validação
    seed=SEED
)

NUM_CLASSES = train_gen.num_classes
class_indices = train_gen.class_indices # mapeamento das classes
print("Classes detetadas:", class_indices)

# ============================================================
# 4. Função para construir modelo com hiperparâmetros variáveis
# ============================================================

def build_model(hidden_units=256, dropout_rate=0.5, lr=1e-4):
    """
    Cria um novo modelo MobileNetV2 + top layer com os
    hiperparâmetros indicados.
    Cada chamada constrói um modelo de raiz.
    """

    base_model = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output # obtenção de features da MobileNetV2
    x = GlobalAveragePooling2D()(x) # reduzir dimensionalidade
    x = Dense(hidden_units, activation="relu")(x) # aprender a combinação de features
    x = Dropout(dropout_rate)(x) # reduzir overfitting
    outputs = Dense(NUM_CLASSES, activation="softmax")(x) # probabilidades por classe

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ============================================================
# 5. Definição da grelha de hiperparâmetros
# ============================================================

hidden_units_list = [128, 256, 512] # tamanhos da camada densa
dropout_list      = [0.3, 0.5] # testar reularização
lr_list           = [1e-3, 1e-4] # testar taxas de aprendizagem 

print("Grid de hiperparâmetros:")
print("hidden_units:", hidden_units_list)
print("dropout:     ", dropout_list)
print("learning_rate:", lr_list)

# ============================================================
# 6. Loop da Grid Search
# ============================================================

results = []   

best_val_acc = -1
best_config = None
best_model_path = os.path.join(MODELOS_DIR, "meta3_grid_best.h5")

comb_index = 0
total_combs = len(hidden_units_list) * len(dropout_list) * len(lr_list)

for hidden_units in hidden_units_list:
    for dropout_rate in dropout_list:
        for lr in lr_list:
            comb_index += 1
            print("\n")
            print("=" * 60)
            print(f"Combinação {comb_index}/{total_combs}")
            print(f"hidden_units={hidden_units}, dropout={dropout_rate}, lr={lr}")
            print("=" * 60)

            # limpar sessão para não acumular modelos na memória
            K.clear_session()
            tf.random.set_seed(SEED)
            np.random.seed(SEED)

            model = build_model(
                hidden_units=hidden_units,
                dropout_rate=dropout_rate,
                lr=lr
            )

            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=EPOCHS,
                verbose=1
            )

            # obter métricas da última época
            train_acc = history.history["accuracy"][-1]
            val_acc   = history.history["val_accuracy"][-1]
            train_loss = history.history["loss"][-1]
            val_loss   = history.history["val_loss"][-1]

            print(f"Resultados finais desta combinação:")
            print(f"  train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
            print(f"  train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            # guardar no array de resultados
            results.append({
                "hidden_units": hidden_units,
                "dropout": dropout_rate,
                "learning_rate": lr,
                "epochs": EPOCHS,
                "train_accuracy": float(train_acc),
                "val_accuracy": float(val_acc),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss)
            })

            # se esta combinação for melhor, guardar modelo
            if val_acc > best_val_acc:
                print("Novo melhor modelo encontrado, a guardar...")
                best_val_acc = val_acc
                best_config = {
                    "hidden_units": hidden_units,
                    "dropout": dropout_rate,
                    "learning_rate": lr
                }
                model.save(best_model_path)
                print("Modelo guardado em:", best_model_path)

# ============================================================
# 7. Guardar resultados em Excel
# ============================================================
df_results = pd.DataFrame(results)
excel_path = os.path.join(EXCEL_DIR, "meta3_grid_results.xlsx")
df_results.to_excel(excel_path, index=False)
print("\nResultados completos da Grid Search guardados em:")
print(excel_path)

print("\nMelhor configuração encontrada:")
print(best_config)
print(f"Melhor val_accuracy: {best_val_acc:.4f}")
print("Melhor modelo guardado em:", best_model_path)