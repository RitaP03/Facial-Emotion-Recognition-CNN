# ============================================================
# Meta 3 - Modelo default com MobileNetV2 (Transfer Learning)
# Usamos o dataset já dividido em:
#   train, validation, test
# Guarda o modelo e gráficos obtidos
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# ============================================================
# 1. Definição de caminhos
# ============================================================

# caminho para a pasta TP
BASE_TP_DIR = r"C:\Users\rpess\OneDrive\Área de Trabalho\Curso\3ºANO\1_Semestre\IC\TP"

DATASET_DIR = os.path.join(BASE_TP_DIR, "dataset")
TRAIN_DIR   = os.path.join(DATASET_DIR, "train")
VAL_DIR     = os.path.join(DATASET_DIR, "validation")
TEST_DIR    = os.path.join(DATASET_DIR, "test")

BASE_META3_DIR = os.path.join(BASE_TP_DIR, "Meta3")
MODELOS_DIR    = os.path.join(BASE_META3_DIR, "Modelos")
FIG_DIR        = os.path.join(BASE_META3_DIR, "Figuras")

os.makedirs(MODELOS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print("Train dir:     ", TRAIN_DIR)
print("Validation dir:", VAL_DIR)
print("Test dir:      ", TEST_DIR)

# ============================================================
# 2. Definir parâmetros gerais e seeds
# ============================================================

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# 3. Geradores de imagens (train, validation, test)
# ============================================================

# normalização
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# carregamento de imagens
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
    shuffle=True, # evita que o modelo veja os dados na mesma ordem
    seed=SEED
)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False # no teste não queremos aleatoriedade; avaliação consistente
)

NUM_CLASSES = train_gen.num_classes
class_indices = train_gen.class_indices
print("Classes detetadas:", class_indices)

# ============================================================
# 4. Definir a MobileNetV2 e a top layer default
# ============================================================

base_model = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

for layer in base_model.layers:
    layer.trainable = False # congela MobileNetV2 (evita re-treinameto)

def build_default_model(hidden_units=256, dropout_rate=0.5, lr=1e-4):
    """
    Modelo default da Meta 3.
    Hidden_units e dropout_rate depois vão ser alvo de otimização na Grid, Random e PSO.
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # reduz mapa de características a um vetor; diminui o nº de parâmetros
    x = Dense(hidden_units, activation="relu")(x) # aprende combinações
    x = Dropout(dropout_rate)(x) # reduz overfitting
    outputs = Dense(NUM_CLASSES, activation="softmax")(x) # gera probabilidades

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        # categorical_crossentropy -> problema de classificação multi-classe com rótulos one-hot, 
        # e a função penaliza (loss) previsões pouco confiantes mesmo quando a classe está correta.
        loss="categorical_crossentropy", 
        metrics=["accuracy"]
    )
    return model

model = build_default_model()
model.summary()

# ============================================================
# 5. Treino do modelo default (com balanceamento das classes)
# ============================================================
from sklearn.utils.class_weight import compute_class_weight

# 1. Calcular pesos baseados na frequência das classes no treino
cls_train = train_gen.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(cls_train),
    y=cls_train
)

# 2. Penalizar mais erros em classes minoritárias
# Balanceamento do treino
class_weights_dict = dict(enumerate(class_weights))
print("Pesos das classes calculados (Balanceamento):", class_weights_dict)

# 3. Treinar com o argumento class_weight
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights_dict  
)

# ============================================================
# 6. Função para guardar gráficos de accuracy e loss
# ============================================================

def plot_history(history, fig_prefix, fig_dir):
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(len(acc))

    # Accuracy
    plt.figure()
    plt.plot(epochs_range, acc, label="Treino")
    plt.plot(epochs_range, val_acc, label="Validação")
    plt.xlabel("Épocas")
    plt.ylabel("Accuracy")
    plt.title("Accuracy - Modelo default")
    plt.legend()
    acc_path = os.path.join(fig_dir, f"{fig_prefix}_accuracy.png")
    plt.savefig(acc_path, bbox_inches="tight")
    plt.show()

    # Loss
    plt.figure()
    plt.plot(epochs_range, loss, label="Treino")
    plt.plot(epochs_range, val_loss, label="Validação")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.title("Loss - Modelo default")
    plt.legend()
    loss_path = os.path.join(fig_dir, f"{fig_prefix}_loss.png")
    plt.savefig(loss_path, bbox_inches="tight")
    plt.show()

    print("Gráficos guardados em:")
    print(" -", acc_path)
    print(" -", loss_path)

plot_history(history, "meta3_default_mobilenetv2", FIG_DIR)

# ============================================================
# 7. Avaliação no conjunto de teste
# ============================================================

test_loss, test_acc = model.evaluate(test_gen)
print(f"Accuracy no conjunto de teste (modelo default): {test_acc:.4f}")
print(f"Loss no conjunto de teste (modelo default): {test_loss:.4f}")

# ============================================================
# 8. Guardar modelo default
# ============================================================

default_model_path = os.path.join(MODELOS_DIR, "meta3_default_mobilenetv2.h5")
model.save(default_model_path)
print("Modelo default guardado em:", default_model_path)

# ============================================================
# 9. Análise do Impacto do Tamanho do Dataset
# ============================================================
print("\n--- A iniciar análise do impacto do tamanho do dataset ---")

# Percentagens de dados a testar
fractions = [0.25, 0.50, 0.75, 1.0]
acc_results = []

# Guardar o número total de passos originais
original_steps = train_gen.samples // BATCH_SIZE

for frac in fractions:
    print(f"\nTreinando com {frac*100}% dos dados...")
    
    # 1. Determinar quantos passos correspondem a esta fração
    # Limita quantos batches o modelo vê
    steps_fraction = int(original_steps * frac)
    if steps_fraction < 1: steps_fraction = 1 
    
    # 2. Construir um modelo NOVO (reset aos pesos)
    temp_model = build_default_model() 
    
    # 3. Treinar com número limitado de passos (steps_per_epoch)
    temp_hist = temp_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=5,  
        steps_per_epoch=steps_fraction,
        class_weight=class_weights_dict, 
        verbose=0 
    )
    
    # 4. Avaliar no conjunto de validação
    val_loss, val_acc = temp_model.evaluate(val_gen, verbose=0)
    acc_results.append(val_acc)
    print(f"Fração: {frac} -> Val Accuracy: {val_acc:.4f}")

# ============================================================
# 10. Gráfico do Impacto do Tamanho
# ============================================================

plt.figure(figsize=(8, 6))
plt.plot([f"{int(f*100)}%" for f in fractions], acc_results, marker='o', linestyle='-', color='b')
plt.title("Impacto do Tamanho do Dataset na Performance")
plt.xlabel("Percentagem do Dataset de Treino Usado")
plt.ylabel("Validation Accuracy")
plt.grid(True)

impact_path = os.path.join(FIG_DIR, "meta3_impacto_dataset_size.png")
plt.savefig(impact_path)
plt.show()

print("Gráfico de impacto guardado em:", impact_path)
