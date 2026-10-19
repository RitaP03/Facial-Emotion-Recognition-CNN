# ============================================================
# Meta 3 - Avaliação Final dos Modelos no Conjunto de Teste
# 
# Objetivo:
#   - Carregar os melhores modelos:
#       * Modelo default (Transfer Learning)
#       * Melhor modelo Grid Search
#       * Melhor modelo Random Search
#       * Melhor modelo PSO
#   - Avaliar todos no conjunto de teste independente
#   - Gerar matrizes de confusão
#   - Guardar resultados em Excel
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

from sklearn.metrics import confusion_matrix, classification_report

# ============================================================
# 1. Caminhos principais 
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
# 2. Parâmetros gerais e seed
# ============================================================

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# 3. Gerador apenas para TESTE
# ============================================================

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input # normalizar como a MobileNetV2 espera
)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical", # labels one-hot (5 classes)
    shuffle=False
)

NUM_CLASSES = test_gen.num_classes
class_indices = test_gen.class_indices # mapeamento das classes; ordem das classes
print("Classes detetadas:", class_indices)

# obtenção das classes e respetivos índices 
idx_to_class = {v: k for k, v in class_indices.items()}
class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
print("Ordem das classes:", class_names)

# ============================================================
# 4. Função auxiliar para gerar matriz de confusão 
# ============================================================

def plot_confusion_matrix(cm, class_names, title, fig_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    # anotar valores dentro da matriz
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("Classe verdadeira")
    plt.xlabel("Classe predita")
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight")
    plt.show()
    print("Matriz de confusão guardada em:", fig_path)

# ============================================================
# 5. Lista dos modelos a avaliar
# ============================================================

modelos_info = [
    {
        "nome": "Default_TransferLearning",
        "ficheiro": "meta3_default_mobilenetv2.h5",
        "descricao": "Modelo default com MobileNetV2"
    },
    {
        "nome": "GridSearch_Best",
        "ficheiro": "meta3_grid_best.h5",
        "descricao": "Melhor modelo obtido por Grid Search"
    },
    {
        "nome": "RandomSearch_Best",
        "ficheiro": "meta3_random_best.h5",
        "descricao": "Melhor modelo obtido por Random Search"
    },
    {
        "nome": "PSO_Best",
        "ficheiro": "meta3_pso_best.h5",
        "descricao": "Melhor modelo obtido pelo Particle Swarm Optimization"
    }
]

# ============================================================
# 6. Loop de avaliação dos modelos no conjunto de teste
# ============================================================

resultados = [] 

y_true = test_gen.classes # rótulos verdadeiros

for info in modelos_info:
    nome_modelo = info["nome"]
    ficheiro_modelo = info["ficheiro"]
    descricao = info["descricao"]

    model_path = os.path.join(MODELOS_DIR, ficheiro_modelo)

    print("\n" + "=" * 60)
    print(f"A avaliar modelo: {nome_modelo}")
    print("Descrição:", descricao)
    print("Ficheiro:", model_path)
    print("=" * 60)

    if not os.path.exists(model_path):
        print("ATENÇÃO: ficheiro não encontrado, modelo ignorado.")
        continue

    # carregar modelo
    model = load_model(model_path)
    model.summary()

    # é importante fazer reset ao gerador antes de avaliar e prever
    # garante que percorremos o conjunto completo desde o início em cada passo.
    test_gen.reset()
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)
    print(f"\nDesempenho no conjunto de teste - {nome_modelo}:")
    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"  Test loss:     {test_loss:.4f}")

    # previsões
    test_gen.reset()
    y_prob = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)

    # matriz de confusão
    cm = confusion_matrix(y_true, y_pred)

    # relatório detalhado por classe (precision, recall, f1)
    print("\nRelatório de classificação para o modelo:", nome_modelo)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # guardar gráfico da matriz de confusão
    fig_filename = f"meta3_confusion_{nome_modelo}.png"
    fig_path = os.path.join(FIG_DIR, fig_filename)
    plot_confusion_matrix(
        cm,
        class_names,
        title=f"Matriz de confusão - {nome_modelo}",
        fig_path=fig_path
    )

    # guardar resultados numéricos numa lista
    resultados.append({
        "modelo": nome_modelo,
        "descricao": descricao,
        "ficheiro_modelo": ficheiro_modelo,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss)
    })

# ============================================================
# 7. Guardar resumo em Excel
# ============================================================

if len(resultados) > 0:
    df_res = pd.DataFrame(resultados)
    excel_path = os.path.join(EXCEL_DIR, "meta3_final_eval_results.xlsx")
    df_res.to_excel(excel_path, index=False)
    print("\nResumo final da avaliação guardado em:")
    print(excel_path)
else:
    print("\nNenhum modelo foi avaliado, verifique os ficheiros em:", MODELOS_DIR)
