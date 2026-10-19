# ============================================================
# Treino da CNN
# Esta versão é usada pela função de fitness:
#   - permite parametrizar lr e dropout (otimizados por Swarm)
#   - devolve val_accuracy para o algoritmo Swarm (GSA/PSO)
#   - guarda modelo treinado + gráficos
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

# --- Correção para o bug do Keras com SciPy ---
# Alguns builds não expõem "scipy" dentro do keras.preprocessing.
# Garantia de compatibilidade em diferentes instalações.
try:
    import scipy
    try:
        import keras.src.preprocessing.image as kimage_mod
    except ImportError:
        from keras.preprocessing import image as kimage_mod
    kimage_mod.scipy = scipy
except Exception as e:
    print("Aviso: não foi possível injectar scipy no keras.preprocessing.image:", e)

# Import da função que cria a CNN
try:
    from .cnn_model import create_cnn_model
except ImportError:
    from cnn_model import create_cnn_model

    """
    Treina a CNN com os hiper-parâmetros fornecidos e devolve:
        → val_accuracy (última época)
        → modelo treinado
    Esta função é chamada repetidamente pelo GSA/PSO como função de fitness.
    """

def train_model(train_dir,
                val_dir,
                input_shape,
                num_classes,
                batch_size: int = 64, # Valor equilabrado para CNN
                epochs: int = 20, # Valor por omissão. Nós usámos 3
                model_dir: str | None = None,
                img_size: int = 64, # Dataset está a ser trabalhado a 64x64
                lr: float = 0.003614, # Valor razoável para Adam (ordem de 1e-3). 
                dropout: float = 0.25, # Valor típico para o dropout (baixa risco de overfiting)
                variant: str = "advanced"): # Arquitetura utilizada


    # Se não for especificada pasta, guardar outputs em Meta2/results/
    if model_dir is None:
        MODEL_DIR = Path(__file__).resolve().parents[2] / "results"
    else:
        MODEL_DIR = Path(model_dir)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # 1) Data loaders
    # -----------------------------
    # Normalização simples: divide os valores dos pixéis por 255 para os trazer para [0, 1].
    # Isto estabiliza o treino e segue a prática padrão em CNNs para visão.
    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    # Geradores de treino e validação
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,     # treino deve baralhar os batches
    )
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,    # validação não deve baralhar
    )

    # -----------------------------
    # 2) Construção do modelo CNN (parametrizado)
    # -----------------------------
    # Entrada dos híper-parâmetros otimizados:
    #   - lr (learning rate)
    #   - dropout
    # O "variant" permite escolher entre a CNN simple ou advanced.
    model = create_cnn_model(
        input_shape=input_shape,
        num_classes=num_classes,
        variant=variant,
        lr=lr,
        dropout=dropout,
    )

    # Número de batches por época
    steps_per_epoch = int(np.ceil(train_generator.samples / batch_size))
    validation_steps = int(np.ceil(val_generator.samples / batch_size))

    # -----------------------------
    # 3) Treino do modelo
    # -----------------------------
    # O GSA/PSO chamam esta função várias vezes
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=validation_steps,
        verbose=1,
    )

    # -----------------------------
    # 4) Guardar modelo e gráficos
    # -----------------------------
    # Guarda o modelo final
    model_out = MODEL_DIR / "emotion_classifier_cnn.h5"
    model.save(str(model_out))

    # Gráfico da precisão
    plt.figure()
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Precisão do Modelo")
    plt.ylabel("Precisão")
    plt.xlabel("Época")
    plt.legend(["Treino", "Validação"], loc="upper left")
    plt.savefig(str(MODEL_DIR / "accuracy_plot.png"))
    plt.close()

    # Gráfico da perda
    plt.figure()
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Perda do Modelo")
    plt.ylabel("Perda")
    plt.xlabel("Época")
    plt.legend(["Treino", "Validação"], loc="upper left")
    plt.savefig(str(MODEL_DIR / "loss_plot.png"))
    plt.close()

    # -----------------------------
    # 5) Retorno para o Swarm
    # -----------------------------
    # Fitness = última val_accuracy (quanto maior melhor)
    val_acc = float(history.history["val_accuracy"][-1])
    return val_acc, model
