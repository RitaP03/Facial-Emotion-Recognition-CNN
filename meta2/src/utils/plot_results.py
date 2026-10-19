# train_model.py
# ============================================================
# Apesar do nome do ficheiro ser plot_results.py,
# este código é na verdade uma versão simplificada do train_model.py.
# Treina a CNN e gera os gráficos de precisão e perda.
# É útil para testes isolados sem o sistema de otimização.
# ============================================================

import matplotlib.pyplot as plt                  # Biblioteca para gráficos
import numpy as np                               # Operações numéricas
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cnn_model import create_cnn_model           # Função que constrói a CNN
from pathlib import Path                         # Manipulação de paths

# Função de treino (versão simples, sem fitness nem integração com swarm)
def train_model(train_dir, val_dir, input_shape, num_classes,
                batch_size=64, epochs=20, model_dir=None, img_size=64):
    """
    Treina a CNN usando diretórios de treino e validação.
    Gera:
      - modelo treinado
      - gráficos de precisão e perda
    Esta função NÃO devolve val_accuracy (ao contrário da versão usada no swarm).
    É uma versão para uso manual e visualização dos resultados.
    """

    print("Carregando dados de treino...")

    # Se o utilizador não indicar a pasta onde guardar outputs,
    # cria automaticamente uma pasta 'model' no projeto.
    if model_dir is None:
        MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
    else:
        MODEL_DIR = Path(model_dir)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 1. Preparação dos dados (ImageDataGenerator)
    # ============================================================
    # Normaliza todas as imagens para valores entre 0 e 1
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen   = ImageDataGenerator(rescale=1./255)

    # Cria "geradores" de batches — leitura incremental das imagens
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True                 # treino deve baralhar para não enviesar batches
    )
    print(f"Imagens de treino carregadas: {train_generator.samples} imagens")

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False                # validação não deve baralhar
    )
    print(f"Imagens de validação carregadas: {val_generator.samples} imagens")

    # ============================================================
    # 2. Criação do modelo CNN
    # ============================================================
    # Chama o construtor definido em cnn_model.py
    model = create_cnn_model(input_shape, num_classes)
    print("Modelo criado. Iniciando treino...")

    # Número de batches necessários por época
    steps_per_epoch  = int(np.ceil(train_generator.samples / batch_size))
    validation_steps = int(np.ceil(val_generator.samples / batch_size))

    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")

    # ============================================================
    # 3. Treino do modelo
    # ============================================================
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=validation_steps
    )
    print("Treino completo.")

    # ============================================================
    # 4. Guardar modelo treinado
    # ============================================================
    model_out = MODEL_DIR / 'emotion_classifier_cnn.h5'
    model.save(str(model_out))
    print(f"Modelo guardado em: {model_out}")

    # ============================================================
    # 5. Criação dos gráficos
    # ============================================================
    # ACCURACY
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Precisão do Modelo')
    plt.ylabel('Precisão')
    plt.xlabel('Época')
    plt.legend(['Treino', 'Validação'], loc='upper left')
    plt.savefig(str(MODEL_DIR / 'accuracy_plot.png'))

    # LOSS
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Perda do Modelo')
    plt.ylabel('Perda')
    plt.xlabel('Época')
    plt.legend(['Treino', 'Validação'], loc='upper left')
    plt.savefig(str(MODEL_DIR / 'loss_plot.png'))
