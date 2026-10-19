# cnn_model.py
# ============================================================
# Define a arquitetura da CNN usada no projeto.
# A função create_cnn_model cria e COMPILA o modelo, para que
# possa ser treinado diretamente pelo train_model.py ou pelo Swarm.
#
# Duas variantes disponíveis:
#   - "simple"   → arquitetura pequena (testes rápidos)
#   - "advanced" → arquitetura mais profunda (modelo final)
# ============================================================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam


# Função que cria a CNN parametrizada
def create_cnn_model(input_shape, num_classes, variant="advanced", lr=0.003614, dropout=0.25):
    """
    input_shape : dimensões das imagens (ex: 64x64x3)
    num_classes : nº de classes da classificação (5 emoções)
    variant     : escolhe arquitetura 'simple' ou 'advanced'
    lr          : learning rate (otimizado pelo Swarm) [Valor razoável para Adam (ordem de 1e-3).]
    dropout     : taxa de dropout (otimizada pelo Swarm) [Valor típico para o dropout (baixa risco de overfiting)]
    """

    # ============================================================
    # ------ MODELO SIMPLES (arquitetura mais leve) -------------
    # ------------------------------------------------------------
    # Usado para testes rápidos — menos camadas, menos cálculos.
    # ============================================================
    if variant == "simple":
        model = Sequential()

        # Input layer (define o formato das imagens)
        model.add(Input(shape=input_shape))

        # 1º bloco convolucional (padrões simples: bordas, linhas)
        model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
        model.add(BatchNormalization())         # estabiliza distribuição das ativações
        model.add(MaxPooling2D((2,2)))          # reduz resolução pela metade

        # 2º bloco convolucional (padrões mais complexos)
        model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2)))

        # Flatten transforma o mapa 2D de features em vetor 1D
        model.add(Flatten())

        # Camada densa totalmente conectada
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(dropout))             # regularização para evitar overfitting

        # Camada de saída — probabilidade para cada emoção
        model.add(Dense(num_classes, activation='softmax'))

        # Compilação — define como a rede aprende
        model.compile(
            optimizer=Adam(learning_rate=lr),   # learning rate vem do Swarm
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


    # ============================================================
    # ------ MODELO AVANÇADO (versão usada no trabalho) ------
    # ------------------------------------------------------------
    # CNN mais profunda → aprende padrões complexos de expressões.
    # Tem 5 blocos de convolução + pooling.
    # ============================================================
    model = Sequential()
    model.add(Input(shape=input_shape))

    # 1º bloco — padrões básicos (bordas)
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    # 2º bloco — combina padrões do bloco anterior
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    # 3º bloco — padrões intermédios
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    # 4º bloco — padrões ainda mais complexos (expressões completas)
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    # 5º bloco — reforço de features profundas
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    # Flatten → prepara dados para a camada densa
    model.add(Flatten())

    # Camada densa (128 neurónios)
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout))        

    # Saída final — 5 classes com softmax
    model.add(Dense(num_classes, activation='softmax'))

    # Compilação do modelo (Adam com learning rate ajustado pelo Swarm)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
