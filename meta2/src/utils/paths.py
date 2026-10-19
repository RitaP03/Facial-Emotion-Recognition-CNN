from pathlib import Path

# Este ficheiro está em: .../IC/TP/Meta2/src/utils/paths.py
# Serve apenas para centralizar os caminhos do dataset.

BASE = Path(__file__).resolve()

TP_DIR = BASE.parents[3]

# A partir da pasta raiz, apontamos para o dataset organizado em 3 conjuntos
DATASET_DIR = TP_DIR / "dataset"

# Caminhos usados pelo train_model.py e pela otimização
TRAIN_DIR   = str(DATASET_DIR / "train")
VAL_DIR     = str(DATASET_DIR / "validation")
TEST_DIR    = str(DATASET_DIR / "test")
