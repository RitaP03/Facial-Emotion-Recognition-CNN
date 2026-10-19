# fix_split_80_20.py  
# ============================================================
# Este script divide as imagens das pastas "train" e "validation"
# de forma equilibrada (80% treino / 20% validação) por classe.
#
# O objetivo é garantir que cada emoção (anger, fear, etc.)
# tem a mesma proporção de imagens de treino e validação.
# ============================================================

from pathlib import Path       # para criar e navegar entre pastas
import random, shutil, math     # random → embaralhar; shutil → mover ficheiros; math → cálculos simples

# Define uma semente fixa para o gerador aleatório
# (assim, se corrermos outra vez, o resultado é sempre o mesmo)
random.seed(42)

# ============================================================
# 1) Definir as pastas principais do dataset
# ------------------------------------------------------------
# ROOT        -> pasta principal do projeto (TP)
# DATASET_DIR -> onde estão as imagens
# TRAIN, VAL  -> subpastas com as imagens de treino e validação
# POOL        -> pasta temporária onde as imagens são reunidas
# ============================================================
ROOT        = Path(__file__).resolve().parent.parent        # .../TP
DATASET_DIR = ROOT / "dataset"
TRAIN = DATASET_DIR / "train"
VAL   = DATASET_DIR / "validation"
POOL  = DATASET_DIR / "_pool_tmp"   # pasta temporária (é apagada no final)

# ============================================================
# 2) Definir as classes (emoções) e tipos de ficheiros aceites
# ------------------------------------------------------------
# CLASSES_OK -> conjunto das 5 emoções que esperamos encontrar
# EXTS -> extensões de ficheiros de imagem válidas
# ============================================================
CLASSES_OK = {"anger", "fear", "happy", "sad", "surprise"}
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# ============================================================
# 3) Função auxiliar para listar todas as imagens numa pasta
# ------------------------------------------------------------
# Percorre recursivamente a pasta dada e devolve uma lista
# com todos os ficheiros que tenham extensões de imagem.
# ============================================================
def files_in(p):
    return [f for f in p.rglob("*") if f.suffix.lower() in EXTS]

# ============================================================
# 4) Verificações iniciais ("sanity checks")
# ------------------------------------------------------------
# Aqui confirmamos se as pastas train e validation existem
# e criamos as subpastas de cada classe se ainda não existirem.
# ============================================================

# Se as pastas principais não existirem, o programa para com erro
assert TRAIN.exists() and VAL.exists(), "Pastas dataset/train e dataset/validation não encontradas."

# Procura quais as classes já existentes dentro de 'train'
found = {p.name for p in TRAIN.iterdir() if p.is_dir()}

# Verifica se falta alguma das 5 classes
missing = CLASSES_OK - found
if missing:
    print(f" Faltam classes em TRAIN: {sorted(missing)} (vai criar pastas vazias).")

# Cria as pastas das classes em train e validation se ainda não existirem
for c in CLASSES_OK:
    (TRAIN / c).mkdir(parents=True, exist_ok=True)
    (VAL / c).mkdir(parents=True, exist_ok=True)

# Cria também a pasta temporária "_pool_tmp"
POOL.mkdir(parents=True, exist_ok=True)

# ============================================================
# 5) Passo 1 — Reunir todas as imagens (train + val) na "pool"
# ------------------------------------------------------------
# Para cada classe:
#   - junta as imagens de treino e validação numa única pasta temporária
#   - isto garante que podemos redistribuir de novo de forma limpa (80/20)
# ============================================================
for c in sorted(CLASSES_OK):
    train_c = TRAIN / c
    val_c   = VAL / c
    pool_c  = POOL / c
    pool_c.mkdir(parents=True, exist_ok=True)

    # Junta todas as imagens desta classe (tanto de treino como de validação)
    all_files = files_in(train_c) + files_in(val_c)

    # Se não houver imagens, passa à próxima classe
    if not all_files:
        print(f"[{c}] sem imagens; a saltar…")
        continue

    # Move cada imagem para a pasta temporária "pool"
    for f in all_files:
        # Se existirem nomes iguais, o último move sobrescreve o anterior
        shutil.move(str(f), str(pool_c / f.name))

# ============================================================
# 6) Passo 2 — Redistribuir as imagens (80% treino / 20% validação)
# ------------------------------------------------------------
# Agora que as imagens estão todas juntas na "pool", baralhamos
# e voltamos a colocá-las nas pastas certas.
# ============================================================
for c in sorted(CLASSES_OK):
    train_c = TRAIN / c
    val_c   = VAL / c
    pool_c  = POOL / c

    # Lista de todas as imagens da classe atual dentro da pool
    pool_list = files_in(pool_c)

    # Se não houver imagens nesta classe, passa à próxima
    if not pool_list:
        continue

    # Baralha a lista de imagens (para garantir aleatoriedade)
    random.shuffle(pool_list)

    # Conta o número total de imagens
    total = len(pool_list)

    # Calcula quantas devem ir para treino (80%)
    n_train = math.ceil(0.8 * total)  # arredonda para cima

    # Move as imagens baralhadas para as pastas certas
    for i, f in enumerate(pool_list):
        # As primeiras 80% vão para treino, o resto para validação
        dst_dir = train_c if i < n_train else val_c
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), str(dst_dir / f.name))

    # Tenta apagar a pasta temporária desta classe (pode não conseguir se sobrar algo)
    try:
        pool_c.rmdir()
    except OSError:
        pass

    # Mostra no terminal o número de imagens em cada divisão
    print(f"[{c}] total={total} → train={n_train}, val={total - n_train}")

# ============================================================
# 7) Passo 3 — Limpar a pasta temporária (se estiver vazia)
# ------------------------------------------------------------
# Se a "_pool_tmp" já não tiver nada dentro, apagamos completamente.
# ============================================================
try:
    POOL.rmdir()
except OSError:
    pass

# ============================================================
# 8) Mensagem final — tudo correu bem
# ============================================================
print("\n 80/20 seguro concluído por classe (apenas train/validation).")
