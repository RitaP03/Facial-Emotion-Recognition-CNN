# ============================================================
# Meta 2 - Otimização da CNN com GSA e PSO
#
# Script principal da Meta 2.
# Aqui é feita a procura pelos melhores hiper-parâmetros da CNN
# usando algoritmos swarm (GSA e PSO), conforme a teoria do PDF.
#
# Os hiper-parâmetros optimizados são:
#   - learning rate (lr)
#   - dropout (dr)
#
# A avaliação do fitness exige treinar a CNN → muito custoso.
# Por isso, a pesquisa é reduzida (4 agentes, 6 iterações).
# ============================================================

from pathlib import Path
import sys, csv, time, math
import numpy as np

# ------------------------------------------------------------
# 1) Caminho para as pastas corretas 
# ------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent      
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import dos algoritmos swarm da biblioteca SwarmPackagePy
# (GSA = Gravitational Search Algorithm ; PSO = Particle Swarm Optimization)
from SwarmPackagePy import gsa, pso

# Função de treino da CNN (retorna val_accuracy)
from model.train_model import train_model

# Caminhos do dataset (train / validation)
from utils.paths import TRAIN_DIR, VAL_DIR

# ------------------------------------------------------------
# 2) Configuração base da CNN
# ------------------------------------------------------------
INPUT_SHAPE = (64, 64, 3)   # resolução das imagens + rgb
NUM_CLASSES = 5             # 5 emoções 

# Espaço de procura dos hiper-parâmetros:
# lr ∈ [1e-5, 1e-2] — intervalo típico e seguro para o Adam
# dr ∈ [0.10, 0.50] — valores recomendados para regularização em CNN
# O vetor de parâmetros tem sempre a forma [learning_rate, dropout].
# LB e UB definem os limites mínimo e máximo de cada hiper-parâmetro
# respeitando essa ordem: LB = [lr_min, dr_min] e UB = [lr_max, dr_max].
LB = np.array([1e-5, 0.10], dtype=float)
UB = np.array([1e-2, 0.50], dtype=float)

# Pasta results/ onde guardamos CSVs e logs
RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Pasta para gráficos das execuções
PLOTS_DIR = RESULTS / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 3) Função de fitness
# Treina a CNN com [lr, dr] e devolve -val_accuracy
# (Os algoritmos swarm minimizam → por isso devolvemos negativo)
# ------------------------------------------------------------
def fitness(params):
    # Converter para numpy array
    p = np.asarray(params, dtype=float)

    # Penalizamos parâmetros inválidos (NaN/Inf) devolvendo um fitness muito alto.
    # Isto impede o swarm de explorar regiões impossíveis e mantém a procura estável.
    if np.any(np.isnan(p)) or np.any(np.isinf(p)):
        return 1e3

    # Garantimos que os parâmetros ficam sempre dentro dos limites [LB, UB].
    # Isto impede que o swarm explore valores inválidos ou instáveis (por ex. lr demasiado alto)
    # e mantém a procura dentro do espaço definido
    p = np.clip(p, LB, UB)
    # Extração do lr e do dr, que o Swarm está a otimizar
    lr, dr = float(p[0]), float(p[1])

    # Verificação extra para evitar valores absurdos que podem surgir durante a procura:
    #   - lr fora do intervalo seguro pode causar divergência no treino (NaNs)
    #   - dropout > 0.9 desliga neurónios a mais e impede a rede de aprender
    # Se algum valor for inválido, devolvemos um fitness muito alto.
    # Como o swarm minimiza o fitness, isto vai marcar a solução como "muito má"
    # e faz com que as partículas evitem essa região do espaço de procura.
    if not (1e-6 < lr <= 1.0) or not (0.0 <= dr <= 0.9):
        return 1e3

    # Treino real da CNN com os hiper-parâmetros fornecidos.
    # Esta chamada executa 3 épocas de treino para obter a val_accuracy,
    # que serve como medida de qualidade (fitness) da solução.
    # Treinamos apenas 3 épocas para reduzir o custo. O objetivo é comparar hiper-parâmetros,
    # não treinar a CNN até convergir totalmente.
    val_acc, _ = train_model(
        TRAIN_DIR,
        VAL_DIR,
        input_shape=INPUT_SHAPE,
        num_classes=NUM_CLASSES,
        batch_size=64, # Nº de img processadas de cada vez
        epochs=3,          
        img_size=64,
        lr=lr,
        dropout=dr,
        variant="advanced", # Uso da arquitetura completa da CNN
    )

    # Penalização se accuracy for inválida
    if math.isnan(val_acc) or math.isinf(val_acc):
        return 1e3

    # Os algoritmos swarm minimizam o fitness. 
    # Como queremos maximizar a val_accuracy, devolvemos o seu negativo para que 
    # accuracy maior → fitness menor (melhor).
    return -float(val_acc)



# ------------------------------------------------------------
# 4) Wrappers para GSA e PSO
# Simplificam a chamada aos algoritmos swarm.
# Cada um devolve o melhor vetor [lr, dr] encontrado.
# ------------------------------------------------------------
def run_gsa(n_agents=4, iters=6):
    """
    Executa o algoritmo GSA com n_agents definidos. 
    n_agents: número de partículas usadas na procura. 
        Cada partícula representa uma solução [lr, dr]. 
        Mais partículas -> exploram melhor -> processo mais lento
    O argumento '2' indica que o vetor de parâmetros tem dimensão 2: [learning_rate, dropout].
    O GSA procura melhorar as soluções e devolve o melhor vetor encontrado.
    """
    opt = gsa(n_agents, fitness, LB, UB, 2, iters)
    return opt.get_Gbest()


def run_pso(n_agents=4, iters=6, w=0.7, c1=1.7, c2=1.7):
    """
    Executa o algoritmo PSO para otimizar os hiper-parâmetros [lr, dr].

    No PSO, cada partícula é uma solução [lr, dr] e ajusta a sua velocidade usando:
      - w  → inércia (mantém parte do movimento anterior)
      - c1 → puxa para aquilo que a própria partícula descobriu ser bom
      - c2 → puxa para aquilo que o grupo descobriu ser bom
      No nosso caso, c1 e c2, são iguais. 
      Ou seja, há equilíbrio entre memória individual e influência social,
      que é o comportamento é estável e clássico para PSO
          
    n_agents é o número de partículas a explorar o espaço.
    '2' indica que otimizamos dois parâmetros: [lr, dropout].
    Devolve a melhor solução encontrada.
    """
    opt = pso(n_agents, fitness, LB, UB, 2, iters, w, c1, c2)
    return opt.get_Gbest()


# ------------------------------------------------------------
# 5) MAIN – Execução principal
# Aqui é feita a otimização completa.
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=== Meta 2 - Otimização da CNN com GSA e PSO ===")
    print(f"Train dir : {TRAIN_DIR}")
    print(f"Val dir   : {VAL_DIR}")
    print(f"Search LB : {LB}")
    print(f"Search UB : {UB}")
    print("Config: 4 agentes, 6 iterações, 3 épocas por avaliação.\n")

    start = time.time()

    # --------------------------------------------------------
    # GSA
    # --------------------------------------------------------
    best_gsa = run_gsa(n_agents=4, iters=6)

    # --------------------------------------------------------
    # PSO
    # --------------------------------------------------------
    best_pso = run_pso(n_agents=4, iters=6)

    dur = time.time() - start

    # Verificação de validade da solução do GSA
    best_gsa_arr = np.asarray(best_gsa, dtype=float) if best_gsa is not None else None
    gsa_is_nan = (
        best_gsa_arr is None or
        np.any(np.isnan(best_gsa_arr)) or
        np.any(np.isinf(best_gsa_arr))
    )

    # Mensagem de aviso no terminal (GSA instável nesta aplicação)
    if gsa_is_nan:
        print("\n[AVISO] GSA devolveu parâmetros inválidos (NaN/Inf). "
              "No relatório podemos referir que o algoritmo não convergiu "
              "de forma estável para este problema.\n")

    print("--------------- RESULTADOS FINAIS ---------------")
    print(f"GSA → melhor (lr, dr): {best_gsa}")
    print(f"PSO → melhor (lr, dr): {best_pso}")
    print(f"Tempo total aproximado: {dur/3600:.2f} horas")
    print("-------------------------------------------------\n")

    # --------------------------------------------------------
    # Guardar CSV com os melhores parâmetros (para o relatório)
    # --------------------------------------------------------
    out_csv = RESULTS / "cnn_results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["algoritmo", "lr", "dropout"])
        writer.writeheader()

        # Só guardamos GSA se for válido
        if not gsa_is_nan:
            writer.writerow(
                {"algoritmo": "GSA",
                 "lr": float(best_gsa_arr[0]),
                 "dropout": float(best_gsa_arr[1])}
            )

        # PSO normalmente funciona bem aqui
        writer.writerow(
            {"algoritmo": "PSO",
             "lr": float(best_pso[0]),
             "dropout": float(best_pso[1])}
        )

    # Guardar também em formato .txt para consulta rápida
    (RESULTS / "best_params.txt").write_text(
        f"GSA best (lr, dropout): {best_gsa}\n"
        f"PSO best (lr, dropout): {best_pso}\n"
        f"Tempo total (horas): {dur/3600:.2f}\n",
        encoding="utf-8",
    )
