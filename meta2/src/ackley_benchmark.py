# Benchmark GSA vs PSO na função de Ackley (2D e 3D).
# Este ficheiro valida se os algoritmos swarm estão a funcionar corretamente
# antes de serem aplicados à CNN, que é muito mais cara e ruidosa de otimizar.
# A Ackley é usada como teste de referência, permitindo comparar exploração e 
# convergência.
# Os resultados daqui confirmam o comportamento esperado de GSA e PSO e dão
# confiança para usar os algoritmos no trabalho.

import csv
from pathlib import Path
from SwarmPackagePy import gsa as gsa_fn, pso as pso_fn, testFunctions as tf

# Pasta onde serão guardados os resultados 
RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Função auxiliar para garantir que o valor da Ackley vem como float
def f_ackley(vec):
    # A função da biblioteca pode devolver numpy types -> convertemos para float
    return float(tf.ackley_function(vec))

def run_ackley(dim=2, n=30, iters=200, lb=-5.0, ub=5.0, gsa_g0=3.0, pso_params=None, test_name="Padrão"):
    """
    Corre GSA e PSO para a função de Ackley na dimensão especificada.
    - dim: número de dimensões do problema (2D ou 3D)
        2: permite ver rapidamente convergência e comportamento do swarm
    - n: nº de agentes
        30: diversidade suficiente, não torna alg. lento, resultados comparáveis - standard
    - iters: nº de iterações
        200: suficiente para ver convergência no PSO e GSA
    - lb/ub: limites do espaço de procura
        [-5, 5]: benchmarks oficiais
    - gsa_g0: parâmetro de gravidade inicial
        3: default da biblioteca SwarmPackagePy
    - pso_params: dicionário com W, C1, C2 para testes de sensibilidade
        w = 0.7, c1 = 1.7, c2 = 1.7: bom equilíbrio entre exploração e convergência (recomendados na literatura)
    - test_name: nome do teste (para identificação no CSV)
    """

    # -------------------------
    # GSA (versão base da lib)
    # -------------------------
    # Corre o GSA da biblioteca na função de Ackley. Esta versão não deixa ajustar o G0,
    # por isso usamos o GSA padrão. 
    # get_Gbest() dá a melhor posição encontrada e 
    # f_ackley calcula o valor da Ackley nesse ponto (convertido para float).
    G = gsa_fn(n, tf.ackley_function, lb, ub, dim, iters)
    g_best_pos = list(map(float, G.get_Gbest()))
    g_best_val = f_ackley(g_best_pos)

    # -------------------------
    # PSO (com variação de W)
    # -------------------------
    # Parâmetros padrão (W=0.7, C1=C2=1.7) 
    pso_config = pso_params if pso_params else {"w": 0.7, "c1": 1.7, "c2": 1.7}

    P = pso_fn(
        n, tf.ackley_function, lb, ub, dim, iters,
        pso_config.get("w", 0.7),
        pso_config.get("c1", 1.7),
        pso_config.get("c2", 1.7)
    )
    p_best_pos = list(map(float, P.get_Gbest()))
    p_best_val = f_ackley(p_best_pos)

    # Devolve resultados e posterior escrita em CSV
    return {
        "Teste": test_name,
        "dim": dim, "n": n, "iters": iters, "lb": lb, "ub": ub,
        "GSA_best_pos": g_best_pos, "GSA_best_val": g_best_val,
        "PSO_best_pos": p_best_pos, "PSO_best_val": p_best_val,
        "PSO_w": pso_config.get("w", 0.7),
        "PSO_c1": pso_config.get("c1", 1.7),
        "PSO_c2": pso_config.get("c2", 1.7),
    }

# Execução principal do ficheiro
if __name__ == "__main__":
    rows = []

    # ------------------------------------------------------
    # 1) Testes padrão (2D e 3D)
    # Servem como benchmark principal apresentado no relatório.
    # ------------------------------------------------------
    for d in (2, 3):
        r = run_ackley(
            dim=d,
            iters=200,
            lb=-5.0, ub=5.0,
            pso_params={"w": 0.7, "c1": 1.7, "c2": 1.7},
            test_name=f"Padrão ({d}D)"
        )
        rows.append(r)

    # ------------------------------------------------------
    # 2) Testes de sensibilidade do PSO
    # Mostram como o parâmetro W afecta a exploração/convergência.
    # ------------------------------------------------------

    # W elevado → maior exploração (muita velocidde → dificulta movimentos pequenos perto do ótimo)
    # pode dificultar convergência fina (ajuste preciso da solução).  
    rows.append(run_ackley(
        dim=2,
        iters=200,
        lb=-5.0, ub=5.0,
        pso_params={"w": 1.0, "c1": 1.7, "c2": 1.7},
        test_name="Alta Inércia (W=1.0)"
    ))

    # W baixo → menos velocidade → exploram pouco → podem ficar presas
    # em mínimos locais, porque não ganham impulso suficiente para escapar.
    rows.append(run_ackley(
        dim=2,
        iters=200,
        lb=-5.0, ub=5.0,
        pso_params={"w": 0.4, "c1": 1.7, "c2": 1.7},
        test_name="Baixa Inércia (W=0.4)"
    ))

    # Impressão dos resultados no terminal
    for r in rows:
        print(f"[{r['Teste']}] Ackley {r['dim']}D → "
              f"W={r['PSO_w']} | "
              f"PSO f={r['PSO_best_val']:.6e} | "
              f"GSA f={r['GSA_best_val']:.6e}")

    # ------------------------------------------------------
    # Guardar resultados detalhados em CSV
    # ------------------------------------------------------
    out_csv = RESULTS / "ackley_results.csv"

    fieldnames_full = [
        "Teste", "dim", "n", "iters", "lb", "ub",
        "PSO_w", "PSO_c1", "PSO_c2",
        "GSA_best_val", "PSO_best_val",
        "GSA_best_pos", "PSO_best_pos",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_full)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResultados completos guardados em: {out_csv}")
