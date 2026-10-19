from math import exp
import numpy as np

from . import intelligence


class gsa(intelligence.sw):
    """
    Gravitational Search Algorithm (GSA)
    Implementa o modelo físico: partículas atraem-se conforme a sua "massa"
    (qualidade da solução), actualizando posição através de força → aceleração.
    """

    def __init__(self, n, function, lb, ub, dimension, iteration, G0=3):
        """
        n: número de agente (tamanho da população)
        function: função objetivo a minimizar
        lb: limite inferior do espaço de procura
        ub: limite superior do espaço de procura
        dimension: dimensão do vector posição
        iteration: número de iterações
        G0: constante gravitacional inicial (controla exploração)
        """

        # Inicializa estrutura base da classe 'sw' do SwarmPackagePy
        super(gsa, self).__init__()

        # Inicializa agentes uniformemente no espaço definido
        self.__agents = np.random.uniform(lb, ub, (n, dimension))
        
        # Guarda histórico das posições (usado para animações/visualização)
        self._points(self.__agents)

        # Pbest: melhor posição actual (da população inicial)
        Pbest = self.__agents[np.array([function(x)
                                        for x in self.__agents]).argmin()]
        # Gbest: melhor posição global até ao momento
        Gbest = Pbest

        # Velocidade inicial a zero para evitar que as partículas arranquem em
        # direções aleatórias.
        velocity = np.array([[0 for k in range(dimension)] for i in range(n)])

        # Loop principal do GSA
        for t in range(iteration):

            # Factor aleatório (gera valores aleatórios para adicionar variação
            # ao movimento das partículas, evitando que todas se movam de forma
            # igual e fiquem presas no mesmo sítio.)
            csi = np.random.random((n, dimension))
            eps = np.random.random((1, n))[0]

            # Avaliação do fitness de cada agente
            fitness = np.array([function(x) for x in self.__agents])

            # Cálculo das massas normalizadas:
            # agentes com menor fitness (melhores soluções) têm maior massa
            # Converte valores fitness em valores numéricos:
            #     melhor fitness -> valor mais baixo  
            m = np.array([(function(x) - max(fitness)) /
                       (min(fitness) - max(fitness)) for x in self.__agents])
            # Normalização: a soma das massas passa a ser 1
            # Mantém as massas proporcionais e estáveis para os cálculos seguintes
            M = np.array([i / sum(m) for i in m])

            # Actualização da constante gravitacional 
            # G diminui ao longo do tempo para começar com movimentos grandes (exploração)
            # e terminar com movimentos pequenos e precisos (convergência).
            G = G0 / exp(0.01 * t)

            # Calcula a aceleração de cada agente somando as forças dos outros.
            # Agentes com melhor solução (maior massa) puxam mais os restantes.
            # Isto é essencial para guiar o movimento do swarm e permitir a convergência.
            a = np.array([sum([eps[j] * G * M[j] *
                            (self.__agents[j] - self.__agents[i]) / (
                np.linalg.norm(self.__agents[i] - self.__agents[j]) + 0.001)
                            for j in range(n)]) for i in range(n)])

            # Atualização da velocidade:
            # Multiplicamos a velocidade pelo fator aleatório 'csi' para evitar 
            # que todas as partículas sigam exatamente o mesmo percurso. 
            # Isto introduz variabilidade no movimento e ajuda o algoritmo a 
            # escapar de mínimos locais.
            velocity = csi * velocity + np.array([a[i] for i in range(n)])
                        
            # Atualização da posição:
            # A nova posição resulta simplesmente da posição anterior mais a 
            # velocidade calculada.
            self.__agents += velocity
            
            # Impedimos que as partículas saiam dos limites definidos do espaço de procura.
            # Sem este "clipping", poderiam surgir valores inválidos ou fora do domínio desejado.
            self.__agents = np.clip(self.__agents, lb, ub)
            
            
            # Guardamos as posições atuais para permitir a visualização/análise do movimento.
            self._points(self.__agents)
            
            
            # Determina o melhor agente da população nesta iteração.
            Pbest = self.__agents[
                np.array([function(x) for x in self.__agents]).argmin()
            ]
            
            
            # Atualiza o melhor global (Gbest) se a nova melhor solução for superior.
            # Se nesta iteração encontrámos um agente com melhor fitness que o 
            # melhor de sempre, então atualizamos o melhor global.
            if function(Pbest) < function(Gbest):
                Gbest = Pbest
            
            
            # Guardamos o melhor global encontrado após todas as iterações.
            self._set_Gbest(Gbest)

