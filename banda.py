import pulp
import numpy as np
from itertools import combinations
import networkx as nx
from collections import defaultdict, deque
import matplotlib.pyplot as plt

# 1. Data Preparation

# Matriz de distâncias (km)
distancias = np.array([
    [0, 20, 20, 0.5, 20, 20, 190, 40, 20, 20, 0.5],
    [20, 0, 1, 20, 1, 0.5, 190, 40, 0.5, 0.5, 20],
    [20, 1, 0, 20, 1, 0.5, 190, 40, 0.5, 0.5, 20],
    [0.5, 20, 20, 0, 20, 20, 190, 40, 20, 20, 0.5],
    [20, 1, 1, 20, 0, 1, 190, 40, 1, 1, 20],
    [20, 0.5, 0.5, 20, 1, 0, 190, 40, 0.5, 0.5, 20],
    [190, 190, 190, 190, 190, 190, 0, 160, 190, 190, 190],
    [40, 40, 40, 40, 40, 40, 160, 0, 40, 40, 40],
    [20, 0.5, 0.5, 20, 1, 0.5, 190, 40, 0, 0.5, 20],
    [20, 0.5, 0.5, 20, 1, 0.5, 190, 40, 0.5, 0, 20],
    [0.5, 20, 20, 0.5, 20, 20, 190, 40, 20, 20, 0]
])

# Matriz de tráfego máximo (Gbps)
trafego = np.array([
    [0, 10, 5, 2, 15, 8, 3, 4, 7, 10, 5],
    [10, 0, 20, 6, 10, 25, 8, 6, 12, 15, 10],
    [5, 20, 0, 8, 10, 15, 6, 7, 8, 20, 10],
    [2, 6, 8, 0, 10, 12, 5, 4, 6, 10, 8],
    [15, 10, 10, 10, 0, 18, 7, 5, 9, 14, 10],
    [8, 25, 15, 12, 18, 0, 20, 15, 18, 25, 20],
    [3, 8, 6, 5, 7, 20, 0, 10, 12, 18, 15],
    [4, 6, 7, 4, 5, 15, 10, 0, 10, 12, 10],
    [7, 12, 8, 6, 9, 18, 12, 10, 0, 15, 12],
    [10, 15, 20, 10, 14, 25, 18, 12, 15, 0, 18],
    [5, 10, 10, 8, 10, 20, 15, 10, 12, 18, 0]
])

num_centros = len(distancias)

# Lista de todas as possíveis conexões (i < j)
conexoes = list(combinations(range(num_centros), 2))

# Lista de todos os pares de tráfego (k, l)
pares_trafego = [(i, j) for i in range(num_centros) for j in range(num_centros) if i != j and trafego[i][j] > 0]

# 2. Funções Auxiliares

def verificar_conectividade(conexoes_estabelecidas, num_centros):
    grafo = defaultdict(list)
    for (i, j) in conexoes_estabelecidas:
        grafo[i].append(j)
        grafo[j].append(i)

    visitados = set()
    fila = deque([0])  # Iniciar a busca a partir do nó 0
    visitados.add(0)

    while fila:
        atual = fila.popleft()
        for vizinho in grafo[atual]:
            if vizinho not in visitados:
                visitados.add(vizinho)
                fila.append(vizinho)

    return len(visitados) == num_centros

def visualizar_rede(conexoes_estabelecidas, distancias, num_centros):
    G_rede = nx.Graph()
    for (i, j) in conexoes_estabelecidas:
        G_rede.add_edge(i, j, weight=distancias[i][j])

    pos = nx.spring_layout(G_rede, seed=42)  # Layout para visualização
    weights = [G_rede[u][v]['weight'] for u, v in G_rede.edges()]

    plt.figure(figsize=(12, 8))
    nx.draw(G_rede, pos, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray')
    edge_labels = {(u, v): f"{G_rede[u][v]['weight']} km" for u, v in G_rede.edges()}
    nx.draw_networkx_edge_labels(G_rede, pos, edge_labels=edge_labels, font_color='red')
    plt.title("Rede Otimizada")
    plt.show()

# 3. Encontrar a Árvore Geradora Mínima (AGM)

# Criar o grafo completo com as distâncias como pesos
G = nx.Graph()
for i in range(num_centros):
    for j in range(i+1, num_centros):
        G.add_edge(i, j, weight=distancias[i][j])

AGM = nx.minimum_spanning_tree(G, weight='weight')
arestas_agm = list(AGM.edges())

# 4. Implementação do Algoritmo Iterativo

# Lista de todas as arestas ordenadas por distância (crescente)
arestas_sorted = sorted(conexoes, key=lambda x: distancias[x[0], x[1]])

# Inicializar a lista de arestas selecionadas com a AGM
conexoes_selecionadas = arestas_agm.copy()

# Remover as arestas já na AGM da lista ordenada
arestas_restantes = [edge for edge in arestas_sorted if edge not in conexoes_selecionadas]

# Definir a capacidade inicial para as arestas da AGM
# Começamos com capacidades insuficientes para forçar o algoritmo a adicionar arestas
capacidade_dict = {}
for (i, j) in conexoes_selecionadas:
    capacidade_dict[(i, j)] = 0  # Capacidade inicial zero
    capacidade_dict[(j, i)] = 0  # Bidirecional

# Loop iterativo para adicionar arestas e aumentar capacidades
while True:
    print(f"\nTentando com {len(conexoes_selecionadas)} arestas selecionadas...")

    # Resolver o problema de fluxo
    prob = pulp.LpProblem("Fluxo_Multi_Commodities", pulp.LpMinimize)

    # Variáveis de fluxo: f_k_l_i_j para cada par (k,l) e cada aresta (i,j)
    f = {}
    for (k, l) in pares_trafego:
        for (i, j) in conexoes_selecionadas:
            f[(k, l, i, j)] = pulp.LpVariable(f"f_{k}_{l}_{i}_{j}", lowBound=0, cat=pulp.LpContinuous)
            f[(k, l, j, i)] = pulp.LpVariable(f"f_{k}_{l}_{j}_{i}", lowBound=0, cat=pulp.LpContinuous)

    # Variáveis de capacidade: c_i_j para cada aresta (i, j)
    c = {}
    for (i, j) in conexoes_selecionadas:
        c[(i, j)] = pulp.LpVariable(f"c_{i}_{j}", lowBound=0, cat=pulp.LpContinuous)
        c[(j, i)] = c[(i, j)]  # Capacidade bidirecional

    # Função Objetivo: minimizar o custo total das capacidades
    prob += pulp.lpSum(distancias[i][j] * c[(i, j)] for (i, j) in conexoes_selecionadas), "Custo_Total"

    # Restrições de conservação de fluxo para cada par (k, l)
    for (k, l) in pares_trafego:
        for m in range(num_centros):
            inflow = pulp.lpSum(f[(k, l, i, m)] for i in range(num_centros) if ((i, m) in conexoes_selecionadas or (m, i) in conexoes_selecionadas))
            outflow = pulp.lpSum(f[(k, l, m, j)] for j in range(num_centros) if ((m, j) in conexoes_selecionadas or (j, m) in conexoes_selecionadas))

            if m == k:
                prob += (outflow - inflow) == trafego[k][l], f"Fluxo_Origem_{k}_{l}_no_{m}"
            elif m == l:
                prob += (inflow - outflow) == trafego[k][l], f"Fluxo_Destino_{k}_{l}_no_{m}"
            else:
                prob += (outflow - inflow) == 0, f"Fluxo_Intermediario_{k}_{l}_no_{m}"

    # Restrições de capacidade das arestas
    for (i, j) in conexoes_selecionadas:
        total_flux = pulp.lpSum(f[(k, l, i, j)] + f[(k, l, j, i)] for (k, l) in pares_trafego)
        prob += total_flux <= c[(i, j)], f"Capacidade_Aresta_{i}_{j}"

    # Capacidade mínima inicial (pode ser zero ou um valor mínimo)
    for (i, j) in conexoes_selecionadas:
        prob += c[(i, j)] >= capacidade_dict.get((i, j), 0), f"Capacidade_Minima_{i}_{j}"

    # Resolver o problema
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[status] == 'Optimal':
        print("Solução viável encontrada!")
        break
    else:
        if not arestas_restantes:
            print("Não foi possível encontrar uma solução viável mesmo adicionando todas as arestas.")
            break
        # Adicionar a próxima aresta com menor distância
        proxima_aresta = arestas_restantes.pop(0)
        conexoes_selecionadas.append(proxima_aresta)
        print(f"Aresta adicionada: {proxima_aresta}, Distância: {distancias[proxima_aresta[0], proxima_aresta[1]]} km")
        # Inicializar a capacidade desta aresta como zero
        capacidade_dict[(proxima_aresta[0], proxima_aresta[1])] = 0
        capacidade_dict[(proxima_aresta[1], proxima_aresta[0])] = 0

# 5. Exibir os Resultados

if pulp.LpStatus[status] == 'Optimal':
    # Exibir as arestas selecionadas e suas capacidades
    print("\nArestas selecionadas para a rede com suas capacidades:")
    total_custo = 0
    for (i, j) in conexoes_selecionadas:
        capacidade = pulp.value(c[(i, j)])
        custo = distancias[i][j] * capacidade
        total_custo += custo
        print(f"Aresta: {i} - {j}, Distância: {distancias[i][j]} km, Capacidade: {capacidade} Gbps, Custo: {custo}")

    print(f"\nCusto Total da Rede: {total_custo} unidades monetárias")

    # Verificar a conectividade final
    if verificar_conectividade(conexoes_selecionadas, num_centros):
        print("A rede está conectada.")
    else:
        print("A rede está desconectada.")

    # Visualizar a rede (opcional)
    visualizar_rede(conexoes_selecionadas, distancias, num_centros)
else:
    print("Não foi possível encontrar uma solução viável.")

# 6. Análise dos Fluxos (Opcional)

# Exibir os fluxos nas arestas (se necessário)
if pulp.LpStatus[status] == 'Optimal':
    print("\nFluxos nas arestas:")
    for (i, j) in conexoes_selecionadas:
        fluxo_total = sum(pulp.value(f[(k, l, i, j)]) + pulp.value(f[(k, l, j, i)]) for (k, l) in pares_trafego)
        print(f"Aresta: {i} - {j}, Fluxo Total: {fluxo_total} Gbps, Capacidade: {pulp.value(c[(i, j)])} Gbps")
