import pulp
import networkx as nx

# Centros
centros = ["CCJE", "CCMN", "CCS", "CFCH", "CLA", "CT", "Macaé", "Duque", "CFP", "CH", "FCC"]

# Matriz de custos (em km)
distancias = [
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
]

# Matriz de necessidades de tráfego (em Gbps)
trafego = [
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
]

# Diagnóstico: Soma das demandas totais
total_traffic = sum(sum(row) for row in trafego)
print(f"Total de tráfego necessário: {total_traffic} Gbps")

# Capacidade total da rede
C = max(2 * total_traffic, 1000)  # Dobrar o tráfego total ou usar 1000 Gbps como mínimo
print(f"Capacidade total definida: {C} Gbps")

# Parâmetros gerais
M = 10 * max(max(row) for row in trafego)  # Grande o suficiente para todos os fluxos

# Criar as arestas do grafo
edges = [(i, j) for i in range(len(centros)) for j in range(len(centros)) if i < j]

# Verificar conectividade do grafo inicial
grafo = nx.Graph()
grafo.add_nodes_from(range(len(centros)))
grafo.add_edges_from(edges)

conexo = nx.is_connected(grafo)
print(f"Grafo inicial é conexo: {conexo}")
if not conexo:
    print("O grafo inicial não é conexo. Ajuste as arestas possíveis.")
    exit()

# Criar o problema de otimização
problem = pulp.LpProblem("Minimize_Cost_with_Flow", pulp.LpMinimize)

# Variáveis de decisão
x = pulp.LpVariable.dicts("x", edges, cat="Binary")
f = pulp.LpVariable.dicts("f", [(i, j, u, v) for i, j in edges for u in range(len(centros)) for v in range(len(centros)) if u != v], lowBound=0, cat="Continuous")

# Função objetivo: Minimizar o custo total das conexões
problem += pulp.lpSum(x[(i, j)] * distancias[i][j] for i, j in edges)

# Restrição de capacidade total
problem += pulp.lpSum(f[(i, j, u, v)] for i, j in edges for u in range(len(centros)) for v in range(len(centros)) if u != v) <= C

# Ativação de conexões
for i, j in edges:
    for u in range(len(centros)):
        for v in range(len(centros)):
            if u != v:
                problem += f[(i, j, u, v)] <= M * x[(i, j)]

# Conservação de fluxo (opcional para depuração)
apply_flux_conservation = True  # Mude para False para testar sem conservação de fluxo

if apply_flux_conservation:
    for u in range(len(centros)):
        for v in range(len(centros)):
            if u != v:
                for i in range(len(centros)):
                    fluxo_saida = pulp.lpSum(f[(min(i, j), max(i, j), u, v)] for j in range(len(centros)) if (min(i, j), max(i, j)) in edges)
                    fluxo_entrada = pulp.lpSum(f[(min(j, i), max(j, i), u, v)] for j in range(len(centros)) if (min(j, i), max(j, i)) in edges)

                    # Conservação de fluxo
                    problem += fluxo_saida - fluxo_entrada == (trafego[u][v] if i == u else -trafego[u][v] if i == v else 0)

# Resolver o problema
problem.solve()

# Resultado
print(f"Status: {pulp.LpStatus[problem.status]}")
if pulp.LpStatus[problem.status] == "Optimal":
    print(f"Custo total: {pulp.value(problem.objective)}")
    # Conexões ativas
    for i, j in edges:
        if x[(i, j)].value() == 1:
            print(f"Conexão ativa: {centros[i]} - {centros[j]}")
else:
    print("O problema é inviável. Verifique as restrições ou os parâmetros.")
