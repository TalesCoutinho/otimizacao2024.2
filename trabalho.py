import pulp
import pandas as pd
import networkx as nx

# Centros e distâncias (em km) - tabela de distâncias
centros = ["CCJE", "CCMN", "CCS", "CFCH", "CLA", "CT", "CM UFRJ-Macaé", "Campus Duque de Caxias", "CFP", "CH", "FCC"]
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

# Criar o problema de otimização
problem = pulp.LpProblem("Minimize_Cost", pulp.LpMinimize)

# Lista de arestas e seus custos
edges = [
    (centros[i], centros[j], distancias[i][j])
    for i in range(len(centros))
    for j in range(i + 1, len(centros))
]

# Variáveis binárias indicando se uma conexão é usada
x = pulp.LpVariable.dicts("x", edges, cat="Binary")

# Função objetivo: minimizar o custo total das conexões
problem += pulp.lpSum(x[(u, v, c)] * c for (u, v, c) in edges)

# Restrição: exatamente len(centros) - 1 arestas devem ser escolhidas
problem += pulp.lpSum(x[(u, v, c)] for (u, v, c) in edges) == len(centros) - 1

# Garantir que cada prédio está conectado a pelo menos um outro
for centro in centros:
    problem += pulp.lpSum(
        x[(u, v, c)] for (u, v, c) in edges if u == centro or v == centro
    ) >= 1

# Subtour elimination constraints (planos de corte) - iterativo
while True:
    # Resolver o problema relaxado
    problem.solve()

    # Construir o grafo resultante
    G = nx.Graph()
    for (u, v, c) in edges:
        if x[(u, v, c)].value() == 1:
            G.add_edge(u, v)

    # Verificar conectividade do grafo
    if nx.is_connected(G):
        break

    # Adicionar restrições para subconjuntos desconexos
    for component in nx.connected_components(G):
        if len(component) < len(centros):
            # Componentes que não conectam todos os vértices
            cut = [(u, v, c) for (u, v, c) in edges if u in component and v not in component]
            problem += pulp.lpSum(x[(u, v, c)] for (u, v, c) in cut) >= 1

# Coletar resultados
selected_edges = [(u, v, c) for (u, v, c) in edges if x[(u, v, c)].value() == 1]
total_cost = pulp.value(problem.objective)

# Criar tabela final com pandas
df = pd.DataFrame(selected_edges, columns=["Prédio 1", "Prédio 2", "Custo (km)"])

# Exibir resultados
print("\nTabela de Conexões Selecionadas:")
print(df)

print(f"\nCusto Total: {total_cost} km")
print(f"O grafo gerado é conexo: {nx.is_connected(G)}")
