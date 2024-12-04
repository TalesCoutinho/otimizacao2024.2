import pulp
import numpy as np

# Dados do problema
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

# Função auxiliar para acessar as variáveis de decisão
def acessar_x(x, i, j):
    if i < j:
        return x[(i, j)]
    else:
        return x[(j, i)]

# Resolver grafo com banda mínima
def resolver_com_banda_minima():
    print("Modelando o problema com restrições de tráfego mínimo.")
    prob = pulp.LpProblem("Grafo_Conexo_Banda", pulp.LpMinimize)

    # Variáveis de decisão
    print("Definindo variáveis de decisão para conexões...")
    x = pulp.LpVariable.dicts("x", [(i, j) for i in range(num_centros) for j in range(i + 1, num_centros)], 
                              cat=pulp.LpBinary)
    print("Definindo variáveis de fluxo de tráfego...")
    f = pulp.LpVariable.dicts("f", [(i, j) for i in range(num_centros) for j in range(num_centros) if i != j], 
                              lowBound=0, cat=pulp.LpContinuous)

    # Função objetivo
    print("Definindo a função objetivo...")
    prob += pulp.lpSum(distancias[i, j] * x[(i, j)] for i in range(num_centros) for j in range(i + 1, num_centros))

    # Restrições de conectividade
    print("Adicionando restrições de conectividade...")
    prob += pulp.lpSum(x[(i, j)] for i in range(num_centros) for j in range(i + 1, num_centros)) == num_centros - 1

    # Restrição de tráfego mínimo
    print("Adicionando restrições de tráfego mínimo para cada par de centros...")
    for i in range(num_centros):
        for j in range(num_centros):
            if i != j:
                prob += f[(i, j)] >= trafego[i, j]  # Garantir banda mínima
                prob += f[(i, j)] <= pulp.lpSum(acessar_x(x, i, k) for k in range(num_centros) if k != i)  # Limitar pelo grafo

    # Restrições de tráfego pelas conexões
    print("Adicionando restrições de fluxo através das conexões existentes...")
    for i in range(num_centros):
        for j in range(num_centros):
            if i != j:
                prob += f[(i, j)] <= pulp.lpSum(acessar_x(x, i, k) * trafego[k, j] for k in range(num_centros) if k != i)

    # Resolvendo o problema
    print("Resolvendo o problema...")
    prob.solve(pulp.PULP_CBC_CMD(msg=True))

    return prob, x, f

# Resolver problema
modelo, conexoes, fluxos = resolver_com_banda_minima()

# Exibir resultados
print("\nConexões selecionadas:")
for (i, j) in conexoes:
    if conexoes[(i, j)].varValue > 0.5:
        print(f"Conexão: {i} - {j}, Distância: {distancias[i, j]} km")

print("\nTráfego entre os centros:")
for (i, j) in fluxos:
    if fluxos[(i, j)].varValue > 0:
        print(f"Tráfego: {i} -> {j}, Banda: {fluxos[(i, j)].varValue} Gbps")

print(f"\nCusto Total: {pulp.value(modelo.objective)} km")

