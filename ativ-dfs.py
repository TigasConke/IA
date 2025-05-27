def dfs(grafo, inicio, objetivo):
    """
    Implementação do algoritmo de Busca em Profundidade (DFS).
    Diferente do BFS, ele explora primeiro um caminho até o final antes de voltar e explorar outros caminhos.
    """

    pilha = [inicio]  # Criamos uma pilha para armazenar os nós a serem explorados
    visitados = set()  # Criamos um conjunto para armazenar os nós já visitados

    # Enquanto houver elementos na pilha, continuamos a busca
    while pilha:
        # Pegamos o último nó da pilha (LIFO - Last In, First Out)
        no = pilha.pop()
        print(f"Visitando: {no}")  # Exibimos o nó que está sendo processado

        # Se encontramos o objetivo, finalizamos a busca
        if no == objetivo:
            print("Objetivo encontrado!")
            return True

        # Marcamos o nó como visitado para evitar repetições
        visitados.add(no)

        # Pegamos os vizinhos do nó atual e invertemos a ordem
        # O motivo da inversão é garantir que os nós sejam visitados na ordem esperada
        for vizinho in reversed(grafo.get(no, [])):
            if vizinho not in visitados:
                pilha.append(vizinho)  # Adicionamos o vizinho à pilha para ser explorado depois

    # Se terminarmos a busca sem encontrar o objetivo, imprimimos essa informação
    print("Objetivo não encontrado!")
    return False  # Retornamos False para indicar que o nó objetivo não foi encontrado

# Exemplo de grafo representado por um dicionário
grafo = {
    1: [4],
    4: [9, 6],
    6: [],
    9: [20],
    20: [15,30],
    15: [],
    30: []
}

# Chamamos a função DFS para buscar o nó 'E' começando a partir do nó 'A'
dfs(grafo, 1, 9)
