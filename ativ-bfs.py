
from collections import deque

def bfs(grafo, inicio, objetivo):
    """
    Implementação do algoritmo de Busca em Largura (BFS).
    Ele explora todos os nós vizinhos primeiro antes de avançar para os próximos níveis.
    """

    fila = deque([inicio])

    visitados = set()

  
    while fila:

        no = fila.popleft()
        print(f"Visitando: {no}")  

        if no == objetivo:
            print("Objetivo encontrado!")
            return True

        visitados.add(no)

        for vizinho in grafo.get(no, []):

            if vizinho not in visitados:
                fila.append(vizinho)


    print("Objetivo não encontrado!")
    return False  

grafo = {
    1: [4],
    4: [9, 6],
    6: [],
    9: [20],
    20: [15,30],
    15: [],
    30: []
}

bfs(grafo, 1, 30)