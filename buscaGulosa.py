import heapq

def greedy_best_first_search(graph, start, goal, heuristic):
    frontier = []
    heapq.heappush(frontier, (heuristic[start], start))

    came_from = {start: None}

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for neighbor, _ in graph[current]:
            if neighbor not in came_from:
                came_from[neighbor] = current
                heapq.heappush(frontier, (heuristic[neighbor], neighbor))
                came_from[neighbor] = current

    return None

graph = {
    'A': [('B', 10), ('D', 15)],
    'B': [('C', 12), ('E', 10)],
    'C': [('F', 15)],
    'D': [('E', 5)],
    'E': [('F', 10)],
    'F': [('G', 8)],
    'G': []
}

heuristic = {
    'A': 25, 'B': 18, 'C': 12,
    'D': 20, 'E': 8, 'F': 5, 'G': 0
}

start = 'A'
goal = 'G'

path = greedy_best_first_search(graph, start, goal, heuristic)
print("Caminho encontrado:", path)