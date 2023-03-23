# A default graph for testing provided by Ms. Lewellen
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}


# BFS
def bfs(graph, start, goal):
    # Adding starting node
    visited = [start]
    fringe = [start]

    # Looping until there's no more
    while fringe:
        # FIFO
        current = fringe.pop(0)
        print(current)

        if current is goal:
            # End function if found
            return
        else:
            # Adds neighbors if they have not been visited yet
            for neighbor in graph[current]:
                if neighbor not in visited:
                    fringe.append(neighbor)
                    visited.append(neighbor)
    # If function has not returned yet, the node does not exist
    print("Node does not exist")


# DFS
def dfs(graph, start, goal):
    # Adding starting node
    visited = [start]
    fringe = [start]

    # Looping until there's no more
    while fringe:
        # FIFO
        current = fringe.pop()
        print(current)

        if current is goal:
            # End function if found
            return
        else:
            # Adds neighbors if they have not been visited yet
            for neighbor in graph[current]:
                if neighbor not in visited:
                    fringe.append(neighbor)
                    visited.append(neighbor)
    # If function has not returned yet, the node does not exist
    print("Node does not exist")


# Testing code
bfs(graph, 'A', 'F')
print("\n\n")
dfs(graph, 'A', 'H')
