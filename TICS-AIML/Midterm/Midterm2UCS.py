from queue import PriorityQueue


def ucs(graph, start, goal):
    visited = []
    fringe = PriorityQueue()

    current = start
    current_cost = 0
    fringe.put((current_cost, current))

    # Adding start to visited since we don't want to check it again
    visited.append(start)

    while not fringe.empty():
        next_best = fringe.get()
        current = next_best[1]

        # Print current so that there is some output
        print(current)

        if current == goal:
            print("done in ", current_cost, "iterations!")
            return

        # Incrementing current_cost since the cost would be 1 more every round
        # I am assuming that the cost is the same for every move since the problem doesn't specify
        current_cost += 1

        for next_move in graph[current]:
            if next_move not in visited:
                fringe.put((current_cost, next_move))

                ''' Add all the next_moves being added to the fringe to the visited as well to make sure they don't get
                added again later on '''
                visited.append(next_move)
