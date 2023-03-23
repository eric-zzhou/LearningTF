# from queue import PriorityQueue
#
#
# # Input is the start state of the board
# def greedy(state):
#     # List to keep track of all visited states to not repeat
#     visited = []
#     # Create fringe to keep track of possible options
#     # Difference: Uses PriorityQueue instead of basic Stack or Queue implemented on a list
#     fringe = PriorityQueue()  # Initialize a queue
#
#     # Adds the start state to fringe as a tuple, sorting by heuristic and keeping the current state with it
#     # Heuristic is used to predict how close the current state is to the goal
#     # Difference: Uses a heuristic for sorting rather than following a specific preset order (LIFO AND FIFO)
#     fringe.put((heuristic_v1(state), state))
#
#     # While there are still options
#     while not fringe.empty():
#         # Getting the next best state
#         next_best = fringe.get()
#         current = next_best[1]
#
#         # Adding current state to visited to avoid visiting it again
#         visited.append(flatten_board(current))
#
#         # Print current to see it
#         pretty_print(current)
#
#         # Checks to see if goal is reached, return if yes
#         if goal_check(current):
#             print("done!")
#             return
#
#         # Adds all the possible next moves to the fringe if it has not already been checked
#         for next_move in get_next_moves(current):
#             if flatten_board(next_move) not in visited:
#                 ''' Difference: Added as a tuple with the heuristic for sorting to determine how close it is to the goal
#                 state and what the best next move is'''
#                 fringe.put((heuristic_v1(next_move), next_move))
