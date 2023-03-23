# Template Provided by: Katelin Lewellen
# EPS TiCS: AIML
# December 2021
#
# This file generates a random 8-square board (solvable)
# and then uses an A* algorithms to solve it.
# this import brings in the random number generation
# we use this to create random board
# Generate a number between x (inclusive) and y (exclusive) with
# the function random.randrange(x,y)
import random
# this import brings in the ability to different copy functions
# in this case, it allows you to have a deep copy (which works on list of lists)
# the function is dest = copy.deepcopy(source_array)
# where dest is where you are copying to
# and source_array is where you are copying from
import copy
# this import brings in a priority queue to use for the fringe
# we can construct an empty PQ by saying
# fringe = PriorityQueue()
#
# we can add weighted items to the priority queue with the function
# fringe.put( (total_cost, cost, state) )
# where weight is the cost plus the heuristic value, cost is the backwards cost,
# and state is the state to add
#
# we can get the lowest-weight item out of the fringe by using the function
# next_best = fringe.get()
# which returns a tuple of the form (total_cost, cost, state)
# to get individual elements out, index them as you would an array
# cost = next_best(1)
# current = next_best(2)
from queue import PriorityQueue

# Define the end state: our goal
end_state = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
# Defines an empty board for easy copying and filling in.
empty = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]


# function prints out a board (state) in a nice format
# 8 5 3
# 2 1 _
# 7 4 6
# with an empty line at the end, for visibility
def pretty_print(board):
    for row in board:
        for space in row:
            if space == 0:
                print("_", end=" ")
            else:
                print(space, end=" ")
        print()
    print()


# takes a 2d list and "flattens" it down to a 1d list
# by adding each element to a new list
# used to allow shallow compares to work well
# I suggest flattening boards before adding them to visited
def flatten_board(board):
    flat = []
    for row in board:
        for space in row:
            flat.append(space)
    return flat


# randomly places the numbers 1 to 8 in unoccupied positions on the board
# by randomly generating a row and column
# if the place is already occupied, generates again.
# the last remaining spot holds a 0 and will be our blank
def randomize_board(board):
    placed = 0
    while placed < 8:
        row = random.randrange(0, 3)
        col = random.randrange(0, 3)
        if board[row][col] == 0:
            placed += 1
            board[row][col] = placed


# not all 8square problems are solvable - we can determine solvability by
# checking the number of inversions in the board
# an inversion is where a number is out of sorted order in the array
# (i.e. greater and earlier than another element)
# if the number of inversions is even, it is solvable.
def is_solvable(board):
    arr = flatten_board(board)
    inv_count = 0
    for index in range(0, 9):
        for cmp_index in range(index + 1, 9):
            if arr[index] != 0 and \
                    arr[cmp_index] != 0 and \
                    arr[index] > arr[cmp_index]:
                inv_count += 1
    return (inv_count % 2 == 0)


# function to generate random boards until one is generated that is solvable
def gen_solvable():
    board = []
    solvable = False
    while not solvable:
        board = copy.deepcopy(empty)
        randomize_board(board)
        solvable = is_solvable(board)
    return board


# a long function used to generate a list of next possible moves after state
# first searches to find the location of the 0 (the blank)
# then manually generates states for the each of the 4 sliding directions:
# a state, if the blank moves up, down, right, and left.
def get_next_moves(state):
    row_index = 0  # holds the row of the blank
    col_index = 0  # holds the col of the blank
    # find the blank
    for row in range(0, 3):
        for col in range(0, 3):
            if state[row][col] == 0:
                row_index = row
                col_index = col
    # List to store potential future states.
    next_moves = []
    # for each of the following, find the appropriate next step
    # when applicable and not at a border
    # by swapping the zero (blank) with a neighbor spot in the current state
    # in a copy of the current state
    # move up
    if row_index > 0:
        up_state = copy.deepcopy(state)
        up_state[row_index][col_index] = state[row_index - 1][col_index]
        up_state[row_index - 1][col_index] = 0
        next_moves.append(up_state)
    # move down
    if row_index < 2:
        down_state = copy.deepcopy(state)
        down_state[row_index][col_index] = state[row_index + 1][col_index]
        down_state[row_index + 1][col_index] = 0
        next_moves.append(down_state)
    # move left
    if col_index > 0:
        left_state = copy.deepcopy(state)
        left_state[row_index][col_index] = state[row_index][col_index - 1]
        left_state[row_index][col_index - 1] = 0
        next_moves.append(left_state)
    # move right
    if col_index < 2:
        right_state = copy.deepcopy(state)
        right_state[row_index][col_index] = state[row_index][col_index + 1]
        right_state[row_index][col_index + 1] = 0
        next_moves.append(right_state)
    return next_moves


# a function that takes in a state and returns an integer value indicating
# the estimated distance this state is from the end
# Possible heuristics:
#  - number of items out of place
#  - number of items out of row + number of items out of column
#  - sum of the manhattan distances of an item to its location
#  - sum of euclidean distances or an item to its location
def heuristic(state):
    cost = 0

    # Loops through the entire matrix
    for i in range(3):
        for j in range(3):
            # Adds Manhattan Distance
            current = state[i][j]
            if current != 0:
                # Vertical error
                cost += abs(i - (current // 3))
                # Horizontal error
                cost += abs(j - (current % 3))
    return cost


# a function that returns True if the goal has been met and we are in the
# goal state, and returns False otherwise.
def goal_check(state):
    # flattens board into array
    arr = flatten_board(state)

    # loops through array to make sure all numbers match, return False if not
    for x in range(9):
        if arr[x] is not x:
            return False

    # returns true if everything matched
    return True


# the actual algorithm necessary to perform A* search.
# reminder - should look very similar to your BFS (for example) but with a few
# minor changes:
# - use a priority queue instead of a list to hold the fringe
# - things added to the fringe have a weight
# - that weight is the cost so far + the heurisitic cost
# - use goal_check, rather than checking == goal
# - instead of looping through all neighbors you want to loop over all the
#     possible next states (where states is generated by get_next_moves)
#               for next_move in states:
# remember that you want to enter tuples weighted by total cost H(x)+cost
# but you also need the current cost so you know the cost of the neighbor node
# which is cost+1
def a_star(state):
    print("start state:")
    pretty_print(state)
    # adding starting node
    fringe = PriorityQueue()
    visited = []
    fringe.put((0, 0, state))

    # looping until there's no more
    while fringe.qsize() > 0:
        # getting current state and board
        current_state = fringe.get()
        current_board = current_state[2]
        print("current:")
        pretty_print(current_board)

        # add to visited
        visited.append(flatten_board(current_board))

        # goal check
        if goal_check(current_board):
            # end function if found
            return
        else:
            # adds next moves if they have not been visited yet
            possible_moves = get_next_moves(current_board)
            for move in possible_moves:
                if flatten_board(move) not in visited:
                    # calculates total predicted cost and total backward cost based on old backward cost
                    backward_cost = current_state[1] + 1
                    total_cost = backward_cost + heuristic(move)
                    fringe.put((total_cost, backward_cost, move))


board = gen_solvable()
a_star(board)
