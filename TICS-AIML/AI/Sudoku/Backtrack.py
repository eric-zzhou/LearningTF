board = [
    [3, 4, 1, 0],
    [0, 2, 0, 0],
    [0, 0, 2, 0],
    [0, 1, 4, 3]
]

SUDOKU_SIZE = len(board)


def construct_board(board):
    b = []
    for i in range(SUDOKU_SIZE):
        b.append([])
    for i in range(SUDOKU_SIZE):
        for j in range(SUDOKU_SIZE):
            cur = board[i][j]
            if cur != 0:
                b[i].append(cur)
            else:
                b[i].append([])
                for k in range(len(SUDOKU_SIZE)):
                    b[i][j].append(k + 1)
    return b


def constraint_check(board):
    for i in range(SUDOKU_SIZE):
        exists = []
        for j in range(SUDOKU_SIZE):
            exists.append(False)
        for j in range(SUDOKU_SIZE):
            cur = board[i][j]
            if isinstance(cur, int):
                if exists[cur - 1]:
                    return False
                exists[cur - 1] = True
    for i in range(SUDOKU_SIZE):
        exists = []
        for j in range(SUDOKU_SIZE):
            exists.append(False)
        for j in range(SUDOKU_SIZE):
            cur = board[j][i]
            if isinstance(cur, int):
                if exists[cur - 1]:
                    return False
                exists[cur - 1] = True

    # TODO: fix this portion to check 3x3 squares
    for i in range(SUDOKU_SIZE):
        exists = []
        for j in range(SUDOKU_SIZE):
            exists.append(False)
        for j in range(SUDOKU_SIZE):
            cur = board[j][i]
            if isinstance(cur, int):
                if exists[cur - 1]:
                    return False
                exists[cur - 1] = True
    return True

'''
I will write the description of the rest of it here:
After creating the board, loop to the first empty slot and check by using isinstance(cur, list)
Implement a Queue with a list
Add all the possible new board layouts that do not cause immediate conflicts
Continue
If there is a conflict, pop a new layout and try it out
'''
