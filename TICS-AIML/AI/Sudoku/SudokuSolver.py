# start_board = [
#     [7, 4, 0, 0, 3, 0, 0, 1, 0],
#     [0, 1, 9, 0, 6, 8, 5, 0, 2],
#     [0, 0, 0, 0, 0, 4, 3, 0, 0],
#     [0, 5, 6, 3, 7, 0, 0, 0, 1],
#     [0, 0, 1, 8, 0, 0, 0, 9, 5],
#     [0, 9, 0, 0, 2, 0, 6, 0, 0],
#     [1, 0, 3, 4, 0, 7, 2, 0, 0],
#     [5, 0, 0, 2, 0, 0, 0, 0, 8],
#     [0, 8, 0, 0, 0, 1, 4, 7, 0]
# ]

start_board = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
]

SUDOKU_SIZE = len(start_board)


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
                b[i].append([1, 2, 3, 4, 5, 6, 7, 8, 9])
    return b


def constraint_check(board):
    for i in range(SUDOKU_SIZE):
        pass


board = construct_board(start_board)
for i in board:
    print(str(i).replace('[1, 2, 3, 4, 5, 6, 7, 8, 9]', 'a'))
