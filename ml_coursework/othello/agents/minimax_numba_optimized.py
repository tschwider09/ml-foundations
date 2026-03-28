import numpy as np
from numba import jit
from numba.typed import List
# MODIFY THIS TO CHANGE THE AI's DEPTH BOUND
DEPTHBOUND = 7
SIZE = 8

MAXINT = np.iinfo(np.int32).max
weights = [[99,-10,10,7,7,10,-10,99],
[-10,-25,-5,-3,-3,-5,-25,-10],
[10,-5,6,4,4,6,-5,10],
[7,-3,4,0,0,4,-3,7],
[7,-3,4,0,0,4,-3,7],
[10,-5,6,4,4,6,-5,10],
[-10,-25,-5,-3,-3,-5,-25,-10],
[99,-10,10,7,7,10,-10,99]]
def heuristic(board):
	node = board.board
	heur = 0
	for i in range(len(node)):
		for j in range(len(node[i])):
			heur += weights[i][j]* node[i,j]
	for i in [1,-1]:
		heur += i*20*len(board.possibleMoves(i))
	return heur

@jit(nopython=True)
def numba_checkMove(board, row, col, turn):
    # Don't import here - just use List directly
    stonesToFlip = List()
    if board[row, col] != 0:
        return stonesToFlip
    
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for d in directions:
        flank = False
        r = row + d[0]
        c = col + d[1]
        
        tempflips = List()
        
        while r >= 0 and r < 8 and c >= 0 and c < 8:
            if board[r, c] == -turn:
                flank = True
                tempflips.append((r, c))
            elif board[r, c] == turn:
                if flank:
                    for i in tempflips:
                        stonesToFlip.append(i)
                    break
                else:
                    break
            else:
                break
            r += d[0]
            c += d[1]
    
    return stonesToFlip

@jit(nopython=True)	
def numba_possibleMoves(board, turn):
    moves = []
    for i in range(8):
        for j in range(8):
            if board[i,j] == 0:
                flips = numba_checkMove(board,i,j,turn)
                if len(flips) > 0:
                    moves.append(((i,j),flips))
    return moves

@jit(nopython=True)		
def numba_makeMove(board,move, flips, turn):
    board[move] = turn  #place stone
    for pos in flips:  #flips stones
        board[pos] *= -1
    return board
@jit(nopython=True)	
def numba_heur(board):
	Weights = [[99,-10,10,7,7,10,-10,99],
[-10,-40,-5,-3,-3,-5,-40,-10],
[10,-5,6,4,4,6,-5,10],
[7,-3,4,0,0,4,-3,7],
[7,-3,4,0,0,4,-3,7],
[10,-5,6,4,4,6,-5,10],
[-10,-40,-5,-3,-3,-5,-40,-10],
[99,-10,10,7,7,10,-10,99]]
	node = board
	heur = 0
	for i in range(len(node)):
		for j in range(len(node[i])):
			heur += Weights[i][j]* node[i,j]
	for i in [1,-1]:
		heur += i*20*len(numba_possibleMoves(board,i))
	heur_score = heur
	return heur_score
# Use when game is already over
# to check the winner
@jit(nopython=True)
def winner(board):
	sc = board.sum() #black pieces - white pieces
	if sc > 0: #MAX (black) wins
		return MAXINT
	elif sc < 0: #MIN (white) wins
		return -MAXINT
	else:
		return 0

"""
Performs minimax with alpha-beta pruning on a given Othello board
Params:
	board: An Othello Board object representing the state of the game
	player: The player who's turn it is (-1 white/MIN, 1 black/MAX)
	A: value of alpha passed from above (int)
	B: value of beta passed from above (int)
	depth: current depth (int)
	db: depth bound - maximum depth before calling heuristic (int)
Returns:
	A score if each player plays optimally according to the heuristic
"""

def miniMaxAB(board, player, A=-MAXINT, B=MAXINT, depth=0, db=DEPTHBOUND):
    # Get all of our moves
    moves = numba_possibleMoves(board, player)

    # Check if opponent has any moves (for checking if the game is over)
    opp_no_moves = numba_possibleMoves(board, -1*player)

    # game ends if neither player has a move
    if len(moves) == 0 and len(opp_no_moves) == 0:
        return winner(board)

    elif depth >= db:
        return numba_heur(board)
    
    # alpha and beta initialized to values passed in
    alpha = A
    beta = B

    # Convert typed list to regular Python list
    moves = list(moves)
    
    # if we have no moves, we must pass
    if len(moves) == 0:
        moves = [((-1, -1), [])]

    if player == 1:
        # for all moves
        for mv, flips in moves:
            # child state: copy board and make move
            newBoard = board.copy()
            # Convert flips to regular list
            flips_list = list(flips) if flips else []
            if len(flips_list) > 0:
                newBoard = numba_makeMove(newBoard, mv, flips_list, 1)
            # recursive call
            res = miniMaxAB(newBoard, -player, alpha, beta, depth + 1, db)
            
            # update alpha
            if res > alpha:
                alpha = res
            # prune
            if alpha >= beta:
                return alpha
        return alpha
    else:
        for mv, flips in moves:
            # child state: copy board and make move
            newBoard = board.copy()
            # Convert flips to regular list
            flips_list = list(flips) if flips else []
            if len(flips_list) > 0:
                newBoard = numba_makeMove(newBoard, mv, flips_list, -1)

            res = miniMaxAB(newBoard, -player, alpha, beta, depth + 1, db)
            if res < beta:
                beta = res
            if alpha >= beta:
                return beta
        return beta
#@jit(nopython=False)
def getMove(board, player, time_left):

	moves = numba_possibleMoves(board.board, player)

	opp_moves = numba_possibleMoves(board.board, -player)

	num_total_moves = len(moves) + len(opp_moves)

	alpha = -MAXINT
	beta = MAXINT

	if not moves:
		moves = [((-1,-1), [])]
	myMove = moves[0][0]
	
	if player == 1:
		for mv, flips in moves:
			newBoard = board.board.copy()
			if len(flips) > 0:
				newBoard = numba_makeMove(newBoard, mv, flips, 1)
			res = miniMaxAB(newBoard, -player, alpha, beta, 1)

			if res > alpha: 
				alpha = res
				myMove = mv

		return myMove
	else:
		for mv, flips in moves:
			newBoard = board.board.copy()
			if len(flips) > 0:
				newBoard = numba_makeMove(newBoard, mv, flips, -1)
			res = miniMaxAB(newBoard, -player, alpha, beta, 1)
			
			if res < beta:
				beta = res
				myMove = mv

		return myMove
