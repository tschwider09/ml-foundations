import numpy as np 

# MODIFY THIS TO CHANGE THE AI's DEPTH BOUND
DEPTHBOUND = 4
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
# Use when game is already over
# to check the winner
def winner(board):
	sc = board.parity() #black pieces - white pieces
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

def miniMaxAB(board, player, A=-MAXINT, B=MAXINT, depth=0, db = DEPTHBOUND):

	#get all of our moves
	moves = board.possibleMoves(player)

	#Check if opponent has any moves (for checking if the game is over)
	opp_no_moves = board.noMoves([-player])

	#game ends if neither player has a move
	if (not moves and opp_no_moves):
		return winner(board)

	elif depth >= db:
		return heuristic(board)
	
	# alpha and beta initialized to values passed in
	alpha = A
	beta = B

	#if we have no moves, we must pass
	if not moves:
		moves = [((-1,-1), [])]

	if player == 1:
		#for all moves
		for mv, flips in moves:
			#child state: copy board and make move
			newBoard = board.copy()
			newBoard.makeMove(mv, flips) #automatically switches turn
			#recursive call
			res = miniMaxAB(newBoard, -player, alpha, beta, depth + 1,)
			
			#update alpha
			if res > alpha:
				alpha = res
			#prune
			if alpha >= beta:
				return alpha
		return alpha
	else:
		for mv, flips in moves:
			#child state: copy board and make move
			newBoard = board.copy()
			newBoard.makeMove(mv, flips) #automatically switches turn
			res = miniMaxAB(newBoard, -player, alpha, beta, depth + 1)
			if res < beta:
				beta = res
			if alpha >= beta:
				return beta
		return beta


"""
params:
    board: an Othello Board object containing
        self.board: an 8x8 number array of integers (n.int8)
                    -1 is white
                    1 is black
                    0 is empty
        see board.py for other useful methods
    color: The color you are
            -1 for white
            1 for black
    time_left: the number of seconds you have left

return:
    tuple containing (0-based) coordinates (row,col) of the position you
    place a stone
    return (-1,-1) to pass if there are no legal moves for your player
"""
def getMove(board, player, time_left):

	moves = board.possibleMoves(player)

	opp_moves = board.possibleMoves(-player)

	num_total_moves = len(moves) + len(opp_moves)


	alpha = -MAXINT
	beta = MAXINT

	if not moves:
		moves = [((-1,-1), [])]
	myMove = moves[0][0]
	if player == 1:
		for mv, flips in moves:
			newBoard = board.copy()
			newBoard.makeMove(mv, flips)

			res = miniMaxAB(newBoard, -player, alpha, beta, 1)

			# We never prune at the top level
			if res > alpha: 
				alpha = res
				myMove = mv

		return myMove
	else:
		for mv, flips in moves:
			newBoard = board.copy()
			newBoard.makeMove(mv, flips)
			res = miniMaxAB(newBoard, -player, alpha, beta, 1)
			
			#update beta and best move,
			#but we never prune at the top level
			if res < beta:
				beta = res
				myMove = mv

		return myMove


