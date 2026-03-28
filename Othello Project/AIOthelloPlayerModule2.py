import numpy as np 
import random
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
# main function for the Monte Carlo Tree Search
class Node:
	def __init__(self, board, visits=0, total=0, root=False, children=[], turn=1, move=(0,0)):
		self.board = board.copy()
		self.visits = visits
		self.total = total
		self.root = root
		self.children = children
		self.turn = turn
		self.move = move
		self.possible_children = []

		moves = board.possibleMoves(turn)
		for mv, flips in moves:
			#child state: copy board and make move
			newBoard = board.copy()
			newBoard.makeMove(mv, flips)
			self.possible_children.append((newBoard,(mv,flips)))
		


	def root_check(self):
		if self.root ==True:
			return self.root
	def best_uct(self):
		score = -MAXINT
		winner = None
		for node in self.children:
			current = node.total/node.visits + .8*np.sqrt(np.log(node.root.visits)/node.visits)
			if current > score:
				score = current
				winner = node
		return winner
	def pick_unvisited(self):
		new = random.choice(self.possible_children)
		self.possible_children.remove(new)
		board = new[0]
		move =new[1]
		self.children.append(Node(board, 0, 0, self, [], self.turn *-1,move))
		return self.children[-1]
	
	def fully_expanded(self):
		if len(self.possible_children)==0 and len(self.children) != 0:
			return True
		else:
			return False
	def best_child(self):
		score = -MAXINT
		winner = None
		
		for node in self.children:
			current = node.visits
			if current > score:
				
				score = current
				winner = node
		return winner
 
	
    
def monte_carlo_tree_search(root, turn):
    i = 0
    while i <20000:
        leaf = traverse(root) 
        simulation_result = rollout_corner(leaf)
        backpropagate(leaf, simulation_result, turn)
        i+=1    
    best = root.best_child()
    if best ==None:
        return None

    return best,best.move[0]


# function for node traversal
def traverse(node):
    while node.fully_expanded():
        node = node.best_uct()
         
    # in case no children are present / node is terminal
    if len(node.possible_children) ==0:
        return node
    return node.pick_unvisited() 
 
# function for the result of the simulation
def rollout_corner(node):
	board = node.board.copy()
	turn = node.turn
	while True:
		moves = board.possibleMoves(turn)
		if len(moves) == 0:
			return 0
		i = random.randint(0, len(moves)-1)

		board.makeMove(moves[i][0],moves[i][1])	
		for i in ([(0,0),(0,7),(7,0),(7,7)]):
			if board.board[i[0],i[1]] != 0:
				return board.board[i[0],i[1]]
		turn = -turn
def rollout(node):
	board = node.board.copy()
	turn = node.turn
	while True:
		moves = board.possibleMoves(turn)
		if len(moves) == 0:
			break
		i = random.randint(0, len(moves)-1)
		board.makeMove(moves[i][0],moves[i][1])
		turn = -turn
	if len(board.possibleMoves(-turn)) != 0:
		return -turn
	sumA = board.board.sum()

	if sumA > 0:
		return 1
	elif sumA < 0:
		return -1
	else:
		return 0

		
 
# function for randomly selecting a child node

# function for backpropagation
def backpropagate(node, result, turn):
	node.visits += 1
	if node.root_check():
		return None
	node.total += result * turn
	backpropagate(node.root, result, turn)
 
# function for selecting the best child
# node with highest number of visits


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
	zeros = 0
	for i in range(len(board.board)):
		for j in range(len(board.board[i])):
			if board.board[i,j] == 0:
				zeros += 1
	
	
	if zeros >= 59:
		global rootm 
		rootm = Node(board,0,0, True, [], player, (0,0) )
	else:
		for i in range(len(board.board)):
			for j in range(len(board.board[i])):
				if board.board[i,j] != rootm.board.board[i,j]:
					for c in rootm.children:
						if c.move[0] ==  (i,j):
							rootm = c

	if len(board.possibleMoves(player)) == 0:
		return (-1,-1)
	

	new_root, move = monte_carlo_tree_search(rootm, player)
	rootm = new_root

	return move

