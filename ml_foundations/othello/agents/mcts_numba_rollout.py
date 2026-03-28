import numpy as np 
import random
from numba import jit
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
	def __init__(self, board, visits=0, total=0, root=False, children=None, turn=1,true_turn=1, move=(0,0)):
		if children is None:
			children = []
		self.board = board.copy()
		self.visits = visits
		self.total = total
		self.root = root
		self.children = children
		self.turn = turn
		self.move = move
		self.possible_children = []
		self.true_turn = true_turn
		
		moves = board.possibleMoves(turn)
		for mv, flips in moves:
			#child state: copy board and make move
			newBoard = board.copy()
			newBoard.makeMove(mv, flips)
			self.possible_children.append((newBoard,(mv,flips)))

		"""
		if self.true_turn != self.turn:
			cull = {}
			for p in self.possible_children:
				board = self.board.copy()
				board.makeMove(p[1][0],p[1][1])
				cull[self.turn*numba_heur(board.board)] = p
			options = []
			for i in range(len(cull.keys())//2):
				options.append(max(cull.keys()))
			self.possible_children = []
			for i in options:
				self.possible_children.append(cull[i])
		"""

			

		
		


	def root_check(self):
		if self.root ==True:
			return self.root
	def best_uct(self):
		score = -MAXINT
		winner = None
		for node in self.children:
			current = node.total/node.visits + 5*np.sqrt(np.log(node.root.visits)/node.visits)
			if current > score:
				score = current
				winner = node
		return winner
	def pick_unvisited(self):
		new = random.choice(self.possible_children)
		self.possible_children.remove(new)
		board = new[0]
		move =new[1]
		self.children.append(Node(board, 1, 0, self, [], self.turn *-1,self.true_turn,move))
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
    while i <10000:
        leaf = traverse(root)
        board = leaf.board.board.copy()
        player = leaf.turn 
		
        simulation_result = numba_rollout(board, player)
        backpropagate(leaf, simulation_result, turn)
        i+=1    
    best = root.best_child()
    if best == None:
        return ((-1,-1), None)
    #if best.move[0] in [(1,1),(6,1),(1,6),(6,6)]:
    print(root.board.board)
    for i in root.children:
        print(i.move[0])
        print(i.total)
        print(i.visits)
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
@jit(nopython=True)
def numba_checkMove(board, row, col, turn):

    stonesToFlip = []
    if board[row, col] != 0:
        return None
    # 8 directions incl diagonals
    #down up right left dr dl ur ul
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    for d in directions:
        flank = False # enemy stones must be flanked by stones of your color to be flipped
        r = row + d[0]  # first position we're checking
        c = col + d[1]  # in the current direction

            # temporary list of positions to flip
            # don't know if accurate until we find that the stones are flanked
        tempflips = []


            # while we're still on the board
            # we move in the current direction and check squares
        while r >= 0 and r < 8 and c >= 0 and c < 8:
            # first we should find stones of 8 color
            if board[r, c] == -turn:
                flank = True # found at least one stone we think is flanked
                tempflips.append((r, c))  # opponent stones we might flip

                # then a stone of our color, so the line is surrounded
            elif board[r, c] == turn:
                if flank:  # had found at least one stone of opponent color
                                # now confirmed surrounded by our color
                    for i in tempflips:
                        stonesToFlip.append(i)
                    break

                else:  # if no stones of opponents color
                    break  # break without adding any stones
            else:  # found a blank spot when expecting something else
                break  # break without adding any stones

            r += d[0]  # next
            c += d[1]  # position

            # if we reached the end of the loop without finding flanked stones, no stones are added
        # We've searched all directions
    return stonesToFlip

@jit(nopython=True)	
def numba_possibleMoves(board, turn):
    moves = []
    for i in range(8): #check all positions
        for j in range(8):

            # only run checkMove if spot is empty
            # since checkMove is expensive
            if board[i,j] == 0:
                flips = numba_checkMove(board,i,j,turn)
                if flips:
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
	heur_score = heur / 650

	return heur_score

@jit(nopython=True)
def count_zeros(board):
	zeros = 0
	for i in range(len(board)):
		for j in range(len(board[i])):
			if board[i,j]==0:
				zeros+=1
	return zeros
@jit(nopython=True)
def numba_rollout(board, turn):
	
	for i in range(20):
	
		moves = numba_possibleMoves(board, turn)
		if len(moves) != 0:
			i = random.randint(0, len(moves)-1)
			board = numba_makeMove(board, moves[i][0],moves[i][1], turn)
		elif count_zeros(board)==0:
			sumA = board.sum()
			if sumA > 0:
				return 1
			elif sumA < 0:
				return -1
			else:
				return 0
		turn = -turn

	return numba_heur(board)
	


		
 
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
	rootm = Node(board,1,0, True, [], player,player, (0,0) )
	"""
	if zeros >= 59:
		global rootm 
		rootm = Node(board,1,0, True, [], player,player, (0,0) )
	else:
		for i in range(len(board.board)):
			for j in range(len(board.board[i])):
				if board.board[i,j] != rootm.board.board[i,j]:
					f = False
					for c in rootm.children:
						if c.move[0] ==  (i,j):
							f = True
							rootm = c
							rootm.root = True
					if not f:
						rootm = Node(board,1,0, True, [], player,player, (0,0) )
					
	if len(board.possibleMoves(player)) == 0:
		return (-1,-1)
	"""
	
	new_root, move = monte_carlo_tree_search(rootm, player)
	rootm = new_root
	
	return move




"""
black:
[[ 0  0 -1 -1 -1 -1 -1  0]
 [ 0  0  0  0  1 -1  0  0]
 [ 1  1  1  1 -1  1 -1  1]
 [ 0  1  1 -1  1 -1  1  1]
 [-1 -1 -1 -1 -1  1  0  1]
 [ 0  0  1 -1 -1  1  1  1]
 [ 0  0 -1  1  1  1  0  1]
 [ 0 -1 -1 -1 -1 -1 -1  0]]
 

 [[ 0  0 -1  0  0  0  0  0]
 [ 0  0 -1 -1  0  0  0  0]
 [ 0  0 -1  1 -1  1  0  0]
 [ 0  0 -1 -1  1  1  0  0]
 [ 0  0 -1 -1  1  1  0  0]
 [ 0  0  0  1  1  1  0  0]
 [ 0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0]]

white:
 [[ 0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0]
 [ 1  1  1  1  0 -1  0  0]
 [ 0  1  1  1 -1  0  0  0]
 [ 0  1 -1 -1 -1  0  0  0]
 [ 0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0]]
 """
