import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# hack - index 1 is black, index -1 is white
players = ["", "Black", "White"]
SIZE = 8


"""
Represents an 8x8 Othello board
board state is an 8x8 numpy array
Useful methods provided for checking various
aspects of the game state
"""
class Board:
    def __init__(self, board=None, turn=1):
        if board is None:
            board = np.zeros((SIZE,SIZE), dtype=np.int32)
            # starting board state
            board[3,3] = -1 
            board[4,4] = -1 
            board[3,4] = 1 
            board[4,3] = 1 
        self.turn = turn # default to 1 (aka. Black)
        self.board = board
    
    # call to access the actual numpy array
    def getBoard(self):
        return self.board

    # number of black - number of white stones
    def parity(self):
        return self.board.sum()

    """
    Check the outcome of placing a stone at a given position
    Parameters:
        self: an Othello Board object
        row, col: coordinates of the move to test
        turn: color of the stone to place -1(white) or 1(black)
              if no turn is provided, 
              defaults to the current turn
              stored in the Board object
    Returns:
        A list of (row,col) tuples of stones
            of the other color that flip as a result of
            placing this stone
        RETURNS AN EMPTY LIST IF THE MOVE IS ILLEGAL
        A move is legal ONLY if it causes 
        at least one stone of the other color to flip
    """
    def checkMove(self, row, col, turn=None):
        if not turn:
            turn = self.turn

        # position is not empty, move illegal
        if self.board[row, col] != 0:
            return []

        stonesToFlip = []

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
            while r >= 0 and r < SIZE and c >= 0 and c < SIZE:
                # first we should find stones of opponents color
                if self.board[r, c] == -turn:
                    flank = True # found at least one stone we think is flanked
                    tempflips.append((r, c))  # opponent stones we might flip

                # then a stone of our color, so the line is surrounded
                elif self.board[r, c] == turn:
                    if flank:  # had found at least one stone of opponent color
                                # now confirmed surrounded by our color
                        stonesToFlip += tempflips
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
   
    """
    Returns False if there are any legal moves for either player
    Optional parameter allows you to check if 
        neither player has a move, in which case
        the game is over (default)
        OR
        Check if a specific player has no moves,
        in which case they must pass
    """
    def noMoves(self, check=[-1, 1]):
        for t in check:  # check both players if necessary
            for i in range(SIZE):
                for j in range(SIZE):
                    # only run checkMove if spot is empty
                    # since checkMove is expensive
                    if self.board[i, j] == 0:
                        if self.checkMove(i, j, t):
                            return False
        return True

    """
    Make the move and modify the actual board
    Params:
        move: (row, col) of stone to place
        flips: list of stones of opposite color to flip
                Since checkMove will have already been called
                we don't want to compute this again
    """
    def makeMove(self, move, flips):
        if flips:
            self.board[move] = self.turn  #place stone
        for pos in flips:  #flips stones
            self.board[pos] *= -1
        self.turn = -self.turn # change the turn


    """
    Params:
        player: -1 (white) or 1(black) 
    Return:
        a list of legal moves for player
    Use this method only if you need 
    the actual list of moves.
    If you only need to know if any moves exist,
    Board.noMoves([player]) is faster.
    """
    def possibleMoves(self, player):
        moves = []
        for i in range(SIZE): #check all positions
            for j in range(SIZE):

                # only run checkMove if spot is empty
                # since checkMove is expensive
                if self.board[i,j] == 0:
                    flips = self.checkMove(i,j,player)
                    if flips:
                        moves.append(((i,j),flips))
        return moves



    # calculate black and white scores
    # return (black_score, white_score)
    def score(self):
        score_black = 0
        score_white = 0

        #TO DO: make more efficient with numpy

        # loop through board, count stones
        for r in range(SIZE):
            for c in range(SIZE):
                if self.board[r, c] == 1:
                    score_black += 1
                elif self.board[r, c] == -1:
                    score_white += 1
        return (score_black, score_white)

    # Important since board needs to be copied before 
    # making a move in minimax
    def copy(self):
        return Board(self.board.copy(), self.turn)

    # display board using pyplot and pillow
    def showBoard(self):
        plt.clf() # clear previous board

        # green background
        green = np.array([0,100,0,255], dtype=np.uint8)
        imArr = np.full((800,800,4), green)

        # convert to PIL image to use ImageDraw module
        im = Image.fromarray(imArr)
        draw = ImageDraw.Draw(im)

        # draw gridlines
        for i in range(100,800,100):
            draw.line((i,0,i,im.size[1]), fill = "white", width = 3)
        for j in range(100,800,100):
            draw.line((0,j,im.size[0],j), fill = "white", width = 3)

        # draw stones
        for r in range(SIZE):
            for c in range(SIZE):
                if self.board[r,c] == 1:
                    draw.ellipse([100*c + 5, 100*r+5,100*c + 95, 100*r+95], fill="black")
                elif self.board[r,c] == -1:
                    draw.ellipse([100*c + 5, 100*r+5,100*c + 95, 100*r+95], fill="white")
        
        # convert back to array so plt can plot it.
        arr = np.array(im)
        fig = plt.imshow(arr)


        # get rid of axis ticks and numbers
        # plt.axis('off')
        # fig.axes.get_xaxis().set_visible(False)
        # fig.axes.get_yaxis().set_visible(False)
        plt.xticks(range(50,800,100),range(8))
        plt.yticks(range(50,800,100),range(8))

        # show updated image
        plt.draw()
        plt.pause(.1)


    
