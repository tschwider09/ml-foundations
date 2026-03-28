import numpy as np 
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
def getMove(board, color, time_left):
    #print(board)
    if color == 1:
        print("Black to Move")
    else:
        print("White to Move")
    mv = input("enter coordinates of your move, separated by spaces: ")
    mv = mv.split()
    return (int(mv[0]),int(mv[1]))

