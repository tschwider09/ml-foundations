import numpy as np 
import numba as nb
import matplotlib.pyplot as plt 
import time
import random
from board import Board

# IMPORT AI MODULES - change this to add more strategies
import humanOthello2024
import AIOthelloPlayerModule
import AIOthelloPlayerModule2
import AIOthelloPlayerModule3
import AIOthelloPlayerModule4
import AIOthelloPlayerModule5
SIZE = 8

# Maximimum time per side. 
# Will be 2 minutes (120 sec)
# for tournament, but currently 10 minutes for 
# human testing purposes
MAX_TIME = 120

# number of black - number of white stones
def parity(board):
    return board.sum()

# hack - index 1 is black, index -1 is white
players = ["", "Black", "White"]

# represents a game of Othello with 2 players
# useful methods are in Board class
class Othello:

    # Pass in module names to control for the two players
    # PASS IN DIFFERNT MODULE NAMES TO TEST AIs
    def __init__(self, blackModule, whiteModule):
        #create a board object
        self.board = Board()
        
        self.turn = 1 # will be -1 (white) or 1 (black)


        # Grab move functions from modules to play each side
        self.black = blackModule
        self.white = whiteModule

        self.max_time = MAX_TIME # maximum time (seconds) allocated 
                            # to each AI for the whole game
        self.cur_times = [0,0] # elapsed time for white, black

        # turn on interactive mode for visual
        plt.ion() 


        winner = self.play() # game loop
        plt.ioff() #

        if winner == 1:
            winner = blackModule.__name__
        elif winner == -1:
            winner = whiteModule.__name__
        else:
            winner = "Draw"
        input(f"game over. Winner: {winner}")
        plt.close() # close image of board
        



    # Main Othello Game loop
    def play(self):
        self.board.showBoard()

        # Until game is over
        while not self.board.noMoves():
        
            # convert [-1,1] to [0, 1]
            ind = (self.turn + 1) // 2
            time_left = self.max_time - self.cur_times[ind]

            # Get move from AI
            if self.turn == 1:
                print("Black's turn")
                # measure time expired within function call
                tstart = time.time()
                # passing in a copy of the board to avoid AIs messing with it
                move = self.black.getMove(self.board.copy(), 1, time_left)
                tend = time.time()
            else:
                print("White's turn")
                tstart = time.time()
                move = self.white.getMove(self.board.copy(), -1, time_left)
                tend = time.time()
            

            # add time elapsed this move
            self.cur_times[ind] += tend - tstart

            print(f"Time Remaining\nBlack: {self.max_time - self.cur_times[1]}\nWhite: {self.max_time - self.cur_times[0]}")

            # Timeout game over condition
            if self.cur_times[ind] > self.max_time:
                print(f"{players[self.turn]} is out of time. {players[-self.turn]} wins!")

                return -self.turn       
            try: 
                # Check if the move is in bounds
                if move[0] >= -1 and move[0] < SIZE and move[1] >= -1 and move[1] < SIZE:
                    pass
                else:
                    print(f"Illegal Move ({move}) by {players[self.turn]}. {players[-self.turn]} wins!")
                    return -self.turn
            # Catch exception if move is illegally formated (not a tuple of 2 ints)
            except:
                print(f"Illegal Move ({move}) by {players[self.turn]}. {players[-self.turn]} wins!")
                return -self.turn

            # if player passed
            if (move[0], move[1]) == (-1,-1): 
                #Check that the pass is legal
                if self.board.noMoves([self.turn]):
                    self.turn *= -1
                    continue
                else:
                    print(f"Illegal pass by {players[self.turn]}. {players[-self.turn]} wins!") 
                    return -self.turn

            # list of stones to flip, empty if illegal
            flips = self.board.checkMove(move[0], move[1], self.turn)
            # if move is legal, perform it
            if flips:
                self.board.makeMove(move, flips)
                # self.board.board[move] = self.turn#place stone
                # for pos in flips: #flips stones
                #     self.board.board[pos] *= -1
            # move was invalid. Player auto-loss
            else: 
                print(f"Illegal Move ({move}) by {players[self.turn]}. {players[-self.turn]} wins!")
                return -self.turn

            self.board.showBoard()
            self.turn *= -1 #toggle turn

        # calculate Final scores and declare winner.
        sb, sw = self.board.score()
        print(f"Final Score: Black {sb}({self.black.__name__}), White {sw}({self.white.__name__})")
        if sb > sw:
            print ("Black wins")
            return 1
        elif sb < sw:
            print ("White wins")
            return -1
        else:
            print("It's a draw!")


def main():
    #MODIFY THIS SECTION TO ADD/CHANGE AI PLAYERS
    #####
    #This list needs name string for each module 
    player_names = ['Fisher','Justin and Matthew','Damoni and Paul','Miles and Roan','Philip and Leo', "Ari"] 
    #Names of the modules that will take part in the tournament
    #Module name is the file name without the .py
    player_modules = [FisherSecretAI, JustinMatthewOthello, DamoniPaulOthello, MilesRoanOthello, PhilipLeoOthellocode, AriOthello]
    #####

    num_players = len(player_names)

    # create all unique matchups (order matters)
    matchups = [(i,j) for i in range(num_players) for j in range(num_players) if i!=j]

    # Shuffle so the order of matches is 
    # distributed amongst players
    # but seed the rng so the order is the same
    # in case we have to restart
    random.seed(4)
    random.shuffle(matchups)
    print(matchups)

    gameNum = 1
    # Play all matchups
    for p1, p2 in matchups:
        print(f"Game {gameNum}: {player_names[p1]}(black) vs. {player_names[p2]}(white)")
        # Allow me to enter 's' to skip a matchup
        skip =  input("Press Enter to begin")
        if skip!='s':
            o = Othello(player_modules[p1], player_modules[p2])
        print("--------------")
        gameNum += 1

#if __name__ == '__main__':
    #main()
#Othello(humanOthello2024, AIOthelloPlayerModule)

Othello(AIOthelloPlayerModule4,AIOthelloPlayerModule5)
"""
test =  [[ 0,  0 ,-1 , 0  ,0 , 0 , 0 , 0],
        [ 0  ,0 ,-1,-1 , 0,  0 , 0  ,0],
        [ 0 , 0 ,-1 , 1 ,-1 , 1 , 0 , 0],
        [ 0 , 0, -1, -1,  1,  1,  0 , 0],
        [ 0 , 0, -1 ,-1,  1,  1,  0 , 0],
        [ 0 , 0,  0 , 1 , 1 , 1  ,0 , 0],
        [ 0  ,0,  0 , 0,  0 , 0 , 0  ,0],
        [ 0 , 0 , 0  ,0 , 0 , 0,  0 , 0]]

test = np.array(test)
board = Board(test, 1)
print(AIOthelloPlayerModule3.getMove(board,1, 300))
"""

