import time

import matplotlib.pyplot as plt

from .board import Board
from . import human_player
from .agents import minimax_baseline
from .agents import minimax_numba
from .agents import minimax_numba_optimized

SIZE = 8
MAX_TIME = 120

# hack - index 1 is black, index -1 is white
PLAYERS = ["", "Black", "White"]


class Othello:
    """Run one Othello game between two player modules."""

    def __init__(self, black_module, white_module, max_time=MAX_TIME):
        self.board = Board()
        self.turn = 1  # 1 = black, -1 = white
        self.black = black_module
        self.white = white_module
        self.max_time = max_time
        self.cur_times = [0.0, 0.0]  # [white, black]

    def play(self):
        plt.ion()
        self.board.showBoard()

        while not self.board.noMoves():
            ind = (self.turn + 1) // 2  # map {-1,1} -> {0,1}
            time_left = self.max_time - self.cur_times[ind]

            player_module = self.black if self.turn == 1 else self.white
            print(f"{PLAYERS[self.turn]}'s turn")

            tstart = time.time()
            move = player_module.getMove(self.board.copy(), self.turn, time_left)
            self.cur_times[ind] += time.time() - tstart

            print(
                "Time Remaining\n"
                f"Black: {self.max_time - self.cur_times[1]:.2f}\n"
                f"White: {self.max_time - self.cur_times[0]:.2f}"
            )

            if self.cur_times[ind] > self.max_time:
                print(f"{PLAYERS[self.turn]} is out of time. {PLAYERS[-self.turn]} wins!")
                return -self.turn

            if not self._is_legal_shape(move):
                print(f"Illegal move ({move}) by {PLAYERS[self.turn]}. {PLAYERS[-self.turn]} wins!")
                return -self.turn

            if move == (-1, -1):
                if self.board.noMoves([self.turn]):
                    self.turn *= -1
                    continue
                print(f"Illegal pass by {PLAYERS[self.turn]}. {PLAYERS[-self.turn]} wins!")
                return -self.turn

            flips = self.board.checkMove(move[0], move[1], self.turn)
            if not flips:
                print(f"Illegal move ({move}) by {PLAYERS[self.turn]}. {PLAYERS[-self.turn]} wins!")
                return -self.turn

            self.board.makeMove(move, flips)
            self.board.showBoard()
            self.turn *= -1

        sb, sw = self.board.score()
        print(f"Final score: Black {sb} ({self.black.__name__}), White {sw} ({self.white.__name__})")
        if sb > sw:
            return 1
        if sw > sb:
            return -1
        return 0

    @staticmethod
    def _is_legal_shape(move):
        try:
            row, col = move
            return -1 <= row < SIZE and -1 <= col < SIZE
        except Exception:
            return False


def run_default_match():
    """Default benchmark match used during class experimentation."""
    game = Othello(minimax_numba, minimax_numba_optimized)
    winner = game.play()
    if winner == 1:
        winner_name = game.black.__name__
    elif winner == -1:
        winner_name = game.white.__name__
    else:
        winner_name = "Draw"
    print(f"Game over. Winner: {winner_name}")
    plt.ioff()
    plt.close()


def run_human_vs_baseline():
    game = Othello(human_player, minimax_baseline)
    winner = game.play()
    print(f"Game over. Winner code: {winner}")
    plt.ioff()
    plt.close()


def main():
    run_default_match()


if __name__ == "__main__":
    main()
