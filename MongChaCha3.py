"""
This module contains agents that play reversi.
Version 3.0
"""

import abc
import random
import asyncio
import traceback
import time
import math
from multiprocessing import Process, Value

import numpy as np
import gym
import boardgame2 as bg2

_ENV = gym.make('Reversi-v0')
_ENV.reset()


def transition(board, player, action):
    """Return a new board if the action is valid, otherwise None."""
    if _ENV.is_valid((board, player), action):
        new_board, __ = _ENV.get_next_state((board, player), action)
        return new_board
    return None


class ReversiAgent(abc.ABC):
    """Reversi Agent."""

    def __init__(self, color):
        """
        Create an agent.

        Parameters
        -------------
        color : int
            BLACK is 1 and WHITE is -1. We can get these constants
            from bg2.BLACK and bg2.WHITE.
        """
        super().__init__()
        self._move = None
        self._color = color

    @property
    def player(self):
        """Return the color of this agent."""
        return self._color

    @property
    def pass_move(self):
        """Return move that skips the turn."""
        return np.array([-1, 0])

    @property
    def best_move(self):
        """Return move after the thinking.

        Returns
        ------------
        move : np.array
            The array contains an index x, y.
        """
        if self._move is not None:
            return self._move
        else:
            return self.pass_move

    async def move(self, board, valid_actions):
        """Return a move. The returned is also availabel at self._move."""
        self._move = None
        output_move_row = Value('d', -1)
        output_move_column = Value('d', 0)
        try:
            # await self.search(board, valid_actions)    
            p = Process(target=self.search,
                args=(self._color, board, valid_actions, output_move_row, output_move_column))
            p.start()
            while p.is_alive():
                await asyncio.sleep(0.1)
        except asyncio.CancelledError as e:
            print('The previous player is interrupted by a user or a timer.')
        except Exception as e:
            print(type(e).__name__)
            print('move() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)
        finally:
            p.kill()
            self._move = np.array([output_move_row.value, output_move_column.value], dtype=np.int32)
        return self.best_move

    @abc.abstractmethod
    def search(self, color, board, valid_actions, output_move_row, output_move_column):
        """
        Set the intended move to self._move.

        The intended move is a np.array([r, c]) where r is the row index
        and c is the column index on the board. [r, c] must be one of the
        valid_actions, otherwise the game will skip your turn.
        Parameters
        -------------------
        board : np.array
            An 8x8 array that contains 
        valid_actions : np.array
            An array of shape (n, 2) where n is the number of valid move.
        Returns
        -------------------
        None
            This method should set value for 
            `output_move_row.value` and `output_move_column.value` 
            as a way to return.
        """
        raise NotImplementedError('You will have to implement this.')


class RandomAgent(ReversiAgent):
    """An agent that move randomly."""

    def search(self, color, board, valid_actions, output_move_row, output_move_column):
        """Set the intended move to the value of output_moves."""
        # If you want to "simulate a move", you can call the following function:
        # transition(board, self.player, valid_actions[0])

        # To prevent your agent to fail silently we should an
        # explicit trackback printout.
        try:
            # while True:
            #     pass
            time.sleep(0.25)
            randidx = random.randint(0, len(valid_actions) - 1)
            random_action = valid_actions[randidx]
            output_move_row.value = random_action[0]
            output_move_column.value = random_action[1]
        except Exception as e:
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)


class WarotAgent(ReversiAgent):
    """My agent."""
    """
    Weight of each position on the board [ref. https://github.com/hylbyj/Alpha-Beta-Pruning-for-Othello-Game/blob/master/readme_alpha_beta.txt]
    """
    WEIGHTS = [[4, -3, 2, 2, 2, 2, -3, 4], [-3, -4, -1, -1, -1, -1, -4, -3], [2, -1, 1, 0, 0, 1, -1, 2],
               [2, -1, 0, 1, 1, 0, -1, 2], [2, -1, 0, 1, 1, 0, -1, 2], [2, -1, 1, 0, 0, 1, -1, 2],
               [-3, -4, -1, -1, -1, -1, -4, -3], [4, -3, 2, 2, 2, 2, -3, 4]]

    def alphabetasearch(self, board, valid_actions, depth_limit=10):
        """Alpha Beta Search Method, return the action"""
        v, action = self.maxvalue(board, -math.inf, math.inf, valid_actions, 1, depth_limit)
        return action

    def maxvalue(self, board, alpha, beta, valid_actions, limit, depht_limit):
        winner = _ENV.get_winner((board, self.player))
        if (winner is not None) or limit >= depht_limit:
            return self.score(board)
        v = -math.inf
        new_alpha = alpha
        best_action = None
        for action in valid_actions:
            next_state = transition(board, self.player, action)
            newAction = _ENV.get_valid((next_state, self.player))
            newAction = np.array(list(zip(*newAction.nonzero())))
            MinNode = self.minvalue(next_state, new_alpha, beta, newAction, limit + 1, depht_limit)
            if MinNode > v:
                v = MinNode
                best_action = action
            if v >= beta:
                break
            new_alpha = max(alpha, v)
        if limit == 1:
            return v, best_action
        else:
            return v

    def minvalue(self, board, alpha, beta, valid_actions, limit, depht_limit):
        winner = _ENV.get_winner((board, self.player))
        if (winner is not None) or limit >= depht_limit:
            return self.score(board)
        new_beta = beta
        v = math.inf
        best_action = None
        for action in valid_actions:
            next_state = transition(board, self.player, action)
            newAction = _ENV.get_valid((next_state, self.player))
            newAction = np.array(list(zip(*newAction.nonzero())))
            MaxNode = self.maxvalue(next_state, alpha, new_beta, newAction, limit + 1, depht_limit)
            if MaxNode < v:
                v = MaxNode
                best_action = action
            if v <= alpha:
                break
            new_beta = min(beta, v)
        if limit == 1:
            return v, best_action
        else:
            return v

    def score(self, board):

        def f1(self):
            """
            Mobility
            ----------------
            A number of choice that player can make
            """
            return len(self.valid_actions)

        def f2(self, board):
            """
            Pieces Score & Corner Matrix
            ----------------
            Pieces Score
                return pieces
            A score based on [Player's pieces - Opponent's pieces] on the board.
            Corner Matrix
                return total
            A score based on where the pieces were places according to [WEIGHT] matrix declared above. 
            [ref. https://github.com/hylbyj/Alpha-Beta-Pruning-for-Othello-Game/blob/master/readme_alpha_beta.txt]
            """
            total = 0
            i = 0
            countB = 0
            countW = 0
            for i in range(0, board.shape[0]):
                for j in range(0, board.shape[1]):
                    if board[i, j] == self._color:
                        total += self.WEIGHTS[i][j]
                    if board[i, j] == -self._color:
                        total -= self.WEIGHTS[i][j]
                    countB += (board[i, j] == self._color)
                    countW += (board[i, j] != self._color)
            pieces = countB - countW
            return pieces, total

        pieces, corner = f2(self, board)
        return f1(self) + 3 * pieces + 4 * corner

    def search(self, color, board, valid_actions, output_move_row, output_move_column):
        """Set the intended move to self._move."""
        try:
            self.valid_actions = valid_actions
            action = self.alphabetasearch(board, valid_actions, 4)
            if action is not None:
                output_move_row.value = action[0]
                output_move_column.value = action[1]
        except Exception as e:
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)