"""
This module contains agents that play reversi.

Version 3.1
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
            time.sleep(3)
            randidx = random.randint(0, len(valid_actions) - 1)
            random_action = valid_actions[randidx]
            output_move_row.value = random_action[0]
            output_move_column.value = random_action[1]
        except Exception as e:
            print(type(e).__name__, ':', e)
            print('search() Traceback (most recent call last): ')
            traceback.print_tb(e.__traceback__)


class AgentMongChaChaVI(ReversiAgent):
    """An agent that evaluates each action using trained weights"""

    def __init__(self, color: int):
        """
        Initialize the weight matrix
        :param color: color of the current agent. It can be only either BLACK or WHITE
        """
        super().__init__(color)
        weights = (
            80, -26, 24, -1, -5, 28, -18, 76, -23, -39, -18, -9, -6, -8, -39, -1, 46, -16, 4, 1, -3, 6, -20, 52, -13,
            -5, 2, -1, 4, 3, -12, -2, -5, -6, 1, -2, -3, 0, -9, -5, 48, -13, 12, 5, 0, 5, -24, 41, -27, -53, -11, -1,
            -11, -16, -58, -15, 87, -25, 27, -1, 5, 36, -3, 100)
        self.trainedWeights = np.array(weights).reshape(8, 8)  # print(self.trainedWeights)

    def search(self, color, board, valid_actions, output_move_row, output_move_column):
        """
        Begins searching for a round of the game
        :param color:               the color of the current player.
        :param board:               the state of the board
        :param valid_actions:       the executable actions for the current player
        :param output_move_row:     the variable to store the outcome of searching in ROW manner
        :param output_move_column:  the variable to store the outcome of searching in ROW manner
        :return:                    nothing
        """

        # Hard-coded statements for testing against the same type of agent
        if self._color == 1:
            evaluation, bestAction = self.minimax(board, valid_actions, 3, 0, -99999, 99999, True)
        else:
            evaluation, bestAction = self.minimax(board, valid_actions, 3, 0, -99999, 99999, True)

        # print("Me Selected: " + str(bestAction))

        # Stupid error messages avoidance
        if bestAction is not None:
            output_move_row.value = bestAction[0]
            output_move_column.value = bestAction[1]

    def minimax(self, board: np.array, validActions: np.array, depth: int, levelCount: int, alpha: int, beta: int,
                maximizingPlayer: bool):
        """
        Recursively find the optimal action based on the current observation.
        The algorithm is Minimax with Alpha-Beta pruning
        :param board:               the state of the board
        :param validActions:        executable actions for the current player
        :param depth:               depth limit for recursive Minimax
        :param levelCount:          depth counter
        :param alpha:               alpha value for pruning the Search tree
        :param beta:                beta value for pruning the Search tree
        :param maximizingPlayer:    determine whether the current Minimax node is a Maximizing node or not
        :return:                    At levelCount == 0: returns eval and bestAction; otherwise, returns only eval.
        """
        if depth == 0:
            return self.evaluateStatistically(board)

        bestAction: np.array = None
        if maximizingPlayer:
            mAlpha: int = alpha
            maxEval: int = -99999
            player: int = self._color

            for action in validActions:
                newState, newValidActions = self.createState(board, action, player)
                evaluation = self.minimax(newState, newValidActions, depth - 1, levelCount + 1, mAlpha, beta,
                                          not maximizingPlayer)

                if maxEval < evaluation:
                    maxEval = evaluation

                    if levelCount == 0:
                        bestAction = action

                mAlpha = max(mAlpha, evaluation)
                if beta <= mAlpha:
                    break
            if levelCount != 0:
                return maxEval
            else:
                return maxEval, bestAction
        else:
            mBeta: int = beta
            minEval: int = 99999
            player: int = self.getOpponent(self._color)

            for action in validActions:
                newState, newValidActions = self.createState(board, action, player)
                evaluation = self.minimax(newState, newValidActions, depth - 1, levelCount + 1, alpha, mBeta,
                                          not maximizingPlayer)

                if minEval > evaluation:
                    minEval = evaluation

                    if levelCount == 0:
                        bestAction = action

                mBeta = min(mBeta, evaluation)
                if mBeta <= alpha:
                    break
            if levelCount != 0:
                return minEval
            else:
                return minEval, bestAction

    def evaluateStatistically(self, board: np.array) -> int:
        """
        Calculates the Evaluation at the depth limit
        :param board:   current state of the board
        :return:        evaluation value
        """

        nonZeroPositions = np.array(list(zip(*board.nonzero())))

        myEvaluationScore = 0
        opponentEvaluationScore = 0

        for position in nonZeroPositions:
            positionY, positionX = position[0], position[1]

            # If the current position has a piece of this Agent
            if board[positionY][positionX] == self._color:
                # Get the weight from hard-coded matrix
                myEvaluationScore += self.trainedWeights[positionY][positionX]
            else:
                opponentEvaluationScore += self.trainedWeights[positionY][positionX]

        # The differences between this Agent and its opponent.
        return myEvaluationScore - opponentEvaluationScore

    @staticmethod
    def getOpponent(player: int):
        """
        Returns the opponent player identifier
        :param player:      a player
        :return:            the opponent of that player
        """
        if player == 1:
            return -1
        else:
            return 1

    def createState(self, board: np.array, action: np.array, player: int) -> (np.array, np.array):
        """
        Creates a new state and new actions based on given a state and an action.
        :param board:       a state of the board
        :param action:      an action
        :param player:      a player that performs the action
        :return:            a new state and a set of new possible actions
        """
        newState: np.array = transition(board, player, action)

        validMoves: np.array = _ENV.get_valid((newState, self.getOpponent(player)))
        validMoves: np.array = np.array(list(zip(*validMoves.nonzero())))

        return newState, validMoves

