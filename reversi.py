"""
This module is a reversi engine for the AI.

Version 3.1
"""

import os
from tqdm import tqdm
import time
import asyncio
import subprocess

import gym
import numpy as np
import boardgame2 as bg2

import Project.reversi_agent as agents
import Project.MongChaCha3 as agent3
import Project.MongChaCha4 as agent4
import Project.MongChaCha5 as agent5

_name = {bg2.BLACK: 'BLACK', bg2.WHITE: 'WHITE'}

def clear_screen():
    """Clear the shell."""
    if os.name == 'nt':
        subprocess.call('cls', shell=True)
    else:
        subprocess.call('clear', shell=True)
    # pass

def render(board, turn, prev_move=None, prev_turn=None):
    """Render on the screen."""
    clear_screen()
    if prev_move is not None:
        print(f'Previous Move: {prev_move} by {_name[prev_turn]}')
    print(board)
    black_score = np.sum(board == bg2.BLACK)
    white_score = np.sum(board == bg2.WHITE)
    print(f'  BLACK : {black_score}  -  {white_score} : WHITE')

async def timer(limit):
    """Create a progress bar for timer."""
    for i in tqdm(range(limit*10), desc="Time Limit: "):
        await asyncio.sleep(1/10)

async def main(black, white, timelimit=2):
    """Run the game."""
    env = gym.make('Reversi-v0')
    board, turn = env.reset()
    render(board, turn)

    skip = 0
    # Start the game loop.
    for __ in range(200):
        valids = env.get_valid((board, turn))
        valids = np.array(list(zip(*valids.nonzero())))
        active_player = black
        if turn == white.player:
            active_player = white
        print(f'{_name[turn]}\'s turn')
        if len(valids) == 0:
            move = env.PASS
        else:
            start_time = time.time()
            try:
                agent_task = asyncio.create_task(
                    active_player.move(board, valids))
                time_task = asyncio.create_task(timer(timelimit))
                done, pending = await asyncio.wait(
                    {time_task, agent_task},
                    timeout=timelimit,
                    return_when=asyncio.FIRST_COMPLETED)
                time_task.cancel()
                agent_task.cancel()
            except asyncio.TimeoutError:
                d = time.time() - start_time - timelimit
                print(f'Timeout! Overtime: {d:.2}')
                break
            move = active_player.best_move
            # TODO CHECK SKIP
            if np.all(move == env.PASS):
                print('NO MOVE! SKIP PLAYER TURN.')
                skip += 1
                await asyncio.sleep(3)
            elif not env.is_valid((board, turn), move):
                print(f'MOVE: {move} IS NOT VALID!')
                skip += 1
                await asyncio.sleep(3)
            else:
                skip = 0
        if skip >= 2:
            break
        clear_screen()
        prev_turn = turn
        board, turn = env.get_next_state((board, turn), move)
        render(board, turn, move, prev_turn)
        winner = env.get_winner((board, turn)) 
        if winner is not None:
            print('='*40)
            if winner == bg2.BLACK:
                print('BLACK wins!')
            elif winner == bg2.WHITE:
                print('WHITE wins!')
            else:
                print('DRAW!')
            break
    if skip == 2:
        print('BOTH PLAYERS SKIPPED.')
        black_score = np.sum(board == bg2.BLACK)
        white_score = np.sum(board == bg2.WHITE)
        if black_score > white_score:
            print('BLACK wins!')
        elif black_score < white_score:
            print('WHITE wins!')
        else:
            print('DRAW!')


if __name__ == "__main__":
    # black = agents.Joke(bg2.BLACK)
    # white = agents.Joke(bg2.WHITE)
    # black = agent3.WarotAgent(bg2.BLACK)
    # white = agent3.WarotAgent(bg2.WHITE)
    black = agent4.AgentMongChaChaVI(bg2.BLACK)
    white = agent4.AgentMongChaChaVI(bg2.WHITE)
    # black = agent5.Joke2(bg2.BLACK)
    # white = agent5.Joke2(bg2.WHITE)
    asyncio.run(main(black, white, 10))

