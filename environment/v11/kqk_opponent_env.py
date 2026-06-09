import random

import chess
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import utils.board as b

STEP_REWARD      =  0.15
CHECKMATE_REWARD = -10.0
DRAW_REWARD      =  5.0


class KQKOpponentEnv(gym.Env):
    """
    KQK environment from black's perspective — black king tries to survive.

    white_fn(obs, legal_actions) -> action is required and controls white's
    moves. Rewards are inverted relative to the white env: surviving is good,
    getting checkmated is bad, any draw is a win.
    """

    def __init__(self, max_steps: int = 200, white_fn=None):
        super().__init__()
        if white_fn is None:
            raise ValueError("white_fn is required — pass a trained white agent's policy.")
        self.max_steps  = max_steps
        self.step_count = 0
        self.white_fn   = white_fn
        self.board      = b.random_kqk_position()

        self.observation_space = spaces.Box(0.0, 1.0, shape=(b.OBS_SIZE,), dtype=np.float32)
        self.action_space      = spaces.Discrete(4096)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Regenerate until white's first move doesn't immediately end the game
        while True:
            self.board      = b.random_kqk_position()
            self.step_count = 0
            self._white_move()
            if not self.board.is_game_over():
                break
        return b.board_to_obs(self.board), {}

    def step(self, action: int):
        # Black makes a move
        self.board.push(b.action_to_move(action))
        self.step_count += 1

        # Draws after black's move (insufficient material if black captured queen,
        # threefold repetition, fifty moves)
        draw = self._draw_reason()
        if draw:
            return b.board_to_obs(self.board), DRAW_REWARD, True, False, {"reason": draw}

        if self.step_count >= self.max_steps:
            return b.board_to_obs(self.board), STEP_REWARD, False, True, {}

        # White responds
        self._white_move()

        # Terminal conditions after white's move
        if self.board.is_checkmate():
            return b.board_to_obs(self.board), CHECKMATE_REWARD, True, False, {"reason": "checkmate"}

        draw = self._draw_reason()
        if draw:
            return b.board_to_obs(self.board), DRAW_REWARD, True, False, {"reason": draw}

        return b.board_to_obs(self.board), STEP_REWARD, False, False, {}

    def _white_move(self):
        legal = [b.move_to_action(m) for m in self.board.legal_moves]
        if not legal:
            return
        action = self.white_fn(b.board_to_obs(self.board), legal)
        self.board.push(b.action_to_move(action))

    def _draw_reason(self) -> str | None:
        if self.board.is_stalemate():
            return "stalemate"
        if self.board.is_insufficient_material():
            return "insufficient_material"
        if self.board.can_claim_threefold_repetition():
            return "threefold_repetition"
        if self.board.can_claim_fifty_moves():
            return "fifty_moves"
        return None
