import random

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import utils.board as b


class KQKEnv(gym.Env):
    """
        King + Queen vs King endgame environment.
        Observation: float32 vector of shape (768,) | Action: int in [0, 4095] | Reward: +1 checkmate, -0.5 draw, -1 illegal, -0.001 per step
    """

    def __init__(self, max_steps: int = 200):
        """
            Set up the board, action space, and observation space.
        """
        super().__init__()
        self.max_steps  = max_steps
        self.step_count = 0
        self.board      = b.random_kqk_position()

        self.observation_space = spaces.Box(0.0, 1.0, shape=(b.OBS_SIZE,), dtype=np.float32)
        self.action_space      = spaces.Discrete(4096)

    def reset(self, seed=None, options=None):
        """
            Start a new episode from a fresh random position.
        """
        super().reset(seed=seed)
        self.board      = b.random_kqk_position()
        self.step_count = 0
        return b.board_to_obs(self.board), {}

    def step(self, action: int):
        """
            Apply the agent's move, then let the opponent play a random legal move.
        """
        move = b.action_to_move(action)

        if move not in self.board.legal_moves:
            return b.board_to_obs(self.board), -1.0, True, False, {"reason": "illegal"}

        self.board.push(move)
        self.step_count += 1

        if self.board.is_checkmate():
            return b.board_to_obs(self.board), 1.0, True, False, {"reason": "checkmate"}

        draw_reason = self.draw_reason()
        if draw_reason:
            return b.board_to_obs(self.board), -1.0, True, False, {"reason": draw_reason}

        self.opponent_move()

        truncated = self.step_count >= self.max_steps
        reward = -0.001

        return b.board_to_obs(self.board), reward, False, truncated, {"step": self.step_count}

    def opponent_move(self):
        """
            Pick a random legal move for the black king.
            In v1 the opponent is purely random; we will improve this later.
        """
        legal = list(self.board.legal_moves)
        if legal:
            self.board.push(random.choice(legal))

    def draw_reason(self) -> str | None:
        """Return the draw reason string if the position is a draw, else None."""
        if self.board.is_stalemate():
            return "stalemate"
        if self.board.is_insufficient_material():
            return "insufficient_material"
        if self.board.can_claim_fifty_moves():
            return "fifty_moves"
        if self.board.can_claim_threefold_repetition():
            return "threefold_repetition"
        return None
