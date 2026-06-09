import random

import chess
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import utils.board as b

STEP_PENALTY   = -0.03   # v1 was -0.001; makes 50-move shuffling cost -1.5 vs -1.0 terminal draw
QUEEN_HANG_PENALTY = -0.5   # applied immediately when the queen is left en prise to the black king


class KQKEnv(gym.Env):
    """
    King + Queen vs King endgame environment — v2.

    Changes vs v1:
    - Step penalty raised from -0.001 to -0.01 so fifty-move draws are
      meaningfully costly and the agent has urgency to make progress.
    - Queen-hang penalty: if white's move leaves the queen on a square
      attacked by the black king, a -0.5 penalty is applied on that step.
      This signals the mistake at the exact move that causes it rather than
      one step later when insufficient_material is detected.

    Observation: float32 vector of shape (768,)
    Action:      int in [0, 4095]  (caller must pass a legal move)
    Rewards:     +1 checkmate | -1 draw | -0.01/step | -0.5 queen hang
    """

    def __init__(self, max_steps: int = 200):
        super().__init__()
        self.max_steps  = max_steps
        self.step_count = 0
        self.board      = b.random_kqk_position()

        self.observation_space = spaces.Box(0.0, 1.0, shape=(b.OBS_SIZE,), dtype=np.float32)
        self.action_space      = spaces.Discrete(4096)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board      = b.random_kqk_position()
        self.step_count = 0
        return b.board_to_obs(self.board), {}

    def step(self, action: int):
        self.board.push(b.action_to_move(action))
        self.step_count += 1

        if self.board.is_checkmate():
            return b.board_to_obs(self.board), 1.0, True, False, {"reason": "checkmate"}

        draw_reason = self.draw_reason()
        if draw_reason:
            return b.board_to_obs(self.board), -1.0, True, False, {"reason": draw_reason}

        # Check queen hang before opponent moves so the penalty is on the causal step.
        # board.king() is O(1) via a cached bitboard; is_attacked_by() is also fast.
        queen_bb = self.board.pieces(chess.QUEEN, chess.WHITE)
        hang = QUEEN_HANG_PENALTY if queen_bb and self.board.is_attacked_by(chess.BLACK, next(iter(queen_bb))) else 0.0

        self.opponent_move()

        truncated = self.step_count >= self.max_steps
        return b.board_to_obs(self.board), STEP_PENALTY + hang, False, truncated, {"step": self.step_count}

    def opponent_move(self):
        legal = list(self.board.legal_moves)
        if legal:
            self.board.push(random.choice(legal))

    def draw_reason(self) -> str | None:
        if self.board.is_stalemate():
            return "stalemate"
        if self.board.is_insufficient_material():
            return "insufficient_material"
        if self.board.can_claim_fifty_moves():
            return "fifty_moves"
        return None
