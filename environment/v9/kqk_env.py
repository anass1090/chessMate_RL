import random

import chess
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import utils.board as b

STEP_PENALTY        = -0.15
CHECKMATE_REWARD    = 10.0
MISSED_MATE_PENALTY = -3.0
QUEEN_HANG_PENALTY  = -5.0


class KQKEnv(gym.Env):
    """
    King + Queen vs King endgame environment — v9.

    Same as v8 but threefold repetition ends the episode as a draw (-1.0),
    discouraging the agent from looping back and forth.
    """

    def __init__(self, max_steps: int = 200, curriculum_ratio: float = 0.5,
                 mate_pool: list[str] | None = None, n_mate_positions: int = 500,
                 movement: str = "centrum"):
        super().__init__()
        self.max_steps        = max_steps
        self.step_count       = 0
        self.curriculum_ratio = curriculum_ratio
        self.movement         = movement
        self.board            = b.random_kqk_position()

        self.observation_space = spaces.Box(0.0, 1.0, shape=(b.OBS_SIZE,), dtype=np.float32)
        self.action_space      = spaces.Discrete(4096)

        if mate_pool is not None:
            self.mate_pool = mate_pool
        elif curriculum_ratio > 0:
            print(f"Generating {n_mate_positions} mate-in-1 positions...", flush=True)
            self.mate_pool = [b.random_kqk_mate_in_one().fen() for _ in range(n_mate_positions)]
            print("Done.", flush=True)
        else:
            self.mate_pool = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.curriculum_ratio > 0 and random.random() < self.curriculum_ratio:
            self.board = chess.Board(random.choice(self.mate_pool))
        else:
            self.board = b.random_kqk_position()
        self.step_count = 0
        return b.board_to_obs(self.board), {}

    def has_mating_move(self) -> bool:
        for move in self.board.legal_moves:
            self.board.push(move)
            is_mate = self.board.is_checkmate()
            self.board.pop()
            if is_mate:
                return True
        return False

    def is_mating_move(self, action: int) -> bool:
        move = b.action_to_move(action)
        self.board.push(move)
        is_mate = self.board.is_checkmate()
        self.board.pop()
        return is_mate

    def is_queen_hanging(self) -> bool:
        queen_bb = self.board.pieces(chess.QUEEN, chess.WHITE)
        if not queen_bb:
            return False
        queen_sq     = next(iter(queen_bb))
        capture_move = chess.Move(self.board.king(chess.BLACK), queen_sq)
        return capture_move in self.board.legal_moves

    def step(self, action: int):
        missed_mate = self.has_mating_move() and not self.is_mating_move(action)

        self.board.push(b.action_to_move(action))
        self.step_count += 1

        if self.board.is_checkmate():
            return b.board_to_obs(self.board), CHECKMATE_REWARD, True, False, {"reason": "checkmate"}

        if missed_mate:
            return b.board_to_obs(self.board), MISSED_MATE_PENALTY, True, False, {"reason": "missed_mate"}

        draw_reason = self.draw_reason()
        if draw_reason:
            return b.board_to_obs(self.board), -1.0, True, False, {"reason": draw_reason}

        if self.is_queen_hanging():
            return b.board_to_obs(self.board), QUEEN_HANG_PENALTY, True, False, {"reason": "queen_hang"}

        self.opponent_move()

        truncated = self.step_count >= self.max_steps
        return b.board_to_obs(self.board), STEP_PENALTY, False, truncated, {"step": self.step_count}

    @staticmethod
    def _centrality(sq: int) -> int:
        return min(chess.square_file(sq), 7 - chess.square_file(sq),
                   chess.square_rank(sq), 7 - chess.square_rank(sq))

    def _random_move(self):
        legal = list(self.board.legal_moves)
        if legal:
            self.board.push(random.choice(legal))

    def _centrum_move(self):
        legal = list(self.board.legal_moves)
        if not legal:
            return
        best_score = max(self._centrality(m.to_square) for m in legal)
        best_moves = [m for m in legal if self._centrality(m.to_square) == best_score]
        self.board.push(random.choice(best_moves))

    def opponent_move(self):
        if self.movement == "random":
            self._random_move()
        else:
            self._centrum_move()

    def draw_reason(self) -> str | None:
        if self.board.is_stalemate():
            return "stalemate"
        if self.board.is_insufficient_material():
            return "insufficient_material"
        if self.board.can_claim_threefold_repetition():
            return "threefold_repetition"
        if self.board.can_claim_fifty_moves():
            return "fifty_moves"
        return None