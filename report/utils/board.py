import chess
import numpy as np
from multiprocessing import Pool

# 6 piece types × 2 colors × 64 squares
OBS_SIZE = 768


def random_kqk_position() -> chess.Board:
    """
        Generate a random legal King + Queen versus solo King board position.
    """
    board = chess.Board(fen=None)
    while True:
        board.clear()
        squares = np.random.choice(64, size=3, replace=False)
        board.set_piece_at(int(squares[0]), chess.Piece(chess.KING,  chess.WHITE))
        board.set_piece_at(int(squares[1]), chess.Piece(chess.QUEEN, chess.WHITE))
        board.set_piece_at(int(squares[2]), chess.Piece(chess.KING,  chess.BLACK))
        board.turn = chess.WHITE
        if board.is_valid() and not board.is_game_over():
            return board


def board_to_obs(board: chess.Board) -> np.ndarray:
    """
        Encode the board as a 768-element vector.
        The network needs a fixed-size number representation of the board; this is it.
    """
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    for sq, piece in board.piece_map().items():
        color_idx = 0 if piece.color == chess.WHITE else 1
        obs[(piece.piece_type - 1) * 128 + color_idx * 64 + sq] = 1.0
    return obs


def action_to_move(action: int) -> chess.Move:
    """
        Turn a number (0–4095) into a chess move (from_sq * 64 + to_sq).
        The network outputs a number; this converts it into something python-chess understands.
    """
    return chess.Move(action // 64, action % 64)


def random_kqk_mate_in_one() -> chess.Board:
    """Return a random KQK position where white has at least one mate-in-one."""
    while True:
        board = random_kqk_position()
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return board
            board.pop()


def _gen_mate_in_one(_) -> str:
    return random_kqk_mate_in_one().fen()


def generate_mate_pool(n: int = 500) -> list[str]:
    """Generate n mate-in-1 FENs in parallel across all CPU cores."""
    with Pool() as p:
        return p.map(_gen_mate_in_one, range(n))


def move_to_action(move: chess.Move) -> int:
    """
        Turn a chess move back into a number.
        We need this to look up which network outputs correspond to legal moves.
    """
    return move.from_square * 64 + move.to_square