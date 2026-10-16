"""Tactical start-position generation for Gomoku self-play.

These positions are not supervised labels. They are intended as starting
states for MCTS self-play so the model sees more immediate wins, forced
blocks, and open-four positions during training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from gomoku_env import BOARD_SIZE, WIN_COUNT, GomokuEnv


DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))
TACTICAL_TYPES = (
    "immediate_win",
    "immediate_block",
    "own_open_four",
    "opponent_open_four",
)


@dataclass(frozen=True)
class TacticalPosition:
    env: GomokuEnv
    kind: str
    critical_actions: tuple[int, ...]


def generate_tactical_position(
    board_size: int = BOARD_SIZE,
    win_count: int = WIN_COUNT,
    rng: np.random.Generator | None = None,
    kind: str | None = None,
    extra_noise_pairs: int = 1,
    max_attempts: int = 200,
) -> TacticalPosition:
    """Generate one legal tactical Gomoku position.

    The returned env is ready to use as an MCTS/self-play start state. The
    board is balanced for the side to move, contains no existing five-in-a-row,
    and preserves the requested tactical pattern.
    """
    if board_size < win_count:
        raise ValueError(f"board_size={board_size} is smaller than win_count={win_count}.")

    rng = rng or np.random.default_rng()
    allowed = TACTICAL_TYPES if kind is None else (kind,)
    unknown = [value for value in allowed if value not in TACTICAL_TYPES]
    if unknown:
        raise ValueError(f"Unknown tactical type: {unknown[0]}")

    for _ in range(max_attempts):
        selected_kind = str(rng.choice(allowed))
        env = GomokuEnv(board_size=board_size, win_count=win_count)
        env.reset()
        current_player = int(rng.choice([1, -1]))
        env.turn = current_player

        protected, critical = _place_core_pattern(env, selected_kind, rng)
        if not protected:
            continue

        if not _balance_side_to_move(env, protected, rng):
            continue
        if not _add_noise_pairs(env, protected, extra_noise_pairs, rng):
            continue
        if _has_any_winner(env):
            continue
        if not _critical_actions_still_work(env, selected_kind, current_player, critical):
            continue

        _refresh_history(env)
        return TacticalPosition(
            env=env,
            kind=selected_kind,
            critical_actions=tuple(env.rc_to_action(r, c) for r, c in critical),
        )

    raise RuntimeError(f"Failed to generate tactical position after {max_attempts} attempts.")


def _place_core_pattern(
    env: GomokuEnv,
    kind: str,
    rng: np.random.Generator,
) -> tuple[set[tuple[int, int]], list[tuple[int, int]]]:
    if kind in ("own_open_four", "opponent_open_four"):
        return _place_open_four(env, player=_pattern_player(env, kind), rng=rng)
    return _place_four_in_five(env, player=_pattern_player(env, kind), rng=rng)


def _pattern_player(env: GomokuEnv, kind: str) -> int:
    if kind in ("immediate_win", "own_open_four"):
        return env.turn
    return -env.turn


def _place_four_in_five(
    env: GomokuEnv,
    player: int,
    rng: np.random.Generator,
) -> tuple[set[tuple[int, int]], list[tuple[int, int]]]:
    cells = _sample_line(env.board_size, env.win_count, rng)
    gap_idx = int(rng.integers(0, env.win_count))
    critical = [cells[gap_idx]]
    protected = set(cells)

    for idx, (row, col) in enumerate(cells):
        if idx != gap_idx:
            env.board[row, col] = player
    return protected, critical


def _place_open_four(
    env: GomokuEnv,
    player: int,
    rng: np.random.Generator,
) -> tuple[set[tuple[int, int]], list[tuple[int, int]]]:
    cells = _sample_line(env.board_size, env.win_count + 1, rng)
    critical = [cells[0], cells[-1]]
    protected = set(cells)

    for row, col in cells[1:-1]:
        env.board[row, col] = player
    return protected, critical


def _sample_line(board_size: int, length: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    candidates = []
    for dr, dc in DIRECTIONS:
        for row in range(board_size):
            for col in range(board_size):
                end_row = row + dr * (length - 1)
                end_col = col + dc * (length - 1)
                if 0 <= end_row < board_size and 0 <= end_col < board_size:
                    candidates.append([(row + dr * i, col + dc * i) for i in range(length)])
    return list(candidates[int(rng.integers(0, len(candidates)))])


def _balance_side_to_move(
    env: GomokuEnv,
    protected: set[tuple[int, int]],
    rng: np.random.Generator,
) -> bool:
    target_diff = 0 if env.turn == 1 else 1

    for _ in range(env.board_size * env.board_size):
        black_count, white_count = _stone_counts(env.board)
        diff = black_count - white_count
        if diff == target_diff:
            return True

        player = 1 if diff < target_diff else -1
        if not _place_random_stone(env, player, protected, rng):
            return False

    return False


def _add_noise_pairs(
    env: GomokuEnv,
    protected: set[tuple[int, int]],
    pairs: int,
    rng: np.random.Generator,
) -> bool:
    if pairs <= 0:
        return True

    for _ in range(pairs):
        if not _place_random_stone(env, 1, protected, rng):
            return False
        if not _place_random_stone(env, -1, protected, rng):
            return False
    return True


def _place_random_stone(
    env: GomokuEnv,
    player: int,
    protected: set[tuple[int, int]],
    rng: np.random.Generator,
) -> bool:
    empty_cells = [
        (int(row), int(col))
        for row, col in np.argwhere(env.board == 0)
        if (int(row), int(col)) not in protected
    ]
    rng.shuffle(empty_cells)

    for row, col in empty_cells:
        env.board[row, col] = player
        if not env._check_win(row, col, player):
            return True
        env.board[row, col] = 0
    return False


def _critical_actions_still_work(
    env: GomokuEnv,
    kind: str,
    current_player: int,
    critical: Iterable[tuple[int, int]],
) -> bool:
    pattern_player = current_player if kind in ("immediate_win", "own_open_four") else -current_player
    critical_cells = list(critical)
    for row, col in critical_cells:
        if env.board[row, col] != 0:
            return False

    if kind in ("immediate_win", "immediate_block"):
        for row, col in critical_cells:
            test_env = env.copy()
            test_env.board[row, col] = pattern_player
            if test_env._check_win(row, col, pattern_player):
                return True
        return False

    return True


def _has_any_winner(env: GomokuEnv) -> bool:
    occupied = np.argwhere(env.board != 0)
    for row, col in occupied:
        if env._check_win(int(row), int(col), int(env.board[row, col])):
            return True
    return False


def _stone_counts(board: np.ndarray) -> tuple[int, int]:
    return int(np.sum(board == 1)), int(np.sum(board == -1))


def _refresh_history(env: GomokuEnv):
    env.done = False
    env.winner = 0
    env.move_history = [
        (int(row), int(col), int(env.board[row, col]))
        for row, col in np.argwhere(env.board != 0)
    ]

    opponent_stones = [(row, col) for row, col, player in env.move_history if player == -env.turn]
    if opponent_stones:
        env.last_move = opponent_stones[-1]
    elif env.move_history:
        env.last_move = (env.move_history[-1][0], env.move_history[-1][1])
    else:
        env.last_move = None
