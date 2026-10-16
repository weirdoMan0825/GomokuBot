"""Rule-based Gomoku benchmark agent.

The agent is intentionally simple and deterministic enough to serve as a
baseline for measuring whether the RL model is learning real game knowledge.
It checks tactical wins/blocks first, then scores candidate moves with local
five-cell line patterns.
"""

from __future__ import annotations

import math

import numpy as np

from gomoku_env import GomokuEnv


class HeuristicAgent:
    """A lightweight benchmark agent for 9x9 Gomoku."""

    def __init__(self, player: int | None = None, defense_weight: float = 0.9, random_tie_break: bool = True):
        self.player = player
        self.defense_weight = defense_weight
        self.random_tie_break = random_tie_break

    def act(self, env: GomokuEnv) -> int:
        player = env.turn if self.player is None else env.turn
        legal_actions = env.valid_actions()
        if not legal_actions:
            raise RuntimeError("No legal actions are available.")

        center = env.board_size // 2
        center_action = env.rc_to_action(center, center)
        if len(legal_actions) == env.board_size * env.board_size and center_action in legal_actions:
            return center_action

        for action in legal_actions:
            if self._is_winning_move(env, action, player):
                return action

        opponent = -player
        blocking_moves = [action for action in legal_actions if self._is_winning_move(env, action, opponent)]
        if blocking_moves:
            return self._best_positional_move(env, blocking_moves)

        best_score = -float("inf")
        best_actions = []
        for action in legal_actions:
            score = self._score_action(env, action, player)
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif math.isclose(score, best_score):
                best_actions.append(action)

        if self.random_tie_break and len(best_actions) > 1:
            return int(np.random.choice(best_actions))
        return self._best_positional_move(env, best_actions)

    def _is_winning_move(self, env: GomokuEnv, action: int, player: int) -> bool:
        row, col = env.action_to_rc(action)
        if not env.is_valid_move(row, col):
            return False
        test_env = env.copy()
        test_env.board[row, col] = player
        return test_env._check_win(row, col, player)

    def _score_action(self, env: GomokuEnv, action: int, player: int) -> float:
        row, col = env.action_to_rc(action)
        test_board = env.board.copy()
        test_board[row, col] = player

        own_score = self._score_board(test_board, player)
        opp_score = self._score_board(test_board, -player)
        center_bonus = self._center_bonus(env, row, col)
        return own_score - self.defense_weight * opp_score + center_bonus

    def _score_board(self, board: np.ndarray, player: int) -> float:
        size = board.shape[0]
        weights = {
            1: 1.0,
            2: 10.0,
            3: 80.0,
            4: 1500.0,
            5: 100000.0,
        }

        score = 0.0
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for row in range(size):
            for col in range(size):
                for dr, dc in directions:
                    cells = []
                    for offset in range(5):
                        r = row + dr * offset
                        c = col + dc * offset
                        if not (0 <= r < size and 0 <= c < size):
                            cells = []
                            break
                        cells.append(board[r, c])

                    if not cells:
                        continue
                    if -player in cells:
                        continue

                    count = cells.count(player)
                    if count == 0:
                        continue

                    line_score = weights[count]
                    if count in (2, 3, 4):
                        line_score *= self._open_end_multiplier(board, row, col, dr, dc, player)
                    score += line_score

        return score

    def _open_end_multiplier(self, board: np.ndarray, row: int, col: int, dr: int, dc: int, player: int) -> float:
        size = board.shape[0]
        before_r, before_c = row - dr, col - dc
        after_r, after_c = row + dr * 5, col + dc * 5
        open_ends = 0
        if 0 <= before_r < size and 0 <= before_c < size and board[before_r, before_c] == 0:
            open_ends += 1
        if 0 <= after_r < size and 0 <= after_c < size and board[after_r, after_c] == 0:
            open_ends += 1
        if open_ends == 2:
            return 1.5
        if open_ends == 1:
            return 1.1
        return 1.0

    def _center_bonus(self, env: GomokuEnv, row: int, col: int) -> float:
        center = (env.board_size - 1) / 2.0
        distance = abs(row - center) + abs(col - center)
        return max(0.0, env.board_size - distance)

    def _best_positional_move(self, env: GomokuEnv, actions: list[int]) -> int:
        center = (env.board_size - 1) / 2.0
        return min(
            actions,
            key=lambda action: (
                abs(env.action_to_rc(action)[0] - center) + abs(env.action_to_rc(action)[1] - center),
                action,
            ),
        )
