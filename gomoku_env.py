"""
Gomoku environment.

The environment uses a Gym-like interface:
    state: 2D numpy array, shape = (board_size, board_size)
           0 = empty, 1 = black, -1 = white
    action: integer in [0, board_size * board_size)
            action = row * board_size + col
    turn: 1 = black to move, -1 = white to move
"""

import numpy as np


BOARD_SIZE = 9
WIN_COUNT = 5
COL_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class GomokuEnv:
    """
    Gomoku game environment.

    The reward is from the current player's perspective at the moment an
    action is applied:
        1.0  = current player wins
        -1.0 = current player makes an illegal move and loses
        0.0  = draw or non-terminal move
    """

    def __init__(self, board_size: int = BOARD_SIZE, win_count: int = WIN_COUNT):
        self.board_size = board_size
        self.win_count = win_count
        self.board: np.ndarray = np.zeros((board_size, board_size), dtype=np.int8)
        self.turn: int = 1
        self.done: bool = False
        self.winner: int = 0
        self.last_move: tuple[int, int] | None = None
        self.move_history: list[tuple[int, int, int]] = []

    def reset(self) -> np.ndarray:
        """Reset the board and return the initial state."""
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.turn = 1
        self.done = False
        self.winner = 0
        self.last_move = None
        self.move_history = []
        return self.board.copy()

    def step(self, action: int):
        """
        Apply one move.

        Parameters
        ----------
        action:
            Integer action encoded as row * board_size + col.

        Returns
        -------
        state:
            Board after the action.
        reward:
            1.0 for a win, -1.0 for an illegal move, 0.0 otherwise.
        done:
            True if the game is finished.
        info:
            Extra result information.
        """
        if self.done:
            raise RuntimeError("The game is already finished. Call reset() first.")

        row, col = divmod(action, self.board_size)

        if not self.is_valid_move(row, col):
            self.done = True
            self.winner = -self.turn
            return self.board.copy(), -1.0, True, {"reason": "illegal_move", "winner": self.winner}

        self.board[row, col] = self.turn
        self.last_move = (row, col)
        self.move_history.append((row, col, self.turn))

        if self._check_win(row, col, self.turn):
            self.done = True
            self.winner = self.turn
            reward = 1.0
            info = {"reason": "win", "winner": self.turn}
        elif np.all(self.board != 0):
            self.done = True
            self.winner = 0
            reward = 0.0
            info = {"reason": "draw"}
        else:
            reward = 0.0
            info = {}
            self.turn = -self.turn

        return self.board.copy(), reward, self.done, info

    def is_valid_move(self, row: int, col: int) -> bool:
        """Return True if the given board coordinate is empty and on the board."""
        return (
            0 <= row < self.board_size
            and 0 <= col < self.board_size
            and self.board[row, col] == 0
        )

    def valid_actions(self) -> list[int]:
        """Return all legal moves as flattened integer actions."""
        empties = np.argwhere(self.board == 0)
        return [int(r * self.board_size + c) for r, c in empties]

    def action_to_rc(self, action: int) -> tuple[int, int]:
        """Convert a flattened action to a row, col pair."""
        return divmod(action, self.board_size)

    def rc_to_action(self, row: int, col: int) -> int:
        """Convert a row, col pair to a flattened action."""
        return row * self.board_size + col

    def get_state(self) -> np.ndarray:
        """Return a copy of the current raw board."""
        return self.board.copy()

    def copy(self) -> "GomokuEnv":
        """Return a deep copy of this environment state."""
        new_env = GomokuEnv(self.board_size, self.win_count)
        new_env.board = self.board.copy()
        new_env.turn = self.turn
        new_env.done = self.done
        new_env.winner = self.winner
        new_env.last_move = self.last_move
        new_env.move_history = self.move_history[:]
        return new_env

    def get_state_for_current_player(self) -> np.ndarray:
        """
        Return the board from the current player's perspective.

        Current player's stones are 1, opponent stones are -1, empty cells are 0.
        This representation is useful for reinforcement learning agents.
        """
        return self.board * self.turn

    def _check_win(self, row: int, col: int, player: int) -> bool:
        """Check whether the latest move creates a five-in-a-row line."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while (
                    0 <= r < self.board_size
                    and 0 <= c < self.board_size
                    and self.board[r, c] == player
                ):
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= self.win_count:
                return True
        return False

    def check_win_at(self, row: int, col: int) -> int:
        """Return the winning player at row, col, or 0 if there is no win."""
        player = self.board[row, col]
        if player == 0:
            return 0
        return int(player) if self._check_win(row, col, int(player)) else 0

    def render(self):
        """Print the board to the terminal."""
        symbols = {0: ".", 1: "B", -1: "W"}
        col_labels = "    " + "  ".join(COL_LABELS[:self.board_size])
        print(col_labels)
        for r in range(self.board_size):
            row_str = f"{r + 1:2} " + "  ".join(symbols[int(self.board[r, c])] for c in range(self.board_size))
            print(row_str)

        turn_str = "Black" if self.turn == 1 else "White"
        print(f"\nTurn: {turn_str}. Moves played: {len(self.move_history)}")
        if self.done:
            if self.winner == 1:
                print("=== Black wins ===")
            elif self.winner == -1:
                print("=== White wins ===")
            else:
                print("=== Draw ===")

    def __repr__(self):
        turn = "Black" if self.turn == 1 else "White"
        return f"GomokuEnv(size={self.board_size}, win={self.win_count}, turn={turn}, done={self.done})"


class RandomAgent:
    """Agent that selects a random legal move."""

    def __init__(self, player: int):
        self.player = player

    def act(self, env: GomokuEnv) -> int:
        actions = env.valid_actions()
        return int(np.random.choice(actions))


if __name__ == "__main__":
    env = GomokuEnv(board_size=BOARD_SIZE, win_count=WIN_COUNT)
    black = RandomAgent(1)
    white = RandomAgent(-1)

    env.reset()
    env.render()

    info = {}
    while not env.done:
        agent = black if env.turn == 1 else white
        action = agent.act(env)
        r, c = env.action_to_rc(action)
        player = "Black" if env.turn == 1 else "White"
        print(f"\n{player} plays ({r}, {c})")
        _, _, _, info = env.step(action)

    env.render()
    print(f"\ninfo: {info}")
