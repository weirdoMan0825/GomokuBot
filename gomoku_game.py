"""
Gomoku GUI.

Run:
    python gomoku_game.py
"""

from pathlib import Path
import tkinter as tk
from tkinter import messagebox

from device_utils import default_device
from gomoku_env import BOARD_SIZE, COL_LABELS, GomokuEnv, RandomAgent


CELL_SIZE = 52
MARGIN = 32
STONE_RADIUS = 20

BG_COLOR = "#DCB468"
LINE_COLOR = "#5A3E1B"
BLACK_STONE = "#1A1A1A"
WHITE_STONE = "#F5F5F5"
HIGHLIGHT_COLOR = "#FF4444"
HISTORY_WIDTH = 24
EVAL_BAR_WIDTH = 54
EVAL_BAR_INNER_WIDTH = 18
EVAL_BAR_BLACK = "#2B2925"
EVAL_BAR_WHITE = "#F4EFE5"
EVAL_BAR_NEUTRAL = "#6B6258"
TOP_MOVE_COUNT = 5
TOP_MOVE_COLOR = "#1F7A4D"
CHECKPOINT_DIR = Path(__file__).with_name("checkpoints")
BOARD_SIZE_OPTIONS = (9, 11)
AI_SIMULATIONS = {
    "Easy": 10,
    "Normal": 50,
    "Hard": 200,
    "Expert": 400,
}


class GomokuGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Gomoku")
        self.root.resizable(False, False)

        self.board_size = tk.IntVar(value=BOARD_SIZE)
        self.env = GomokuEnv(board_size=self.board_size.get(), win_count=5)
        self.mode = tk.StringVar(value="pvp")
        self.ai_color = tk.IntVar(value=-1)
        self._checkpoint_size_cache = {}
        self.checkpoints = self._list_checkpoints()
        self.checkpoint_path = self._find_latest_checkpoint()
        checkpoint_name = self.checkpoint_path.name if self.checkpoint_path is not None else "No checkpoint"
        self.checkpoint_choice = tk.StringVar(value=checkpoint_name)
        self.ai_engine = tk.StringVar(value="AlphaZero" if self.checkpoint_path is not None else "Random")
        self.ai_strength = tk.StringVar(value="Normal")
        default_match_engine = "AlphaZero" if self.checkpoint_path is not None else "Heuristic"
        self.black_ai_engine = tk.StringVar(value=default_match_engine)
        self.white_ai_engine = tk.StringVar(value=default_match_engine)
        self.black_checkpoint_path = self.checkpoint_path
        self.white_checkpoint_path = self.checkpoint_path
        self.black_checkpoint_choice = tk.StringVar(value=checkpoint_name)
        self.white_checkpoint_choice = tk.StringVar(value=checkpoint_name)
        self.show_eval_bar = tk.BooleanVar(value=True)
        self.show_top_moves = tk.BooleanVar(value=True)
        self.ai_device = default_device()
        self._agent_cache = {}
        self._eval_model_cache = {}
        self._analysis_cache = {}
        self.checkpoint_menu = None
        self.black_checkpoint_menu = None
        self.white_checkpoint_menu = None
        self._ai_delay = 400

        self._build_ui()
        self.new_game()

    def _build_ui(self):
        top = tk.Frame(self.root, bg="#3B2A1A", pady=6)
        top.pack(fill="x")

        tk.Label(top, text="Mode:", bg="#3B2A1A", fg="white", font=("Arial", 11)).pack(side="left", padx=(12, 2))
        for text, val in [("Human vs Human", "pvp"), ("Human vs AI", "pvc"), ("AI vs AI", "aivai")]:
            tk.Radiobutton(
                top,
                text=text,
                variable=self.mode,
                value=val,
                bg="#3B2A1A",
                fg="white",
                selectcolor="#3B2A1A",
                font=("Arial", 11),
                command=self.new_game,
            ).pack(side="left", padx=4)

        tk.Label(top, text="  AI color:", bg="#3B2A1A", fg="white", font=("Arial", 11)).pack(side="left", padx=(12, 2))
        for text, val in [("White", -1), ("Black", 1)]:
            tk.Radiobutton(
                top,
                text=text,
                variable=self.ai_color,
                value=val,
                bg="#3B2A1A",
                fg="white",
                selectcolor="#3B2A1A",
                font=("Arial", 11),
                command=self.new_game,
            ).pack(side="left", padx=2)

        tk.Button(
            top,
            text="New Game",
            font=("Arial", 11, "bold"),
            bg="#C0392B",
            fg="white",
            relief="flat",
            padx=10,
            command=self.new_game,
        ).pack(side="right", padx=12)

        tk.Button(
            top,
            text="Undo",
            font=("Arial", 11),
            bg="#2980B9",
            fg="white",
            relief="flat",
            padx=10,
            command=self.undo_move,
        ).pack(side="right", padx=4)

        tk.Label(top, text="Board:", bg="#3B2A1A", fg="white", font=("Arial", 11)).pack(side="right", padx=(8, 2))
        board_menu = tk.OptionMenu(
            top,
            self.board_size,
            *BOARD_SIZE_OPTIONS,
            command=lambda _: self._on_board_size_change(),
        )
        board_menu.configure(font=("Arial", 10), bg="#E8D6B8", fg="#1F160E", relief="flat", highlightthickness=0)
        board_menu.pack(side="right", padx=4)

        ai_bar = tk.Frame(self.root, bg="#4A3521", pady=5)
        ai_bar.pack(fill="x")

        tk.Label(ai_bar, text="AI Engine:", bg="#4A3521", fg="white", font=("Arial", 10)).pack(side="left", padx=(12, 3))
        ai_menu = tk.OptionMenu(ai_bar, self.ai_engine, "Random", "Heuristic", "AlphaZero", command=lambda _: self.new_game())
        ai_menu.configure(font=("Arial", 10), bg="#E8D6B8", fg="#1F160E", relief="flat", highlightthickness=0)
        ai_menu.pack(side="left", padx=4)

        tk.Label(ai_bar, text="Strength:", bg="#4A3521", fg="white", font=("Arial", 10)).pack(side="left", padx=(12, 3))
        strength_menu = tk.OptionMenu(ai_bar, self.ai_strength, *AI_SIMULATIONS.keys(), command=lambda _: self.new_game())
        strength_menu.configure(font=("Arial", 10), bg="#E8D6B8", fg="#1F160E", relief="flat", highlightthickness=0)
        strength_menu.pack(side="left", padx=4)

        tk.Label(ai_bar, text="Checkpoint:", bg="#4A3521", fg="white", font=("Arial", 10)).pack(side="left", padx=(12, 3))
        checkpoint_names = self._checkpoint_names()
        self.checkpoint_menu = tk.OptionMenu(
            ai_bar,
            self.checkpoint_choice,
            *checkpoint_names,
            command=self._on_checkpoint_change,
        )
        self.checkpoint_menu.configure(font=("Arial", 10), bg="#E8D6B8", fg="#1F160E", relief="flat", highlightthickness=0)
        self.checkpoint_menu.pack(side="left", padx=4)

        tk.Button(
            ai_bar,
            text="Refresh",
            font=("Arial", 10),
            bg="#6B4A2E",
            fg="white",
            relief="flat",
            padx=8,
            command=self._refresh_checkpoints,
        ).pack(side="left", padx=(4, 8))

        tk.Checkbutton(
            ai_bar,
            text="Eval Bar",
            variable=self.show_eval_bar,
            bg="#4A3521",
            fg="white",
            selectcolor="#3B2A1A",
            activebackground="#4A3521",
            activeforeground="white",
            font=("Arial", 10),
            command=self._refresh_analysis_overlays,
        ).pack(side="left", padx=(8, 4))

        tk.Checkbutton(
            ai_bar,
            text="Top Moves",
            variable=self.show_top_moves,
            bg="#4A3521",
            fg="white",
            selectcolor="#3B2A1A",
            activebackground="#4A3521",
            activeforeground="white",
            font=("Arial", 10),
            command=self._refresh_analysis_overlays,
        ).pack(side="left", padx=4)

        match_bar = tk.Frame(self.root, bg="#5A3E1B", pady=5)
        match_bar.pack(fill="x")

        tk.Label(match_bar, text="AI Match:", bg="#5A3E1B", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=(12, 6))

        tk.Label(match_bar, text="Black", bg="#5A3E1B", fg="white", font=("Arial", 10)).pack(side="left", padx=(4, 3))
        black_engine_menu = tk.OptionMenu(match_bar, self.black_ai_engine, "Random", "Heuristic", "AlphaZero", command=lambda _: self.new_game())
        black_engine_menu.configure(font=("Arial", 10), bg="#E8D6B8", fg="#1F160E", relief="flat", highlightthickness=0)
        black_engine_menu.pack(side="left", padx=3)

        self.black_checkpoint_menu = tk.OptionMenu(
            match_bar,
            self.black_checkpoint_choice,
            *checkpoint_names,
            command=lambda selected: self._on_side_checkpoint_change(1, selected),
        )
        self.black_checkpoint_menu.configure(font=("Arial", 10), bg="#E8D6B8", fg="#1F160E", relief="flat", highlightthickness=0)
        self.black_checkpoint_menu.pack(side="left", padx=(2, 10))

        tk.Label(match_bar, text="White", bg="#5A3E1B", fg="white", font=("Arial", 10)).pack(side="left", padx=(4, 3))
        white_engine_menu = tk.OptionMenu(match_bar, self.white_ai_engine, "Random", "Heuristic", "AlphaZero", command=lambda _: self.new_game())
        white_engine_menu.configure(font=("Arial", 10), bg="#E8D6B8", fg="#1F160E", relief="flat", highlightthickness=0)
        white_engine_menu.pack(side="left", padx=3)

        self.white_checkpoint_menu = tk.OptionMenu(
            match_bar,
            self.white_checkpoint_choice,
            *checkpoint_names,
            command=lambda selected: self._on_side_checkpoint_change(-1, selected),
        )
        self.white_checkpoint_menu.configure(font=("Arial", 10), bg="#E8D6B8", fg="#1F160E", relief="flat", highlightthickness=0)
        self.white_checkpoint_menu.pack(side="left", padx=(2, 10))

        main = tk.Frame(self.root, bg="#2E2115")
        main.pack(fill="both", expand=True)

        eval_frame = tk.Frame(main, bg="#2E2115")
        eval_frame.pack(side="left", padx=(8, 0), pady=(4, 0))

        self.eval_canvas = tk.Canvas(
            eval_frame,
            width=EVAL_BAR_WIDTH,
            height=self._canvas_size(),
            bg="#2E2115",
            highlightthickness=0,
        )
        self.eval_canvas.pack()

        board_frame = tk.Frame(main, bg="#2E2115")
        board_frame.pack(side="left", padx=(4, 8), pady=(4, 0))

        self.canvas = tk.Canvas(
            board_frame,
            width=self._canvas_size(),
            height=self._canvas_size(),
            bg=BG_COLOR,
            highlightthickness=0,
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_click)

        history_frame = tk.Frame(main, bg="#2E2115", padx=8, pady=8)
        history_frame.pack(side="right", fill="y")

        tk.Label(
            history_frame,
            text="Move History",
            font=("Arial", 12, "bold"),
            bg="#2E2115",
            fg="white",
        ).pack(anchor="w", pady=(0, 6))

        history_body = tk.Frame(history_frame, bg="#2E2115")
        history_body.pack(fill="both", expand=True)

        self.history_text = tk.Text(
            history_body,
            width=HISTORY_WIDTH,
            height=24,
            font=("Consolas", 10),
            bg="#F5E4C2",
            fg="#1F160E",
            relief="flat",
            padx=8,
            pady=8,
            wrap="none",
            state="disabled",
        )
        self.history_text.pack(side="left", fill="both", expand=True)

        history_scroll = tk.Scrollbar(history_body, command=self.history_text.yview)
        history_scroll.pack(side="right", fill="y")
        self.history_text.configure(yscrollcommand=history_scroll.set)

        self.status_var = tk.StringVar()
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Arial", 12, "bold"),
            bg="#3B2A1A",
            fg="white",
            pady=6,
        )
        status_bar.pack(fill="x")

    def new_game(self):
        if self.env.board_size != self.board_size.get():
            self.env = GomokuEnv(board_size=self.board_size.get(), win_count=5)
        else:
            self.env.reset()
        self._draw_board()
        self._update_status()
        if self.mode.get() == "pvc" and self.ai_color.get() == self.env.turn:
            self.root.after(self._ai_delay, self._ai_move)
        elif self.mode.get() == "aivai":
            self.root.after(self._ai_delay, self._ai_move)

    def _on_click(self, event):
        if self.env.done:
            return
        if self.mode.get() == "aivai":
            return
        if self.mode.get() == "pvc" and self.env.turn == self.ai_color.get():
            return

        col = round((event.x - MARGIN) / CELL_SIZE)
        row = round((event.y - MARGIN) / CELL_SIZE)

        if not self.env.is_valid_move(row, col):
            return

        action = self.env.rc_to_action(row, col)
        _, _, done, info = self.env.step(action)
        self._draw_stones()
        self._update_status()

        if done:
            self._show_result(info)
        elif self.mode.get() == "pvc":
            self.root.after(self._ai_delay, self._ai_move)

    def _ai_move(self):
        if self.env.done:
            return

        engine_name = self._current_ai_engine_name()
        self.status_var.set(f"{engine_name} is thinking...")
        self.root.update_idletasks()

        try:
            agent = self._create_ai_agent()
            action = agent.act(self.env)
        except Exception as exc:
            messagebox.showerror("AI Error", str(exc))
            self.mode.set("pvp")
            self._update_status()
            return

        _, _, done, info = self.env.step(action)
        self._draw_stones()
        self._update_status()
        if done:
            self._show_result(info)
        elif self.mode.get() == "aivai":
            self.root.after(self._ai_delay, self._ai_move)

    def undo_move(self):
        """Undo one move in PvP, or one human-AI pair in Human vs AI mode."""
        if not self.env.move_history:
            return

        steps = 2 if (self.mode.get() == "pvc" and len(self.env.move_history) >= 2) else 1
        for _ in range(steps):
            if self.env.move_history:
                r, c, _ = self.env.move_history.pop()
                self.env.board[r, c] = 0

        if self.env.move_history:
            _, _, last_player = self.env.move_history[-1]
            self.env.turn = -last_player
            self.env.last_move = (self.env.move_history[-1][0], self.env.move_history[-1][1])
        else:
            self.env.turn = 1
            self.env.last_move = None

        self.env.done = False
        self.env.winner = 0
        self._draw_stones()
        self._update_status()

    def _draw_board(self):
        self.canvas.delete("all")
        board_size = self.env.board_size
        self.canvas.configure(width=self._canvas_size(), height=self._canvas_size())

        for i in range(board_size):
            x = MARGIN + i * CELL_SIZE
            y = MARGIN + i * CELL_SIZE
            self.canvas.create_line(MARGIN, y, MARGIN + (board_size - 1) * CELL_SIZE, y, fill=LINE_COLOR, width=1)
            self.canvas.create_line(x, MARGIN, x, MARGIN + (board_size - 1) * CELL_SIZE, fill=LINE_COLOR, width=1)

        for i in range(board_size):
            x = MARGIN + i * CELL_SIZE
            y = MARGIN + i * CELL_SIZE
            bottom_label_y = MARGIN + (board_size - 1) * CELL_SIZE + 16
            self.canvas.create_text(x, bottom_label_y, text=COL_LABELS[i], font=("Arial", 9, "bold"), fill=LINE_COLOR)
            self.canvas.create_text(MARGIN - 18, y, text=str(i + 1), font=("Arial", 9, "bold"), fill=LINE_COLOR)

        star_points = self._star_points(board_size)
        for r, c in star_points:
            x = MARGIN + c * CELL_SIZE
            y = MARGIN + r * CELL_SIZE
            self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill=LINE_COLOR, outline="")

        self._draw_stones()

    def _draw_stones(self):
        self.canvas.delete("stone")
        for r in range(self.env.board_size):
            for c in range(self.env.board_size):
                if self.env.board[r, c] != 0:
                    self._draw_stone(r, c, int(self.env.board[r, c]))

    def _draw_stone(self, row: int, col: int, player: int):
        x = MARGIN + col * CELL_SIZE
        y = MARGIN + row * CELL_SIZE
        color = BLACK_STONE if player == 1 else WHITE_STONE
        outline = "#444" if player == 1 else "#aaa"

        self.canvas.create_oval(
            x - STONE_RADIUS,
            y - STONE_RADIUS,
            x + STONE_RADIUS,
            y + STONE_RADIUS,
            fill=color,
            outline=outline,
            width=1,
            tags="stone",
        )

        if self.env.last_move == (row, col):
            marker_radius = STONE_RADIUS // 3
            self.canvas.create_oval(
                x - marker_radius,
                y - marker_radius,
                x + marker_radius,
                y + marker_radius,
                fill=HIGHLIGHT_COLOR,
                outline="",
                tags="stone",
            )

    def _update_status(self):
        self._update_move_history()
        self._update_eval_bar()
        self._update_policy_overlay()

        if self.env.done:
            if self.env.winner == 1:
                self.status_var.set("Black wins.")
            elif self.env.winner == -1:
                self.status_var.set("White wins.")
            else:
                self.status_var.set("Draw.")
            return

        turn_str = "Black" if self.env.turn == 1 else "White"
        move_number = len(self.env.move_history) + 1
        ai_note = ""
        if self.mode.get() == "pvc" and self.env.turn == self.ai_color.get():
            ai_note = " AI is thinking..."
        elif self.mode.get() == "aivai":
            ai_note = f" {self._current_ai_engine_name()} is thinking..."
        self.status_var.set(f"Move {move_number}: {turn_str} to move.{ai_note}")

    def _refresh_analysis_overlays(self):
        """Refresh optional model-analysis overlays after toggles change."""
        self._update_eval_bar()
        self._update_policy_overlay()

    def _move_to_notation(self, row: int, col: int) -> str:
        """Convert a row, col pair to board notation such as E5."""
        return f"{COL_LABELS[col]}{row + 1}"

    def _update_move_history(self):
        """Refresh the right-side move list."""
        lines = []
        for idx in range(0, len(self.env.move_history), 2):
            move_no = idx // 2 + 1
            black_move = self.env.move_history[idx]
            black_pos = self._move_to_notation(black_move[0], black_move[1])
            line = f"{move_no:>2}. Black {black_pos:<3}"

            if idx + 1 < len(self.env.move_history):
                white_move = self.env.move_history[idx + 1]
                white_pos = self._move_to_notation(white_move[0], white_move[1])
                line += f"  White {white_pos:<3}"

            lines.append(line)

        text = "\n".join(lines)
        self.history_text.configure(state="normal")
        self.history_text.delete("1.0", "end")
        self.history_text.insert("1.0", text)
        self.history_text.configure(state="disabled")
        self.history_text.see("end")

    def _show_result(self, info: dict):
        self._update_status()
        if info.get("reason") == "win":
            winner_str = "Black" if self.env.winner == 1 else "White"
            message = f"{winner_str} wins.\nTotal moves: {len(self.env.move_history)}\n\nStart a new game?"
            restart = messagebox.askyesno("Game Over", message)
        elif info.get("reason") == "draw":
            restart = messagebox.askyesno("Game Over", "Draw.\n\nStart a new game?")
        else:
            restart = False

        if restart:
            self.new_game()

    def _find_latest_checkpoint(self) -> Path | None:
        """Return the newest trained model checkpoint in the local checkpoints folder."""
        checkpoints = sorted(self.checkpoints, key=lambda path: path.stat().st_mtime, reverse=True)
        return checkpoints[0] if checkpoints else None

    def _list_checkpoints(self) -> list[Path]:
        """Return checkpoint files matching the current board size."""
        if not CHECKPOINT_DIR.exists():
            return []
        board_size = self.board_size.get()
        checkpoints = []
        for path in CHECKPOINT_DIR.glob("*.pt"):
            if self._checkpoint_board_size(path) == board_size:
                checkpoints.append(path)
        return sorted(checkpoints, key=lambda path: path.name.lower())

    def _checkpoint_names(self) -> list[str]:
        """Return names for the checkpoint dropdown."""
        names = [path.name for path in self.checkpoints]
        return names if names else ["No checkpoint"]

    def _on_checkpoint_change(self, selected_name: str):
        """Switch the active AlphaZero checkpoint."""
        self.checkpoint_path = next((path for path in self.checkpoints if path.name == selected_name), None)
        self._agent_cache.clear()
        self._eval_model_cache.clear()
        self._analysis_cache.clear()
        if self.checkpoint_path is None and self.ai_engine.get() == "AlphaZero":
            self.ai_engine.set("Random")
        self.new_game()

    def _on_side_checkpoint_change(self, player: int, selected_name: str):
        """Switch the active checkpoint for one side of AI-vs-AI mode."""
        selected_path = next((path for path in self.checkpoints if path.name == selected_name), None)
        if player == 1:
            self.black_checkpoint_path = selected_path
        else:
            self.white_checkpoint_path = selected_path
        self._agent_cache.clear()
        self._eval_model_cache.clear()
        self._analysis_cache.clear()
        self.new_game()

    def _refresh_checkpoints(self):
        """Refresh checkpoint dropdown after new models are created."""
        self.checkpoints = self._list_checkpoints()
        names = self._checkpoint_names()

        self._rebuild_checkpoint_menu(self.checkpoint_menu, names, self._set_checkpoint_choice)
        self._rebuild_checkpoint_menu(
            self.black_checkpoint_menu,
            names,
            lambda value: self._set_side_checkpoint_choice(1, value),
        )
        self._rebuild_checkpoint_menu(
            self.white_checkpoint_menu,
            names,
            lambda value: self._set_side_checkpoint_choice(-1, value),
        )

        if self.checkpoint_path not in self.checkpoints:
            self.checkpoint_path = self._find_latest_checkpoint()
        if self.black_checkpoint_path not in self.checkpoints:
            self.black_checkpoint_path = self._find_latest_checkpoint()
        if self.white_checkpoint_path not in self.checkpoints:
            self.white_checkpoint_path = self._find_latest_checkpoint()

        selected_name = self.checkpoint_path.name if self.checkpoint_path is not None else "No checkpoint"
        self.checkpoint_choice.set(selected_name)
        black_name = self.black_checkpoint_path.name if self.black_checkpoint_path is not None else "No checkpoint"
        white_name = self.white_checkpoint_path.name if self.white_checkpoint_path is not None else "No checkpoint"
        self.black_checkpoint_choice.set(black_name)
        self.white_checkpoint_choice.set(white_name)
        self._agent_cache.clear()
        self._eval_model_cache.clear()
        self._analysis_cache.clear()
        self.new_game()

    def _on_board_size_change(self):
        """Switch board size and keep only compatible checkpoints available."""
        self.checkpoints = self._list_checkpoints()
        names = self._checkpoint_names()

        self._rebuild_checkpoint_menu(self.checkpoint_menu, names, self._set_checkpoint_choice)
        self._rebuild_checkpoint_menu(
            self.black_checkpoint_menu,
            names,
            lambda value: self._set_side_checkpoint_choice(1, value),
        )
        self._rebuild_checkpoint_menu(
            self.white_checkpoint_menu,
            names,
            lambda value: self._set_side_checkpoint_choice(-1, value),
        )

        self.checkpoint_path = self._find_latest_checkpoint()
        self.black_checkpoint_path = self.checkpoint_path
        self.white_checkpoint_path = self.checkpoint_path

        selected_name = self.checkpoint_path.name if self.checkpoint_path is not None else "No checkpoint"
        self.checkpoint_choice.set(selected_name)
        self.black_checkpoint_choice.set(selected_name)
        self.white_checkpoint_choice.set(selected_name)
        if self.checkpoint_path is None:
            if self.ai_engine.get() == "AlphaZero":
                self.ai_engine.set("Random")
            if self.black_ai_engine.get() == "AlphaZero":
                self.black_ai_engine.set("Heuristic")
            if self.white_ai_engine.get() == "AlphaZero":
                self.white_ai_engine.set("Heuristic")
        self._agent_cache.clear()
        self._eval_model_cache.clear()
        self._analysis_cache.clear()
        self.env = GomokuEnv(board_size=self.board_size.get(), win_count=5)
        self.new_game()

    def _set_checkpoint_choice(self, selected_name: str):
        """Set dropdown text and apply selected checkpoint."""
        self.checkpoint_choice.set(selected_name)
        self._on_checkpoint_change(selected_name)

    def _set_side_checkpoint_choice(self, player: int, selected_name: str):
        """Set side checkpoint dropdown text and apply the selected checkpoint."""
        if player == 1:
            self.black_checkpoint_choice.set(selected_name)
        else:
            self.white_checkpoint_choice.set(selected_name)
        self._on_side_checkpoint_change(player, selected_name)

    def _rebuild_checkpoint_menu(self, option_menu: tk.OptionMenu, names: list[str], callback):
        """Rebuild one checkpoint dropdown menu."""
        menu = option_menu["menu"]
        menu.delete(0, "end")
        for name in names:
            menu.add_command(label=name, command=lambda value=name: callback(value))

    def _current_ai_engine_name(self) -> str:
        """Return the AI label for the side to move."""
        if self.mode.get() == "aivai":
            engine = self.black_ai_engine.get() if self.env.turn == 1 else self.white_ai_engine.get()
            side = "Black" if self.env.turn == 1 else "White"
            return f"{side} {engine}"
        return self.ai_engine.get()

    def _create_ai_agent(self):
        """Create the selected AI agent."""
        if self.mode.get() == "aivai":
            if self.env.turn == 1:
                return self._create_agent(self.black_ai_engine.get(), 1, self.black_checkpoint_path)
            return self._create_agent(self.white_ai_engine.get(), -1, self.white_checkpoint_path)

        return self._create_agent(self.ai_engine.get(), self.ai_color.get(), self.checkpoint_path)

    def _create_agent(self, engine: str, player: int, checkpoint_path: Path | None):
        """Create an agent from an engine name, side, and optional checkpoint."""
        if engine == "Random":
            return RandomAgent(player)

        if engine == "Heuristic":
            from heuristic_agent import HeuristicAgent

            return HeuristicAgent(player=player)

        if engine == "AlphaZero":
            if checkpoint_path is None:
                raise RuntimeError(f"No {self.env.board_size}x{self.env.board_size} .pt checkpoint was found.")
            checkpoint_board_size = self._checkpoint_board_size(checkpoint_path)
            if checkpoint_board_size != self.env.board_size:
                raise RuntimeError(
                    f"{checkpoint_path.name} is {checkpoint_board_size}x{checkpoint_board_size}, "
                    f"but the current board is {self.env.board_size}x{self.env.board_size}."
                )

            from alphazero_agent import AlphaZeroAgent

            simulations = AI_SIMULATIONS[self.ai_strength.get()]
            cache_key = (str(checkpoint_path), self.env.board_size, simulations, self.ai_device)
            if cache_key not in self._agent_cache:
                self._agent_cache[cache_key] = AlphaZeroAgent.from_checkpoint(
                    str(checkpoint_path),
                    simulations=simulations,
                    temperature=0.1,
                    device=self.ai_device,
                )
            return self._agent_cache[cache_key]

        raise RuntimeError(f"Unknown AI engine: {engine}")

    def _checkpoint_board_size(self, path: Path) -> int | None:
        """Read checkpoint board size without loading a GUI agent."""
        if path in self._checkpoint_size_cache:
            return self._checkpoint_size_cache[path]
        try:
            import torch

            checkpoint = torch.load(path, map_location="cpu")
            board_size = int(checkpoint.get("board_size", BOARD_SIZE))
        except Exception:
            board_size = None
        self._checkpoint_size_cache[path] = board_size
        return board_size

    def _update_eval_bar(self):
        """Draw a chess.com-style value bar using the selected AlphaZero checkpoint."""
        self.eval_canvas.delete("all")
        height = self._canvas_size()
        self.eval_canvas.configure(height=height)
        if not self.show_eval_bar.get():
            return

        top = MARGIN
        bottom = MARGIN + (self.env.board_size - 1) * CELL_SIZE
        x1 = (EVAL_BAR_WIDTH - EVAL_BAR_INNER_WIDTH) // 2
        x2 = x1 + EVAL_BAR_INNER_WIDTH

        black_value = self._black_eval_value()
        if black_value is None:
            self.eval_canvas.create_rectangle(x1, top, x2, bottom, fill=EVAL_BAR_NEUTRAL, outline="")
            self.eval_canvas.create_text(
                EVAL_BAR_WIDTH // 2,
                (top + bottom) // 2,
                text="--",
                fill="#F5E4C2",
                font=("Arial", 10, "bold"),
            )
            return

        black_value = max(-1.0, min(1.0, black_value))
        black_share = (black_value + 1.0) / 2.0
        white_share = 1.0 - black_share
        bottom_player = self._bottom_player()

        span = bottom - top
        if bottom_player == 1:
            black_bottom = bottom
            black_top = bottom - span * black_share
            white_top = top
            white_bottom = black_top
        else:
            white_bottom = bottom
            white_top = bottom - span * white_share
            black_top = top
            black_bottom = white_top

        self.eval_canvas.create_rectangle(x1, top, x2, bottom, fill=EVAL_BAR_BLACK, outline="#161411", width=1)
        self.eval_canvas.create_rectangle(x1, white_top, x2, white_bottom, fill=EVAL_BAR_WHITE, outline="")
        self.eval_canvas.create_rectangle(x1, black_top, x2, black_bottom, fill=EVAL_BAR_BLACK, outline="")
        self.eval_canvas.create_line(x1, top + span / 2, x2, top + span / 2, fill="#8C8277", width=1)

        self._draw_eval_number(black_value, x1, x2, black_top, black_bottom, white_top, white_bottom)

    def _black_eval_value(self) -> float | None:
        """Return value from black's perspective, where +1 favors black."""
        if self.env.done:
            if self.env.winner == 1:
                return 1.0
            if self.env.winner == -1:
                return -1.0
            return 0.0

        model = self._eval_model()
        if model is None:
            return None

        try:
            _, current_value = model.predict(self.env, self.ai_device)
        except Exception:
            return None
        return current_value if self.env.turn == 1 else -current_value

    def _eval_model(self):
        """Load the selected checkpoint model for lightweight value inference."""
        checkpoint_path = self._eval_checkpoint_path()
        if checkpoint_path is None:
            return None
        if self._checkpoint_board_size(checkpoint_path) != self.env.board_size:
            return None

        cache_key = (str(checkpoint_path), self.ai_device)
        if cache_key not in self._eval_model_cache:
            from policy_value_net import load_checkpoint

            self._eval_model_cache[cache_key] = load_checkpoint(str(checkpoint_path), device=self.ai_device)
        return self._eval_model_cache[cache_key]

    def _eval_checkpoint_path(self) -> Path | None:
        if self.mode.get() == "aivai":
            if self.env.turn == 1 and self.black_checkpoint_path is not None:
                return self.black_checkpoint_path
            if self.env.turn == -1 and self.white_checkpoint_path is not None:
                return self.white_checkpoint_path
        return self.checkpoint_path or self._find_latest_checkpoint()

    def _update_policy_overlay(self):
        """Show the top moves by one-ply value-head evaluation."""
        self.canvas.delete("policy")
        if not self.show_top_moves.get():
            return
        if self.env.done:
            return

        model = self._eval_model()
        if model is None:
            return

        scored_actions = self._scored_actions_by_eval(model)
        if not scored_actions:
            return

        scored_actions.sort(key=lambda item: item[1], reverse=True)
        top_actions = scored_actions[:TOP_MOVE_COUNT]
        best_score = top_actions[0][1]
        worst_score = top_actions[-1][1]
        score_range = max(best_score - worst_score, 1.0e-6)

        for rank, (action, score) in enumerate(top_actions, start=1):
            strength = 1.0 if len(top_actions) == 1 else (score - worst_score) / score_range
            if rank == 1:
                strength = 1.0
            self._draw_policy_hint(action, strength, score)

    def _scored_actions_by_eval(self, model) -> list[tuple[int, float]]:
        """Evaluate every legal move by applying it and reading the value head."""
        checkpoint_path = self._eval_checkpoint_path()
        cache_key = (
            str(checkpoint_path) if checkpoint_path is not None else "",
            self.ai_device,
            self.env.board_size,
            self.env.turn,
            self.env.last_move,
            self.env.done,
            self.env.winner,
            self.env.board.tobytes(),
        )
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]

        scored_actions = []
        for action in self.env.valid_actions():
            score = self._score_action_by_eval(model, action)
            if score is not None:
                scored_actions.append((action, score))

        self._analysis_cache[cache_key] = scored_actions
        return scored_actions

    def _score_action_by_eval(self, model, action: int) -> float | None:
        """Return action value from the current player's pre-move perspective."""
        player = self.env.turn
        test_env = self.env.copy()
        _, _, done, _ = test_env.step(action)
        if done:
            if test_env.winner == player:
                return 1.0
            if test_env.winner == -player:
                return -1.0
            return 0.0

        try:
            _, next_value = model.predict(test_env, self.ai_device)
        except Exception:
            return None

        return -float(next_value)

    def _draw_policy_hint(self, action: int, strength: float, score: float):
        row, col = self.env.action_to_rc(action)
        x = MARGIN + col * CELL_SIZE
        y = MARGIN + row * CELL_SIZE

        strength = max(0.0, min(1.0, strength))
        radius = 9 + 10 * strength
        fill = self._blend_color(BG_COLOR, TOP_MOVE_COLOR, 0.25 + 0.65 * strength)
        outline = self._blend_color(BG_COLOR, TOP_MOVE_COLOR, 0.50 + 0.45 * strength)

        self.canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill=fill,
            outline=outline,
            width=2,
            tags="policy",
        )
        self.canvas.create_text(
            x,
            y,
            text=f"{score:+.2f}",
            fill="#F8FFF9" if strength > 0.55 else "#123524",
            font=("Arial", 8, "bold"),
            tags="policy",
        )

    def _blend_color(self, background: str, foreground: str, alpha: float) -> str:
        alpha = max(0.0, min(1.0, alpha))

        def parse(hex_color: str) -> tuple[int, int, int]:
            hex_color = hex_color.lstrip("#")
            return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)

        bg = parse(background)
        fg = parse(foreground)
        mixed = tuple(round(bg_channel * (1.0 - alpha) + fg_channel * alpha) for bg_channel, fg_channel in zip(bg, fg))
        return "#{:02x}{:02x}{:02x}".format(*mixed)

    def _draw_eval_number(
        self,
        black_value: float,
        x1: int,
        x2: int,
        black_top: float,
        black_bottom: float,
        white_top: float,
        white_bottom: float,
    ):
        if abs(black_value) < 0.03:
            text = "0.00"
            y = (black_top + black_bottom + white_top + white_bottom) / 4
            fill = "#F5E4C2"
        elif black_value > 0:
            text = f"+{black_value:.2f}"
            y = (black_top + black_bottom) / 2
            fill = "#F4EFE5"
        else:
            text = f"+{-black_value:.2f}"
            y = (white_top + white_bottom) / 2
            fill = "#1A1A1A"

        y = max(MARGIN + 14, min(MARGIN + (self.env.board_size - 1) * CELL_SIZE - 14, y))
        self.eval_canvas.create_text(
            (x1 + x2) / 2,
            y,
            text=text,
            fill=fill,
            angle=90,
            font=("Arial", 9, "bold"),
        )

    def _bottom_player(self) -> int:
        """Return which side should be visually at the bottom of the eval bar."""
        if self.mode.get() == "pvc":
            return -self.ai_color.get()
        return 1

    def _canvas_size(self) -> int:
        return CELL_SIZE * (self.board_size.get() - 1) + 2 * MARGIN

    def _star_points(self, board_size: int) -> list[tuple[int, int]]:
        center = board_size // 2
        if board_size <= 9:
            offset = 2
        else:
            offset = 3
        far = board_size - 1 - offset
        return [(offset, offset), (offset, far), (far, offset), (far, far), (center, center)]


if __name__ == "__main__":
    root = tk.Tk()
    app = GomokuGUI(root)
    root.mainloop()
