"""Training loop for the Gomoku policy-value engine.

The default arguments stay small for quick smoke tests. Use
``--preset standard`` for a larger self-play training run.
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from device_utils import default_device, describe_device
from gomoku_env import BOARD_SIZE, GomokuEnv
from mcts import MCTS
from policy_value_net import PolicyValueNet, checkpoint_model_blocks, load_checkpoint, save_checkpoint
from progress import progress
from self_play import generate_curriculum_data, generate_self_play_data


def apply_preset(args: argparse.Namespace):
    """Apply named training presets after command-line parsing."""
    if args.preset == "none":
        return

    if args.preset == "standard":
        args.iterations = 10
        args.games = 750
        args.simulations = 500
        args.temperature = 1.0
        args.epochs = 5
        args.batch_size = 256
        args.lr = 1e-3
        args.l2 = 1e-6
        if args.out == "checkpoints/gomoku_alphazero.pt":
            args.out = f"checkpoints/standard_{args.board_size}x{args.board_size}.pt"
        return

    raise ValueError(f"Unknown preset: {args.preset}")


def checkpoint_path_for_iteration(out_path: Path, iteration: int, total_iterations: int) -> Path:
    """Return per-iteration checkpoint path."""
    if total_iterations <= 1:
        return out_path
    return out_path.with_name(f"{out_path.stem}_iter_{iteration:03d}{out_path.suffix}")


def train_on_samples(
    model: PolicyValueNet,
    samples: list[tuple[np.ndarray, np.ndarray, float]],
    epochs: int,
    batch_size: int,
    lr: float,
    l2: float,
    device: str,
):
    random.shuffle(samples)
    states = torch.tensor(np.stack([s for s, _, _ in samples]), dtype=torch.float32)
    policies = torch.tensor(np.stack([p for _, p, _ in samples]), dtype=torch.float32)
    values = torch.tensor([v for _, _, v in samples], dtype=torch.float32)

    dataset = TensorDataset(states, policies, values)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    model.train()
    for epoch in progress(range(epochs), desc="epochs", unit="epoch"):
        total_loss = 0.0
        batch_iter = progress(
            loader,
            desc=f"train epoch {epoch + 1}/{epochs}",
            unit="batch",
            leave=False,
        )
        for batch_states, target_policies, target_values in batch_iter:
            batch_states = batch_states.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device)

            logits, pred_values = model(batch_states)
            value_loss = F.mse_loss(pred_values, target_values)
            policy_loss = -(target_policies * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            loss = value_loss + policy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_states)
            if hasattr(batch_iter, "set_postfix"):
                batch_iter.set_postfix(loss=float(loss.item()))

        print(f"epoch {epoch + 1}/{epochs}: loss={total_loss / len(dataset):.4f}")


class InMemoryMCTSAgent:
    """MCTS agent that uses an already-loaded model."""

    def __init__(
        self,
        model: PolicyValueNet,
        simulations: int,
        temperature: float,
        device: str | torch.device,
    ):
        self.model = model
        self.simulations = simulations
        self.temperature = temperature
        self.device = device

    def act(self, env: GomokuEnv) -> int:
        mcts = MCTS(self.model, simulations=self.simulations, device=self.device)
        return mcts.select_action(env, temperature=self.temperature, add_noise=False)


def play_arena_game(
    black_model: PolicyValueNet,
    white_model: PolicyValueNet,
    board_size: int,
    simulations: int,
    temperature: float,
    device: str | torch.device,
) -> tuple[int, int]:
    env = GomokuEnv(board_size=board_size, win_count=5)
    env.reset()
    agents = {
        1: InMemoryMCTSAgent(black_model, simulations, temperature, device),
        -1: InMemoryMCTSAgent(white_model, simulations, temperature, device),
    }

    while not env.done:
        action = int(agents[env.turn].act(env.copy()))
        env.step(action)
    return env.winner, len(env.move_history)


def run_arena_gating(
    challenger: PolicyValueNet,
    best: PolicyValueNet,
    board_size: int,
    games_per_color: int,
    simulations: int,
    temperature: float,
    device: str | torch.device,
) -> dict[str, float]:
    """Evaluate challenger against best as both black and white."""
    wins = 0
    losses = 0
    draws = 0
    total_moves = 0

    for _ in progress(range(games_per_color), desc="arena challenger black", unit="game"):
        winner, moves = play_arena_game(challenger, best, board_size, simulations, temperature, device)
        total_moves += moves
        if winner == 1:
            wins += 1
        elif winner == -1:
            losses += 1
        else:
            draws += 1

    for _ in progress(range(games_per_color), desc="arena challenger white", unit="game"):
        winner, moves = play_arena_game(best, challenger, board_size, simulations, temperature, device)
        total_moves += moves
        if winner == -1:
            wins += 1
        elif winner == 1:
            losses += 1
        else:
            draws += 1

    total = wins + losses + draws
    score = (wins + 0.5 * draws) / total if total else 0.0
    return {
        "wins": float(wins),
        "losses": float(losses),
        "draws": float(draws),
        "score": score,
        "avg_moves": total_moves / total if total else 0.0,
    }


def load_opponent_checkpoints(paths: list[str], board_size: int, device: torch.device) -> list[PolicyValueNet]:
    opponents = []
    for checkpoint in paths:
        model = load_checkpoint(checkpoint, device=device)
        if model.board_size != board_size:
            raise ValueError(
                f"Opponent checkpoint {checkpoint} is {model.board_size}x{model.board_size}, "
                f"but --board-size is {board_size}."
            )
        opponents.append(model)
    return opponents


def finalize_curriculum_args(args: argparse.Namespace):
    if args.curriculum == "mixed":
        if args.tactical_ratio is None:
            args.tactical_ratio = 0.20
        if args.heuristic_ratio is None:
            args.heuristic_ratio = 0.0
        if args.checkpoint_ratio is None:
            args.checkpoint_ratio = 0.10
        if args.previous_opponent is None:
            args.previous_opponent = True
    else:
        args.tactical_ratio = 0.0 if args.tactical_ratio is None else args.tactical_ratio
        args.heuristic_ratio = 0.0 if args.heuristic_ratio is None else args.heuristic_ratio
        args.checkpoint_ratio = 0.0 if args.checkpoint_ratio is None else args.checkpoint_ratio
        args.previous_opponent = False if args.previous_opponent is None else args.previous_opponent

    if not 0.0 <= args.tactical_ratio <= 1.0:
        raise ValueError("--tactical-ratio must be in [0, 1].")
    if not 0.0 <= args.heuristic_ratio <= 1.0:
        raise ValueError("--heuristic-ratio must be in [0, 1].")
    if not 0.0 <= args.checkpoint_ratio <= 1.0:
        raise ValueError("--checkpoint-ratio must be in [0, 1].")
    if args.heuristic_ratio + args.checkpoint_ratio > 1.0:
        raise ValueError("--heuristic-ratio + --checkpoint-ratio must be <= 1.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=["none", "standard"], default="none")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--opponent-checkpoint", action="append", default=[])
    parser.add_argument("--out", type=str, default="checkpoints/gomoku_alphazero.pt")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--board-size", type=int, default=BOARD_SIZE)
    parser.add_argument("--model-blocks", type=int, default=None, help="Number of 128-channel residual blocks after the base CNN.")
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--simulations", type=int, default=25)
    parser.add_argument("--opponent-simulations", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default=default_device())
    parser.add_argument("--curriculum", choices=["self", "mixed"], default="self")
    parser.add_argument("--tactical-ratio", type=float, default=None)
    parser.add_argument("--heuristic-ratio", type=float, default=None)
    parser.add_argument("--checkpoint-ratio", type=float, default=None)
    parser.add_argument("--previous-opponent", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--arena-games-per-color", type=int, default=0)
    parser.add_argument("--arena-simulations", type=int, default=0)
    parser.add_argument("--arena-temperature", type=float, default=0.0)
    parser.add_argument("--arena-threshold", type=float, default=0.55)
    parser.add_argument("--save-rejected", action="store_true")
    args = parser.parse_args()
    apply_preset(args)
    finalize_curriculum_args(args)
    if args.model_blocks is not None and args.model_blocks < 0:
        raise ValueError("--model-blocks must be >= 0.")

    device = torch.device(args.device)
    if args.checkpoint:
        checkpoint_blocks = checkpoint_model_blocks(args.checkpoint)
        target_blocks = checkpoint_blocks if args.model_blocks is None else args.model_blocks
        allow_partial = target_blocks != checkpoint_blocks
        model = load_checkpoint(
            args.checkpoint,
            device=device,
            model_blocks=target_blocks,
            allow_partial=allow_partial,
        )
        if model.board_size != args.board_size:
            raise ValueError(
                f"Checkpoint board size is {model.board_size}, but --board-size is {args.board_size}. "
                f"Use --board-size {model.board_size} when continuing this checkpoint."
            )
        if allow_partial:
            print(
                "partial checkpoint transfer: "
                f"loaded compatible weights from {checkpoint_blocks} residual blocks "
                f"into {target_blocks} residual blocks"
            )
    else:
        target_blocks = 0 if args.model_blocks is None else args.model_blocks
        model = PolicyValueNet(board_size=args.board_size, model_blocks=target_blocks).to(device)
    model.eval()
    explicit_opponents = load_opponent_checkpoints(args.opponent_checkpoint, args.board_size, device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("training configuration:")
    print(f"  preset: {args.preset}")
    print(f"  board size: {args.board_size}x{args.board_size}")
    print(f"  residual blocks: {model.model_blocks}")
    print(f"  iterations: {args.iterations}")
    print(f"  games per iteration: {args.games}")
    print(f"  MCTS simulations per move: {args.simulations}")
    print(f"  epochs per iteration: {args.epochs}")
    print(f"  batch size: {args.batch_size}")
    print(f"  lr: {args.lr}")
    print(f"  l2: {args.l2}")
    print(f"  device: {describe_device(device)}")
    print(f"  output: {out_path}")
    print(f"  curriculum: {args.curriculum}")
    print(f"  tactical ratio: {args.tactical_ratio}")
    print(f"  heuristic ratio: {args.heuristic_ratio}")
    print(f"  checkpoint ratio: {args.checkpoint_ratio}")
    print(f"  previous best as opponent: {args.previous_opponent}")
    print(f"  explicit opponent checkpoints: {len(explicit_opponents)}")
    if args.arena_games_per_color > 0:
        arena_simulations = args.arena_simulations or args.simulations
        print("  arena gating: enabled")
        print(f"  arena games per color: {args.arena_games_per_color}")
        print(f"  arena simulations: {arena_simulations}")
        print(f"  arena threshold: {args.arena_threshold}")
    else:
        print("  arena gating: disabled")

    previous_best_opponent: PolicyValueNet | None = None

    for iteration in progress(range(1, args.iterations + 1), desc="iterations", unit="iter"):
        print(f"\n=== iteration {iteration}/{args.iterations} ===")
        old_best = copy.deepcopy(model).to(device)
        old_best.eval()
        checkpoint_opponents = explicit_opponents[:]
        if args.previous_opponent and previous_best_opponent is not None:
            checkpoint_opponents.append(previous_best_opponent)

        if args.curriculum == "mixed":
            samples = generate_curriculum_data(
                model,
                games=args.games,
                board_size=args.board_size,
                simulations=args.simulations,
                temperature=args.temperature,
                augment=True,
                device=device,
                tactical_ratio=args.tactical_ratio,
                heuristic_ratio=args.heuristic_ratio,
                checkpoint_ratio=args.checkpoint_ratio,
                checkpoint_opponents=checkpoint_opponents,
                opponent_simulations=args.opponent_simulations or args.simulations,
            )
        else:
            samples = generate_self_play_data(
                model,
                games=args.games,
                board_size=args.board_size,
                simulations=args.simulations,
                temperature=args.temperature,
                augment=True,
                device=device,
            )

        challenger = copy.deepcopy(model).to(device)
        train_on_samples(challenger, samples, args.epochs, args.batch_size, args.lr, args.l2, device)
        challenger.eval()

        accepted = True
        if args.arena_games_per_color > 0:
            arena_simulations = args.arena_simulations or args.simulations
            result = run_arena_gating(
                challenger,
                model,
                board_size=args.board_size,
                games_per_color=args.arena_games_per_color,
                simulations=arena_simulations,
                temperature=args.arena_temperature,
                device=device,
            )
            accepted = result["score"] >= args.arena_threshold
            print(
                "arena result: "
                f"W/L/D={int(result['wins'])}/{int(result['losses'])}/{int(result['draws'])}, "
                f"score={result['score']:.3f}, avg_moves={result['avg_moves']:.1f}, "
                f"threshold={args.arena_threshold:.3f}"
            )
            print("arena decision:", "accepted" if accepted else "rejected")

        if accepted:
            previous_best_opponent = old_best
            model = challenger
        elif args.save_rejected:
            rejected_path = checkpoint_path_for_iteration(out_path, iteration, args.iterations)
            rejected_path = rejected_path.with_name(f"{rejected_path.stem}_rejected{rejected_path.suffix}")
            save_checkpoint(challenger, str(rejected_path))
            print(f"saved rejected challenger: {rejected_path}")

        iter_path = checkpoint_path_for_iteration(out_path, iteration, args.iterations)
        save_checkpoint(model, str(iter_path))
        print(f"saved checkpoint: {iter_path}")

    if args.iterations > 1:
        save_checkpoint(model, str(out_path))
        print(f"saved final checkpoint: {out_path}")


if __name__ == "__main__":
    main()
