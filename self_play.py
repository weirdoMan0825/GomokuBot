"""Self-play data generation for AlphaZero-lite Gomoku."""

from __future__ import annotations

import numpy as np

from gomoku_env import BOARD_SIZE, GomokuEnv
from heuristic_agent import HeuristicAgent
from mcts import MCTS
from policy_value_net import PolicyValueNet, board_to_tensor
from progress import progress
from tactical_positions import generate_tactical_position


def augment_sample(state: np.ndarray, policy: np.ndarray, value: float, board_size: int = BOARD_SIZE):
    """Generate eight rotation/reflection variants for one training sample."""
    policy_board = policy.reshape(board_size, board_size)
    samples = []
    for k in range(4):
        state_rot = np.rot90(state, k, axes=(1, 2))
        policy_rot = np.rot90(policy_board, k)
        samples.append((state_rot.copy(), policy_rot.reshape(-1).copy(), value))

        state_flip = np.flip(state_rot, axis=2)
        policy_flip = np.flip(policy_rot, axis=1)
        samples.append((state_flip.copy(), policy_flip.reshape(-1).copy(), value))
    return samples


def play_self_game(
    model: PolicyValueNet,
    board_size: int = BOARD_SIZE,
    simulations: int = 50,
    temperature: float = 1.0,
    augment: bool = True,
    device: str = "cpu",
    initial_env: GomokuEnv | None = None,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Play one self-play game and return (state, mcts_policy, outcome) samples."""
    env = initial_env.copy() if initial_env is not None else GomokuEnv(board_size=board_size, win_count=5)
    if initial_env is None:
        env.reset()
    mcts = MCTS(model, simulations=simulations, device=device)

    trajectory = []
    while not env.done:
        state = board_to_tensor(env)
        player = env.turn
        policy, _ = mcts.search(env, temperature=temperature, add_noise=True)
        action = int(np.random.choice(len(policy), p=policy))
        trajectory.append((state, policy, player))
        env.step(action)

    samples = []
    for state, policy, player in trajectory:
        if env.winner == 0:
            value = 0.0
        else:
            value = 1.0 if player == env.winner else -1.0

        if augment:
            samples.extend(augment_sample(state, policy, value, board_size))
        else:
            samples.append((state, policy, value))

    return samples


def play_model_vs_heuristic_game(
    model: PolicyValueNet,
    board_size: int = BOARD_SIZE,
    simulations: int = 50,
    temperature: float = 1.0,
    augment: bool = True,
    device: str = "cpu",
    initial_env: GomokuEnv | None = None,
    model_player: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Play current model against the heuristic benchmark.

    Only the current model's turns are kept as training samples. Heuristic
    moves are environment responses, not policy targets.
    """
    env = initial_env.copy() if initial_env is not None else GomokuEnv(board_size=board_size, win_count=5)
    if initial_env is None:
        env.reset()
    if model_player not in (1, -1):
        model_player = env.turn if initial_env is not None else int(np.random.choice([1, -1]))

    model_mcts = MCTS(model, simulations=simulations, device=device)
    heuristic = HeuristicAgent()
    trajectory = []

    while not env.done:
        if env.turn == model_player:
            state = board_to_tensor(env)
            player = env.turn
            policy, _ = model_mcts.search(env, temperature=temperature, add_noise=True)
            action = int(np.random.choice(len(policy), p=policy))
            trajectory.append((state, policy, player))
        else:
            action = int(heuristic.act(env.copy()))
        env.step(action)

    return _samples_from_trajectory(trajectory, env.winner, board_size, augment)


def play_model_vs_checkpoint_game(
    model: PolicyValueNet,
    opponent_model: PolicyValueNet,
    board_size: int = BOARD_SIZE,
    simulations: int = 50,
    opponent_simulations: int | None = None,
    temperature: float = 1.0,
    opponent_temperature: float = 0.1,
    augment: bool = True,
    device: str = "cpu",
    initial_env: GomokuEnv | None = None,
    model_player: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Play current model against a frozen checkpoint opponent.

    Only the current model's turns are kept as training samples.
    """
    env = initial_env.copy() if initial_env is not None else GomokuEnv(board_size=board_size, win_count=5)
    if initial_env is None:
        env.reset()
    if model_player not in (1, -1):
        model_player = env.turn if initial_env is not None else int(np.random.choice([1, -1]))
    opponent_simulations = opponent_simulations or simulations

    model_mcts = MCTS(model, simulations=simulations, device=device)
    opponent_mcts = MCTS(opponent_model, simulations=opponent_simulations, device=device)
    trajectory = []

    while not env.done:
        if env.turn == model_player:
            state = board_to_tensor(env)
            player = env.turn
            policy, _ = model_mcts.search(env, temperature=temperature, add_noise=True)
            action = int(np.random.choice(len(policy), p=policy))
            trajectory.append((state, policy, player))
        else:
            action = opponent_mcts.select_action(env, temperature=opponent_temperature, add_noise=False)
        env.step(action)

    return _samples_from_trajectory(trajectory, env.winner, board_size, augment)


def _samples_from_trajectory(
    trajectory: list[tuple[np.ndarray, np.ndarray, int]],
    winner: int,
    board_size: int,
    augment: bool,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    samples = []
    for state, policy, player in trajectory:
        if winner == 0:
            value = 0.0
        else:
            value = 1.0 if player == winner else -1.0

        if augment:
            samples.extend(augment_sample(state, policy, value, board_size))
        else:
            samples.append((state, policy, value))
    return samples


def generate_self_play_data(
    model: PolicyValueNet,
    games: int,
    board_size: int,
    simulations: int,
    temperature: float,
    augment: bool = True,
    device: str = "cpu",
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Generate training samples from multiple self-play games."""
    data = []
    game_iter = progress(range(games), desc="self-play", unit="game")
    for game_idx in game_iter:
        game_data = play_self_game(
            model,
            board_size=board_size,
            simulations=simulations,
            temperature=temperature,
            augment=augment,
            device=device,
        )
        data.extend(game_data)
        if hasattr(game_iter, "set_postfix"):
            game_iter.set_postfix(samples=len(data), last_game_samples=len(game_data))
        else:
            print(f"self-play game {game_idx + 1}/{games}: {len(game_data)} samples")
    return data


def generate_curriculum_data(
    model: PolicyValueNet,
    games: int,
    board_size: int,
    simulations: int,
    temperature: float,
    augment: bool = True,
    device: str = "cpu",
    tactical_ratio: float = 0.0,
    heuristic_ratio: float = 0.0,
    checkpoint_ratio: float = 0.0,
    checkpoint_opponents: list[PolicyValueNet] | None = None,
    opponent_simulations: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Generate mixed curriculum data.

    Game type is sampled per game. Ratios are interpreted as probabilities;
    remaining probability is pure model-vs-model self-play.
    """
    checkpoint_opponents = checkpoint_opponents or []
    checkpoint_ratio = checkpoint_ratio if checkpoint_opponents else 0.0
    total_opponent_ratio = heuristic_ratio + checkpoint_ratio
    if total_opponent_ratio > 1.0:
        raise ValueError("--heuristic-ratio + --checkpoint-ratio must be <= 1.0")

    data = []
    counts = {"self": 0, "heuristic": 0, "checkpoint": 0, "tactical": 0}
    game_iter = progress(range(games), desc="curriculum", unit="game")
    for game_idx in game_iter:
        initial_env = None
        if tactical_ratio > 0.0 and np.random.random() < tactical_ratio:
            initial_env = generate_tactical_position(board_size=board_size).env
            counts["tactical"] += 1

        roll = np.random.random()
        if roll < heuristic_ratio:
            game_data = play_model_vs_heuristic_game(
                model,
                board_size=board_size,
                simulations=simulations,
                temperature=temperature,
                augment=augment,
                device=device,
                initial_env=initial_env,
            )
            counts["heuristic"] += 1
        elif roll < heuristic_ratio + checkpoint_ratio:
            opponent = checkpoint_opponents[int(np.random.randint(len(checkpoint_opponents)))]
            game_data = play_model_vs_checkpoint_game(
                model,
                opponent,
                board_size=board_size,
                simulations=simulations,
                opponent_simulations=opponent_simulations,
                temperature=temperature,
                augment=augment,
                device=device,
                initial_env=initial_env,
            )
            counts["checkpoint"] += 1
        else:
            game_data = play_self_game(
                model,
                board_size=board_size,
                simulations=simulations,
                temperature=temperature,
                augment=augment,
                device=device,
                initial_env=initial_env,
            )
            counts["self"] += 1

        data.extend(game_data)
        if hasattr(game_iter, "set_postfix"):
            game_iter.set_postfix(
                samples=len(data),
                self_games=counts["self"],
                heuristic=counts["heuristic"],
                checkpoint=counts["checkpoint"],
                tactical=counts["tactical"],
            )
        else:
            print(f"curriculum game {game_idx + 1}/{games}: {len(game_data)} samples")
    return data
