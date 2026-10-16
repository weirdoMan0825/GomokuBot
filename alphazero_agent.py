"""Agent wrapper for using AlphaZero-lite MCTS inside the Gomoku project."""

from __future__ import annotations

from gomoku_env import GomokuEnv
from mcts import MCTS
from policy_value_net import PolicyValueNet, load_checkpoint


class AlphaZeroAgent:
    """Choose moves with policy-value MCTS."""

    def __init__(
        self,
        model: PolicyValueNet,
        simulations: int = 100,
        temperature: float = 0.1,
        device: str = "cpu",
    ):
        self.model = model
        self.simulations = simulations
        self.temperature = temperature
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        simulations: int = 100,
        temperature: float = 0.1,
        device: str = "cpu",
    ) -> "AlphaZeroAgent":
        model = load_checkpoint(checkpoint_path, device=device)
        return cls(model, simulations=simulations, temperature=temperature, device=device)

    def act(self, env: GomokuEnv) -> int:
        mcts = MCTS(self.model, simulations=self.simulations, device=self.device)
        return mcts.select_action(env, temperature=self.temperature, add_noise=False)
