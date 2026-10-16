"""PUCT Monte Carlo Tree Search for AlphaZero-lite Gomoku."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from gomoku_env import GomokuEnv
from policy_value_net import PolicyValueNet


@dataclass
class MCTSNode:
    env: GomokuEnv
    parent: "MCTSNode | None" = None
    move: int | None = None
    prior: float = 0.0
    children: dict[int, "MCTSNode"] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def is_expanded(self) -> bool:
        return bool(self.children)


class MCTS:
    """AlphaZero-style MCTS using neural network priors and value estimates."""

    def __init__(
        self,
        model: PolicyValueNet,
        simulations: int = 100,
        c_puct: float = 4.0,
        dirichlet_alpha: float = 0.3,
        root_noise_weight: float = 0.25,
        device: str = "cpu",
    ):
        self.model = model
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.root_noise_weight = root_noise_weight
        self.device = device

    def search(self, env: GomokuEnv, temperature: float = 1.0, add_noise: bool = True) -> tuple[np.ndarray, float]:
        """Run MCTS and return the visit-count policy and root value."""
        root = MCTSNode(env.copy())
        self._expand(root, add_noise=add_noise)

        for _ in range(self.simulations):
            node = root
            while node.is_expanded() and not node.env.done:
                node = self._select_child(node)

            if node.env.done:
                value = self._terminal_value(node.env)
            else:
                value = self._expand(node, add_noise=False)

            self._backup(node, value)

        policy = self._visit_policy(root, temperature)
        return policy, root.q_value

    def select_action(self, env: GomokuEnv, temperature: float = 0.1, add_noise: bool = False) -> int:
        """Choose one move from the MCTS policy."""
        policy, _ = self.search(env, temperature=temperature, add_noise=add_noise)
        return int(np.random.choice(len(policy), p=policy))

    def _expand(self, node: MCTSNode, add_noise: bool) -> float:
        priors, value = self.model.predict(node.env, self.device)
        legal_actions = node.env.valid_actions()

        if add_noise and legal_actions:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_actions))
            for action, eta in zip(legal_actions, noise):
                priors[action] = (1.0 - self.root_noise_weight) * priors[action] + self.root_noise_weight * eta
            priors_sum = priors[legal_actions].sum()
            if priors_sum > 0:
                priors /= priors_sum

        for action in legal_actions:
            child_env = node.env.copy()
            child_env.step(action)
            node.children[action] = MCTSNode(child_env, parent=node, move=action, prior=float(priors[action]))

        return value

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        best_score = -float("inf")
        best_child = None
        parent_visits = max(1, node.visit_count)

        for child in node.children.values():
            q = self._child_q_from_parent_view(node, child)
            u = self.c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child

        if best_child is None:
            raise RuntimeError("MCTS selection reached a node with no children.")
        return best_child

    def _child_q_from_parent_view(self, parent: MCTSNode, child: MCTSNode) -> float:
        if child.env.turn == parent.env.turn:
            return child.q_value
        return -child.q_value

    def _backup(self, node: MCTSNode, value: float):
        """Back up a value that is initially from node.env.turn's perspective."""
        while node is not None:
            node.visit_count += 1
            node.value_sum += value

            parent = node.parent
            if parent is not None and node.env.turn != parent.env.turn:
                value = -value
            node = parent

    def _terminal_value(self, env: GomokuEnv) -> float:
        if env.winner == 0:
            return 0.0
        return 1.0 if env.winner == env.turn else -1.0

    def _visit_policy(self, root: MCTSNode, temperature: float) -> np.ndarray:
        counts = np.zeros(root.env.board_size * root.env.board_size, dtype=np.float32)
        for action, child in root.children.items():
            counts[action] = child.visit_count

        if counts.sum() <= 0:
            legal = root.env.valid_actions()
            counts[legal] = 1.0

        if temperature <= 1.0e-6:
            policy = np.zeros_like(counts)
            policy[int(np.argmax(counts))] = 1.0
            return policy

        adjusted = counts ** (1.0 / temperature)
        adjusted_sum = adjusted.sum()
        if adjusted_sum <= 0:
            return counts / counts.sum()
        return adjusted / adjusted_sum
