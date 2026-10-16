"""
Policy-value neural network for Gomoku.

The network encodes each position with four feature planes:
    0. current player's stones
    1. opponent's stones
    2. opponent's most recent move
    3. current player color, all 1 for black and all 0 for white
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gomoku_env import BOARD_SIZE, GomokuEnv


def board_to_tensor(env: GomokuEnv) -> np.ndarray:
    """Encode the current board as a 4 x board_size x board_size tensor."""
    board = env.board
    tensor = np.zeros((4, env.board_size, env.board_size), dtype=np.float32)
    tensor[0][board == env.turn] = 1.0
    tensor[1][board == -env.turn] = 1.0

    if env.last_move is not None:
        row, col = env.last_move
        tensor[2, row, col] = 1.0

    if env.turn == 1:
        tensor[3, :, :] = 1.0

    return tensor


def legal_action_mask(env: GomokuEnv) -> np.ndarray:
    """Return a flat mask with 1 for legal moves and 0 for occupied cells."""
    mask = np.zeros(env.board_size * env.board_size, dtype=np.float32)
    mask[env.valid_actions()] = 1.0
    return mask


class PolicyValueNet(nn.Module):
    """Small CNN with separate policy and value heads."""

    def __init__(self, board_size: int = BOARD_SIZE, model_blocks: int = 0):
        super().__init__()
        self.board_size = board_size
        self.model_blocks = model_blocks
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.residual_blocks = nn.ModuleList([ResidualBlock(128) for _ in range(model_blocks)])

        self.policy_conv = nn.Conv2d(128, 4, kernel_size=1)
        self.policy_fc = nn.Linear(4 * board_size * board_size, board_size * board_size)

        self.value_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.value_fc1 = nn.Linear(2 * board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        for block in self.residual_blocks:
            x = block(x)

        p = F.relu(self.policy_conv(x))
        p = p.flatten(start_dim=1)
        policy_logits = self.policy_fc(p)

        v = F.relu(self.value_conv(x))
        v = v.flatten(start_dim=1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)).squeeze(-1)

        return policy_logits, value

    @torch.no_grad()
    def predict(self, env: GomokuEnv, device: str | torch.device = "cpu") -> tuple[np.ndarray, float]:
        """Return legal-move probabilities and value for the current player."""
        self.eval()
        state = torch.from_numpy(board_to_tensor(env)).unsqueeze(0).to(device)
        logits, value = self.forward(state)
        logits = logits.squeeze(0).detach().cpu().numpy()

        mask = legal_action_mask(env)
        logits[mask == 0] = -1.0e9
        probs = np.exp(logits - np.max(logits))
        probs *= mask
        prob_sum = probs.sum()
        if prob_sum <= 0:
            probs = mask / max(mask.sum(), 1.0)
        else:
            probs /= prob_sum

        return probs.astype(np.float32), float(value.item())


class ResidualBlock(nn.Module):
    """Two-layer residual block that keeps the trunk width unchanged."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return F.relu(x + residual)


def save_checkpoint(model: PolicyValueNet, path: str):
    """Save model weights."""
    torch.save(
        {
            "board_size": model.board_size,
            "model_blocks": model.model_blocks,
            "model_state": model.state_dict(),
        },
        path,
    )


def load_checkpoint(
    path: str,
    device: str | torch.device = "cpu",
    model_blocks: int | None = None,
    allow_partial: bool = False,
) -> PolicyValueNet:
    """Load model weights."""
    checkpoint = torch.load(path, map_location=device)
    checkpoint_blocks = int(checkpoint.get("model_blocks", 0))
    target_blocks = checkpoint_blocks if model_blocks is None else model_blocks
    model = PolicyValueNet(
        board_size=int(checkpoint.get("board_size", BOARD_SIZE)),
        model_blocks=target_blocks,
    )
    if allow_partial:
        _load_matching_state_dict(model, checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def checkpoint_model_blocks(path: str) -> int:
    """Return residual block count saved in a checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")
    return int(checkpoint.get("model_blocks", 0))


def _load_matching_state_dict(model: PolicyValueNet, checkpoint_state: dict[str, torch.Tensor]):
    """Load same-name, same-shape tensors and leave new layers initialized."""
    current_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in checkpoint_state.items()
        if key in current_state and current_state[key].shape == value.shape
    }
    current_state.update(compatible_state)
    model.load_state_dict(current_state)
