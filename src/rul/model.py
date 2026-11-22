import numpy as np
import torch
from torch import nn


class RULNet(nn.Module):
    """LSTM over a window of sensor readings, then a small head that outputs RUL / cap."""

    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)  # state after the last cycle


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def predict(model: RULNet, X: np.ndarray, cap: int, device: str | None = None, batch: int = 4096) -> np.ndarray:
    """Return RUL in cycles, clipped to [0, cap]."""
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.as_tensor(X[i : i + batch]).to(device)
            out.append((model(xb) * cap).clamp(0, cap).cpu().numpy())
    return np.concatenate(out)
