"""
PyTorch model definitions.

AR    – linear NN (no hidden layer), same Adam training as NN/LSTM
NN    – one hidden layer, ReLU activation
LSTM  – single-layer LSTM, sequences fed oldest-first, predict from final state
"""
import torch
import torch.nn as nn
import numpy as np


# ── Device detection ──────────────────────────────────────────────────────────

def get_device() -> str:
    """Return the best available device: 'cuda', 'mps', or 'cpu'."""
    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        dev = "mps"
    else:
        dev = "cpu"
    return dev


DEVICE: str = get_device()


# ── Model architectures ───────────────────────────────────────────────────────

class ARModel(nn.Module):
    """AR(p) as linear NN: ŷ = b + W·x, no nonlinearity."""
    def __init__(self, p: int):
        super().__init__()
        self.fc = nn.Linear(p, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


class NNModel(nn.Module):
    """
    Simple NN with one hidden layer.
    ŷ = b + Σ_n w_n · ReLU(b_n + Σ_τ w_nτ · y_{t-τ})
    """
    def __init__(self, p: int, n_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(p, n_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.relu(self.fc1(x))
        return self.fc2(h).squeeze(-1)


class LSTMModel(nn.Module):
    """
    Single-layer LSTM.
    Input:  (batch, seq_len) or (batch, seq_len, 1) – oldest lag first.
    Output: scalar prediction from the final hidden state.
    """
    def __init__(self, n_hidden: int):
        super().__init__()
        self.n_hidden = n_hidden
        self.lstm = nn.LSTM(input_size=1, hidden_size=n_hidden, batch_first=True)
        self.fc = nn.Linear(n_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)          # (batch, seq, 1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class TransformerModel(nn.Module):
    """
    Single-layer Transformer encoder for time-series forecasting.
    Each lag is a token fed oldest-first (same convention as LSTM).
    d_model = n_hidden, dim_feedforward = 2 * n_hidden, n_heads = 2.
    Input: (batch, seq_len) or (batch, seq_len, 1).
    Output: scalar from mean-pooled encoder output.
    n_hidden must be even (divisible by n_heads=2).
    Sinusoidal positional encoding is added after input projection so that
    the encoder can distinguish lag positions (parameter-free, any seq_len).
    Param count ≈ 8 641 at n_hidden=32 (cf. LSTM ≈ 10 651 at n_hidden=50).
    """
    def __init__(self, n_hidden: int):
        super().__init__()
        d_model = n_hidden
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=2,
            dim_feedforward=d_model * 2,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(d_model, 1)

    @staticmethod
    def _sinusoidal_pe(seq_len: int, d_model: int,
                       device: torch.device) -> torch.Tensor:
        """Return (1, seq_len, d_model) sinusoidal positional encoding."""
        pos = torch.arange(seq_len, device=device).unsqueeze(1).float()
        i   = torch.arange(0, d_model, 2, device=device).float()
        div = 10_000 ** (i / d_model)
        pe  = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(pos / div)
        pe[:, 1::2] = torch.cos(pos / div)
        return pe.unsqueeze(0)                 # (1, seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)    # (batch, seq, 1)
        x = self.input_proj(x)     # (batch, seq, d_model)
        x = x + self._sinusoidal_pe(x.size(1), x.size(2), x.device)
        x = self.encoder(x)        # (batch, seq, d_model)
        x = x.mean(dim=1)          # (batch, d_model)  mean pooling
        return self.fc(x).squeeze(-1)


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    X: np.ndarray,
    Y: np.ndarray,
    lr: float = 0.001,
    epochs: int = 500,
    device: str = "cpu",
) -> nn.Module:
    """
    Train model with Adam (β1=0.9, β2=0.999) minimising MSE.
    Full-batch gradient descent (paper trains on full training set each epoch).
    """
    model = model.to(device)
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = criterion(model(Xt), Yt)
        loss.backward()
        opt.step()
    model.eval()
    return model


def predict(model: nn.Module, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        return model(Xt).cpu().numpy()


def model_msfe(model: nn.Module, X: np.ndarray, Y: np.ndarray,
               device: str = "cpu") -> float:
    preds = predict(model, X, device)
    return float(np.mean((preds - Y) ** 2))


def build_model(model_type: str, p: int, n_hidden: int = 50) -> nn.Module:
    if model_type == "ar":
        return ARModel(p)
    elif model_type == "nn":
        return NNModel(p, n_hidden)
    elif model_type == "lstm":
        return LSTMModel(n_hidden)
    elif model_type == "transformer":
        return TransformerModel(n_hidden)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
