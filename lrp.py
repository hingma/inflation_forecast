"""
Layer-wise Relevance Propagation (LRP) for NN and LSTM models.

For NN: standard ε-LRP rule propagated through the two linear layers.
For LSTM: gradient × input (GI) as a tractable approximation of the
          Arras et al. (2017) LSTM-LRP, sufficient to reproduce the
          qualitative patterns shown in the paper's Figures 13–14.

Both return an array of length p (one relevance score per input lag),
where index 0 = lag-1 (most recent) and index p-1 = lag-p (oldest).
"""
import numpy as np
import torch
import torch.nn as nn

from models import NNModel, LSTMModel, TransformerModel


EPS = 1e-9


# ── NN LRP (ε-rule) ───────────────────────────────────────────────────────────

def lrp_nn(model: NNModel, x: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    ε-LRP for the two-layer NN.

    Parameters
    ----------
    model : trained NNModel
    x     : (p,) input vector [lag-1, ..., lag-p]

    Returns
    -------
    R_input : (p,) relevance for each input lag
    """
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, p)

        # Forward pass – save activations
        z1 = model.fc1(xt)                     # (1, n_hidden) pre-activation
        h  = model.relu(z1)                    # (1, n_hidden) post-ReLU
        z2 = model.fc2(h)                      # (1, 1)

        out = z2.squeeze().item()

        # Layer 2 → hidden: R_n ← R_out · (h_n w2_n) / (Σ h_m w2_m + ε)
        w2 = model.fc2.weight.data.squeeze()   # (n_hidden,)
        b2 = model.fc2.bias.data.squeeze()
        h_np  = h.squeeze().numpy()
        w2_np = w2.numpy()

        num2 = h_np * w2_np
        den2 = num2.sum() + b2.item() + eps
        R_hidden = num2 / den2 * out           # (n_hidden,)

        # Layer 1 → input: R_i ← Σ_n R_n · (x_i w1_ni) / (Σ_j x_j w1_nj + ε)
        w1 = model.fc1.weight.data.numpy()     # (n_hidden, p)
        b1 = model.fc1.bias.data.numpy()       # (n_hidden,)
        x_np = x.astype(np.float64)

        R_input = np.zeros(len(x_np))
        for n in range(len(R_hidden)):
            num1 = x_np * w1[n]
            den1 = num1.sum() + b1[n] + eps
            R_input += R_hidden[n] * num1 / den1

    return R_input.astype(np.float32)


# ── LSTM LRP (gradient × input approximation) ────────────────────────────────

def lrp_lstm(model: LSTMModel, x: np.ndarray) -> np.ndarray:
    """
    Gradient × input relevance for LSTM (Arras et al. 2017 approximation).

    Parameters
    ----------
    model : trained LSTMModel
    x     : (p,) input sequence [oldest, ..., most recent]  (lag-p, ..., lag-1)

    Returns
    -------
    R_input : (p,) relevance per time step.
              Index 0 = oldest lag, index p-1 = most recent lag.
              To align with paper's plotting convention (lag-1 on right),
              reverse this array before plotting.
    """
    model.eval()
    xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    xt.requires_grad_(True)

    out = model(xt)
    out.backward()

    grad = xt.grad.detach().squeeze().numpy()   # (p,)
    inp  = x                                    # (p,)
    gi   = grad * inp                           # gradient × input

    return gi.astype(np.float32)


# ── AR LRP (analytical – AR weights are the relevances) ──────────────────────

def lrp_ar(model, x: np.ndarray) -> np.ndarray:
    """
    For a linear AR model the relevance of each input is simply w_i · x_i.
    Normalised so they sum to the prediction.
    """
    with torch.no_grad():
        w = model.fc.weight.data.squeeze().numpy()  # (p,)
        b = model.fc.bias.data.item()
    R = w * x
    return R.astype(np.float32)


# ── Transformer LRP (gradient × input approximation) ─────────────────────────

def lrp_transformer(model: TransformerModel, x: np.ndarray) -> np.ndarray:
    """
    Gradient × input relevance for Transformer.

    Parameters
    ----------
    model : trained TransformerModel
    x     : (p,) input sequence [oldest, ..., most recent]  (lag-p, ..., lag-1)

    Returns
    -------
    R_input : (p,) relevance per time step, oldest first.
    """
    model.eval()
    xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    xt.requires_grad_(True)
    out = model(xt)
    out.backward()
    grad = xt.grad.detach().squeeze().numpy()  # (p,)
    return (grad * x).astype(np.float32)


# ── Dispatch ──────────────────────────────────────────────────────────────────

def compute_lrp(model: nn.Module, x: np.ndarray,
                model_type: str) -> np.ndarray:
    """
    Compute input-level relevances for a single input vector x.

    For AR/NN:          x = [lag-1, ..., lag-p]
    For LSTM/Transformer: x = [lag-p, ..., lag-1]  (oldest first)
    """
    if model_type == "ar":
        return lrp_ar(model, x)
    elif model_type == "nn":
        return lrp_nn(model, x)
    elif model_type == "lstm":
        return lrp_lstm(model, x)
    elif model_type == "transformer":
        return lrp_transformer(model, x)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ── Aggregate LRP over multiple test observations ─────────────────────────────

def aggregate_lrp(model: nn.Module, X: np.ndarray,
                  model_type: str) -> np.ndarray:
    """
    Average absolute relevances over all rows of X.
    Returns array of shape (p,).
    """
    Rs = np.stack([compute_lrp(model, X[i], model_type) for i in range(len(X))])
    return Rs.mean(axis=0)
