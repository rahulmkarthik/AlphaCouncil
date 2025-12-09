# ============================================================
# volsense_core/models/global_vol_forecaster.py  (v3)
# GlobalVolForecaster with attention, per-horizon heads,
# feature-dropout, variational (recurrent) dropout, EMA weights,
# cosine restarts, early stopping — still backward-compatible.
# ============================================================

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from volsense_core.utils.scalers import TorchStandardScaler as StandardScaler
from dataclasses import dataclass
from torch.utils.data import DataLoader
from typing import Dict
import copy


# ============================================================
# 🧩 Config Dataclass
# ============================================================
@dataclass
class TrainConfig:
    """
    Training configuration for GlobalVolForecaster.

    :param window: Rolling lookback window length (timesteps).
    :type window: int
    :param horizons: Forecast horizons; either an int (max horizon) or explicit list (e.g., [1,5,10]).
    :type horizons: int or list[int]
    :param stride: Step size between training samples (larger reduces overlap).
    :type stride: int
    :param val_start: Validation window start date (inclusive).
    :type val_start: str
    :param val_end: Optional validation window end date (exclusive). If None, uses all data after val_start.
    :type val_end: str, optional
    :param val_mode: Validation mode; "causal" (train strictly before val_start) or "holdout_slice".
    :type val_mode: str
    :param embargo_days: Days excluded on both sides of validation slice in holdout mode.
    :type embargo_days: int
    :param target_col: Target column name (typically realized_vol_log).
    :type target_col: str
    :param extra_features: Additional feature column names to include beyond "return".
    :type extra_features: list[str], optional
    :param epochs: Number of training epochs.
    :type epochs: int
    :param lr: Learning rate for optimizer.
    :type lr: float
    :param batch_size: Mini-batch size.
    :type batch_size: int
    :param device: Compute device ("cpu" or "cuda").
    :type device: str
    :param dropout: Inter-layer dropout probability in the LSTM and heads.
    :type dropout: float
    :param use_layernorm: Whether to apply LayerNorm to the backbone output.
    :type use_layernorm: bool
    :param separate_heads: If True, use independent heads per horizon; else a single multi-output head.
    :type separate_heads: bool
    :param attention: If True, apply a MultiheadAttention layer over LSTM outputs.
    :type attention: bool
    :param residual_head: If True, add a residual linear head to improve gradient flow.
    :type residual_head: bool
    :param feat_dropout_p: Channel-wise (feature) dropout probability at input.
    :type feat_dropout_p: float
    :param variational_dropout_p: Variational (time-consistent) input dropout probability.
    :type variational_dropout_p: float
    :param loss_horizon_weights: Optional per-horizon loss weights.
    :type loss_horizon_weights: list[float], optional
    :param dynamic_window_jitter: Random jitter (+/-) applied to window size during training.
    :type dynamic_window_jitter: int
    :param grad_clip: Gradient norm clip value; set None/0 to disable.
    :type grad_clip: float
    :param cosine_schedule: Use cosine LR schedule.
    :type cosine_schedule: bool
    :param cosine_restarts: Use cosine schedule with warm restarts (SGDR).
    :type cosine_restarts: bool
    :param oversample_high_vol: Deprecated knob; kept for compatibility.
    :type oversample_high_vol: bool
    :param use_ema: Maintain an exponential moving average (EMA) of model weights.
    :type use_ema: bool
    :param ema_decay: EMA decay factor in (0,1).
    :type ema_decay: float
    :param early_stop: Enable early stopping on validation loss.
    :type early_stop: bool
    :param patience: Epochs to wait without improvement before stopping.
    :type patience: int
    :param min_delta: Minimum improvement in validation loss to reset patience.
    :type min_delta: float
    :param num_workers: DataLoader workers.
    :type num_workers: int
    :param pin_memory: DataLoader pin_memory flag.
    :type pin_memory: bool
    """

    window: int = 30
    horizons: int | list = 1
    stride: int = 2  # 👈 default stride=3 for decorrelation
    val_start: str = "2025-01-01"
    val_end: str | None = None  # optional: end of validation window (exclusive)
    val_mode: str = "causal"  # "causal" or "holdout_slice"
    embargo_days: int = 0  # drop these days on either side of val window from train
    target_col: str = "realized_vol_log"
    extra_features: list | None = None

    # training
    epochs: int = 20
    lr: float = 3e-4
    batch_size: int = 256
    device: str = "cpu"

    # regularization / arch
    dropout: float = 0.3  # LSTM inter-layer dropout
    use_layernorm: bool = True
    separate_heads: bool = True  # per-horizon decoders
    attention: bool = True
    residual_head: bool = True

    # v3 knobs
    feat_dropout_p: float = 0.1  # randomly drop feature channels (input)
    variational_dropout_p: float = 0.1  # recurrent-style mask on inputs per batch
    loss_horizon_weights: list | None = None  # e.g. [0.6, 0.25, 0.15]

    # v4 knobs
    dynamic_window_jitter: int = 0
    grad_clip: float = 1.0

    cosine_schedule: bool = True
    cosine_restarts: bool = True  # use SGDR
    oversample_high_vol: bool = False  # kept for compat; not used by default

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.995

    # early stop
    early_stop: bool = True
    patience: int = 7
    min_delta: float = 0.0

    # dataloader
    num_workers: int = 0
    pin_memory: bool = False


# ============================================================
# 🧠 Utility modules (feature dropout / variational dropout)
# ============================================================
class FeatureDropout(nn.Module):
    """Channel-wise (feature) dropout: same mask for all time-steps."""

    def __init__(self, p: float):
        """
        Initialize feature dropout.

        :param p: Drop probability for feature channels (0 <= p < 1).
        :type p: float
        """
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel-wise dropout to inputs.

        :param x: Input tensor of shape [B, W, F].
        :type x: torch.Tensor
        :return: Tensor with dropped feature channels, scaled by 1/(1-p) during training.
        :rtype: torch.Tensor
        """
        # x: [B, W, F]
        if not self.training or self.p <= 0:
            return x
        B, W, F = x.shape
        mask = torch.empty(B, 1, F, device=x.device, dtype=x.dtype).bernoulli_(
            1 - self.p
        )
        return x * mask / (1 - self.p)


class VariationalInputDropout(nn.Module):
    """Variational (recurrent) dropout on inputs — same mask across time."""

    def __init__(self, p: float):
        """
        Initialize variational input dropout.

        :param p: Drop probability for feature channels (0 <= p < 1).
        :type p: float
        """
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a time-consistent dropout mask across the input sequence.

        :param x: Input tensor of shape [B, W, F].
        :type x: torch.Tensor
        :return: Tensor with per-batch feature mask applied across all time-steps.
        :rtype: torch.Tensor
        """
        if not self.training or self.p <= 0:
            return x
        B, W, F = x.shape
        mask = torch.empty(B, 1, F, device=x.device, dtype=x.dtype).bernoulli_(
            1 - self.p
        )
        return x * mask


# ============================================================
# 🧠 GlobalVolForecaster Model
# ============================================================
class GlobalVolForecaster(nn.Module):
    """
    Global sequence model for multi-ticker, multi-horizon volatility forecasting.

    Combines LSTM backbone, optional temporal self-attention, per-horizon heads,
    and optional residual head. Ticker identity is injected via learned embeddings.
    """

    def __init__(
        self,
        n_tickers: int,
        window: int,
        n_horizons: int | list,
        emb_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        attention: bool = True,
        residual_head: bool = True,
        input_size: int | None = None,
        use_layernorm: bool = True,
        separate_heads: bool = True,
        feat_dropout_p: float = 0.1,
        variational_dropout_p: float = 0.1,
    ):
        """
        Initialize GlobalVolForecaster.

        :param n_tickers: Number of distinct tickers (size of embedding vocabulary).
        :type n_tickers: int
        :param window: Fixed input sequence length expected by the model.
        :type window: int
        :param n_horizons: Number of horizons (int) or explicit list; controls output size.
        :type n_horizons: int or list[int]
        :param emb_dim: Ticker embedding dimensionality.
        :type emb_dim: int
        :param hidden_dim: LSTM hidden size.
        :type hidden_dim: int
        :param num_layers: Number of stacked LSTM layers.
        :type num_layers: int
        :param dropout: Dropout rate used in LSTM and MLP heads.
        :type dropout: float
        :param attention: If True, apply MultiheadAttention over LSTM outputs.
        :type attention: bool
        :param residual_head: Add residual linear projection from backbone to outputs.
        :type residual_head: bool
        :param input_size: Number of input features per timestep.
        :type input_size: int, optional
        :param use_layernorm: Apply LayerNorm to backbone output.
        :type use_layernorm: bool
        :param separate_heads: Use separate 1-unit heads per horizon when True; else a multi-output head.
        :type separate_heads: bool
        :param feat_dropout_p: Channel-wise feature dropout probability at input.
        :type feat_dropout_p: float
        :param variational_dropout_p: Variational input dropout probability at input.
        :type variational_dropout_p: float
        """
        super().__init__()
        self.n_tickers = n_tickers
        self.window = window
        self.input_size = int(input_size) if input_size is not None else 1
        self.attention_enabled = attention
        self.residual_head = residual_head
        self.use_layernorm = use_layernorm
        self.separate_heads = separate_heads

        # Input regularizers
        self.feat_do = FeatureDropout(feat_dropout_p)
        self.var_do = VariationalInputDropout(variational_dropout_p)

        # Ticker embedding
        self.tok = nn.Embedding(n_tickers, emb_dim)

        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=self.input_size + emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        # Optional temporal attention
        if self.attention_enabled:
            self.attn = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=4, dropout=dropout, batch_first=True
            )

        self.ln = nn.LayerNorm(hidden_dim) if self.use_layernorm else nn.Identity()

        # Horizons
        self.n_horizons = n_horizons if isinstance(n_horizons, int) else len(n_horizons)

        # Decoders
        if self.separate_heads and self.n_horizons > 1:
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim // 2, 1),
                    )
                    for _ in range(self.n_horizons)
                ]
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, self.n_horizons),
            )

        if self.residual_head:
            self.residual = nn.Linear(hidden_dim, self.n_horizons)

    def forward(self, tkr_id: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        :param tkr_id: Integer ticker IDs of shape [B].
        :type tkr_id: torch.Tensor
        :param X: Input features of shape [B, W, F] (or [B, F], which will be unsqueezed).
        :type X: torch.Tensor
        :return: Horizon outputs of shape [B, H] (or [B] when H==1).
        :rtype: torch.Tensor
        """
        # X: [B, W, F] or [B, F]
        if X.ndim == 2:
            X = X.unsqueeze(1)

        B, W, F = X.shape
        X = self.feat_do(X)
        X = self.var_do(X)

        emb = self.tok(tkr_id).unsqueeze(1).expand(B, W, -1)
        X = torch.cat([X, emb], dim=-1)  # [B, W, F+E]

        out, _ = self.lstm(X)  # [B, W, H]
        if self.attention_enabled:
            out, _ = self.attn(out, out, out)
        out = self.ln(out)

        last = out[:, -1, :]  # [B, H]

        if self.separate_heads and self.n_horizons > 1:
            ys = [head(last) for head in self.heads]  # list of [B,1]
            yhat = torch.cat(ys, dim=-1)  # [B, H]
        else:
            yhat = self.head(last)  # [B, H]

        if self.residual_head:
            yhat = yhat + self.residual(last)

        return yhat.squeeze(-1) if self.n_horizons == 1 else yhat


# ============================================================
# 🧮 Dataset Builder (per-ticker feature scaling; no target scaling)
# ============================================================
def build_global_splits(df: pd.DataFrame, cfg: TrainConfig):
    """
    Build per-ticker scaled datasets and temporal train/validation splits.

    Performs per-ticker StandardScaler fitting on features, maps tickers to IDs,
    and constructs Dataset objects that emit (ticker_id, windowed features, targets).

    :param df: Input long-form DataFrame with columns ['date','ticker', cfg.target_col, 'return', ...].
    :type df: pandas.DataFrame
    :param cfg: Training configuration with window, horizons, validation window, and feature list.
    :type cfg: TrainConfig
    :raises KeyError: If required columns are missing from df.
    :return: (train_dataset, val_dataset, ticker_to_id, scalers)
    :rtype: tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, dict[str,int], dict[str,StandardScaler]]
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["ticker", "date"], inplace=True)

    required = ["date", "ticker", cfg.target_col, "return"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    tickers = sorted(df["ticker"].unique())
    ticker_to_id = {t: i for i, t in enumerate(tickers)}
    df["ticker_id"] = df["ticker"].map(ticker_to_id)

    features = ["return"] + (cfg.extra_features or [])
    scalers: Dict[str, StandardScaler] = {}
    scaled_parts = []
    for t in tickers:
        sub = df[df["ticker"] == t].copy()
        scaler = StandardScaler()
        sub[features] = scaler.fit_transform(sub[features].astype(float).fillna(0.0))
        scalers[t] = scaler
        scaled_parts.append(sub)
    df_scaled = pd.concat(scaled_parts, ignore_index=True)

    # inside build_global_splits(...) after df_scaled is built
    df_scaled["date"] = pd.to_datetime(df_scaled["date"])
    val_end = getattr(cfg, "val_end", None)
    val_mode = getattr(cfg, "val_mode", "causal")
    embargo = int(getattr(cfg, "embargo_days", 0))

    if val_end is None:
        # original behavior: single cut by val_start
        train_df = df_scaled[df_scaled["date"] < cfg.val_start].reset_index(drop=True)
        val_df = df_scaled[df_scaled["date"] >= cfg.val_start].reset_index(drop=True)
    else:
        mask_val = (df_scaled["date"] >= cfg.val_start) & (df_scaled["date"] < val_end)

        if val_mode == "holdout_slice":
            # train on everything except the held-out slice (+/- embargo window)
            if embargo > 0:
                # widen the excluded region by embargo on both sides
                lo = pd.to_datetime(cfg.val_start) - pd.Timedelta(days=embargo)
                hi = pd.to_datetime(val_end) + pd.Timedelta(days=embargo)
                mask_excl = (df_scaled["date"] >= lo) & (df_scaled["date"] < hi)
            else:
                mask_excl = mask_val
            train_df = df_scaled[~mask_excl].reset_index(drop=True)
            val_df = df_scaled[mask_val].reset_index(drop=True)

        else:  # "causal"
            # forward test flavor: train strictly before val_start; validate within [start, end)
            train_df = df_scaled[df_scaled["date"] < cfg.val_start].reset_index(
                drop=True
            )
            val_df = df_scaled[mask_val].reset_index(drop=True)

    class GlobalVolDataset(torch.utils.data.Dataset):
        def __init__(self, df: pd.DataFrame, cfg: TrainConfig, is_train: bool):
            self.df = df
            self.cfg = cfg
            self.is_train = is_train
            self.groups = {
                tid: g.reset_index(drop=True) for tid, g in df.groupby("ticker_id")
            }
            self.samples = []
            Hmax = (
                max(cfg.horizons)
                if isinstance(cfg.horizons, (list, tuple))
                else cfg.horizons
            )
            for tid, g in self.groups.items():
                for i in range(0, len(g) - cfg.window - Hmax, cfg.stride):
                    self.samples.append((tid, i))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            tid, start = self.samples[idx]
            g = self.groups[tid]

            W = self.cfg.window
            if (
                self.is_train
                and hasattr(self.cfg, "dynamic_window_jitter")
                and self.cfg.dynamic_window_jitter > 0
            ):
                delta = np.random.randint(
                    -self.cfg.dynamic_window_jitter, self.cfg.dynamic_window_jitter + 1
                )
                W = max(10, self.cfg.window + int(delta))

            feats = ["return"] + (self.cfg.extra_features or [])
            X_df = g.iloc[start : start + W][feats]

            # Build target(s)
            if isinstance(self.cfg.horizons, int):
                t_idx = min(start + W + self.cfg.horizons - 1, len(g) - 1)
                y_vals = [g.iloc[t_idx][self.cfg.target_col]]
            else:
                y_vals = []
                for h in self.cfg.horizons:
                    t_idx = start + W + h - 1
                    y_vals.append(
                        g.iloc[t_idx][self.cfg.target_col] if t_idx < len(g) else np.nan
                    )

            X = torch.tensor(X_df.values, dtype=torch.float32)
            # --- Pad or crop to fixed cfg.window ---
            if X.shape[0] < self.cfg.window:
                pad = torch.zeros(
                    self.cfg.window - X.shape[0], X.shape[1], dtype=torch.float32
                )
                X = torch.cat([X, pad], dim=0)
            elif X.shape[0] > self.cfg.window:
                X = X[-self.cfg.window :]

            y = torch.tensor(y_vals, dtype=torch.float32)
            y = torch.nan_to_num(y, nan=0.0)

            return torch.tensor(tid, dtype=torch.long), X, y

    train_ds = GlobalVolDataset(train_df, cfg, is_train=True)
    val_ds = GlobalVolDataset(val_df, cfg, is_train=False)
    return train_ds, val_ds, ticker_to_id, scalers


# ============================================================
# 🔁 EMA helper
# ============================================================
def clone_model_like(model: nn.Module) -> nn.Module:
    """
    Create a non-trainable copy of a model for EMA tracking.

    :param model: Source model to clone.
    :type model: torch.nn.Module
    :return: Cloned model with requires_grad=False for all parameters.
    :rtype: torch.nn.Module
    """
    ema = type(model)(**model._init_args) if hasattr(model, "_init_args") else None
    if ema is None:
        # generic fallback: deep copy of state dict to identically-constructed model
        import copy

        ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    return ema


def update_ema(ema: nn.Module, model: nn.Module, decay: float):
    """
    Update EMA model weights and buffers from a source model.

    :param ema: EMA model to be updated in-place.
    :type ema: torch.nn.Module
    :param model: Source model providing current parameters/buffers.
    :type model: torch.nn.Module
    :param decay: EMA decay factor in (0,1); closer to 1 gives slower updates.
    :type decay: float
    :return: None
    :rtype: None
    """
    with torch.no_grad():
        for (n_e, p_e), (_, p_m) in zip(
            ema.named_parameters(), model.named_parameters()
        ):
            p_e.data.mul_(decay).add_(p_m.data, alpha=1 - decay)
        for (n_e, b_e), (_, b_m) in zip(ema.named_buffers(), model.named_buffers()):
            b_e.data.copy_(b_m.data)


# ============================================================
# 🚀 Training Loop (no manual loops outside; returns artifacts)
# ============================================================
def train_global_model(df: pd.DataFrame, cfg: TrainConfig):
    """
    Train GlobalVolForecaster end-to-end and return artifacts.

    Includes optional cosine scheduling with restarts, EMA tracking, gradient clipping,
    early stopping, and horizon-weighted MSE.

    :param df: Input DataFrame containing ['date','ticker','return', cfg.target_col, ...].
    :type df: pandas.DataFrame
    :param cfg: Training configuration.
    :type cfg: TrainConfig
    :return: Tuple of (model, history, val_loader, ticker_to_id, scalers, features).
    :rtype: tuple[torch.nn.Module, dict[str, list[float]], torch.utils.data.DataLoader, dict[str,int], dict[str,StandardScaler], list[str]]
    """
    # Datasets / loaders
    train_ds, val_ds, ticker_to_id, scalers = build_global_splits(df, cfg)
    features = ["return"] + (cfg.extra_features or [])

    model = GlobalVolForecaster(
        n_tickers=len(ticker_to_id),
        window=cfg.window,
        n_horizons=cfg.horizons,
        emb_dim=16,
        hidden_dim=160,
        num_layers=3,
        dropout=cfg.dropout,
        attention=cfg.attention,
        residual_head=cfg.residual_head,
        input_size=len(features),
        use_layernorm=cfg.use_layernorm,
        separate_heads=cfg.separate_heads,
        feat_dropout_p=cfg.feat_dropout_p,
        variational_dropout_p=cfg.variational_dropout_p,
    )
    # Save init args for EMA clone
    model._init_args = dict(
        n_tickers=len(ticker_to_id),
        window=cfg.window,
        n_horizons=cfg.horizons,
        emb_dim=16,
        hidden_dim=160,
        num_layers=3,
        dropout=cfg.dropout,
        attention=cfg.attention,
        residual_head=cfg.residual_head,
        input_size=len(features),
        use_layernorm=cfg.use_layernorm,
        separate_heads=cfg.separate_heads,
        feat_dropout_p=cfg.feat_dropout_p,
        variational_dropout_p=cfg.variational_dropout_p,
    )

    device = torch.device(cfg.device)
    model.to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # scheduler
    if cfg.cosine_schedule and cfg.cosine_restarts:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_anneal := max(2, cfg.epochs // 4), eta_min=1e-6
        )
    elif cfg.cosine_schedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=1e-6
        )
    else:
        scheduler = None

    # multi-horizon MSE with optional horizon weights
    def mh_mse(pred, target):
        if pred.ndim == 1 or pred.shape[-1] == 1:
            return torch.mean((pred - target.squeeze(-1)) ** 2)
        if cfg.loss_horizon_weights is None:
            return torch.mean((pred - target) ** 2)
        w = torch.tensor(cfg.loss_horizon_weights, device=pred.device, dtype=pred.dtype)
        return torch.mean(((pred - target) ** 2) * w)

    # EMA
    ema_model = clone_model_like(model) if cfg.use_ema else None
    if ema_model is not None:
        ema_model.load_state_dict(model.state_dict())
        ema_model.to(device)  # ✅ move EMA model to same device as main model

    # early stopping
    best_val = float("inf")
    best_sd = None
    patience_left = cfg.patience

    history = {"train": [], "val": []}
    print(f"\n🚀 Training GlobalVolForecaster on {len(ticker_to_id)} tickers...\n")

    for ep in range(1, cfg.epochs + 1):
        # ----- Train -----
        model.train()
        train_loss = 0.0
        for t_id, X, y in train_loader:
            t_id, X, y = t_id.to(device), X.to(device), y.to(device)
            optimizer.zero_grad()
            yhat = model(t_id, X)
            loss = mh_mse(yhat, y)
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            if ema_model is not None:
                update_ema(ema_model, model, cfg.ema_decay)
            train_loss += loss.item() * len(t_id)
        train_loss /= len(train_loader.dataset)

        # ----- Validate -----
        def _eval(m):
            m.eval()
            vl = 0.0
            with torch.no_grad():
                for t_id, X, y in val_loader:
                    t_id, X, y = t_id.to(device), X.to(device), y.to(device)
                    yhat = m(t_id, X)
                    vl += mh_mse(yhat, y).item() * len(t_id)
            return vl / len(val_loader.dataset)

        val_loss = _eval(model)
        if ema_model is not None:
            val_loss = min(val_loss, _eval(ema_model))  # pick better of EMA/current

        if scheduler:
            if isinstance(
                scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
            ):
                scheduler.step(ep - 1)  # epoch-based step for restarts
            else:
                scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {ep}/{cfg.epochs}| LR: {cur_lr:.4e} |  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        # Early stopping
        if cfg.early_stop:
            if val_loss + cfg.min_delta < best_val:
                best_val = val_loss
                best_sd = copy.deepcopy((ema_model or model).state_dict())
                patience_left = cfg.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print("⏹️  Early stopping.")
                    break

    # Load best weights (EMA if available)
    if best_sd is not None:
        print(f"🔙 Restoring best model weights (Loss: {best_val:.5f})...")
        (ema_model or model).load_state_dict(best_sd)
        if ema_model is not None:
            model.load_state_dict(ema_model.state_dict())

    print(f"\n✅ Training complete with feature set: {features}\n")
    return model, history, val_loader, ticker_to_id, scalers, features


# ============================================================
# 🔹 Helper Utilities for Inference and Checkpointing
# ============================================================


def make_last_windows(df: pd.DataFrame, window: int):
    """
    Extract the last `window` timesteps per ticker for inference.

    :param df: Feature DataFrame with ['ticker','date', <features...>] sorted by date within ticker.
    :type df: pandas.DataFrame
    :param window: Number of trailing rows to keep per ticker.
    :type window: int
    :return: Concatenated DataFrame of last windows for all tickers.
    :rtype: pandas.DataFrame
    """
    df = df.copy().sort_values(["ticker", "date"])
    last_windows = []
    for t, g in df.groupby("ticker"):
        g = g.tail(window).copy()
        g["ticker"] = t
        last_windows.append(g)
    return pd.concat(last_windows, ignore_index=True)


def predict_next_day(
    model,
    df_last_windows,
    ticker_to_id,
    scalers,
    window,
    device="cpu",
    show_progress=True,
):
    """
    Predict next-day (multi-horizon) volatility for each ticker using the latest window.

    Aligns inference features to the per-ticker scaler schema (fills missing features with 0,
    drops extras), scales inputs, runs the model, and returns a tidy table.

    :param model: Trained GlobalVolForecaster.
    :type model: torch.nn.Module
    :param df_last_windows: DataFrame with the most recent window per ticker; must include ['ticker','date', <features>].
    :type df_last_windows: pandas.DataFrame
    :param ticker_to_id: Mapping from ticker symbol to integer ID used by the model.
    :type ticker_to_id: dict[str, int]
    :param scalers: Per-ticker fitted StandardScaler objects (from training).
    :type scalers: dict[str, StandardScaler]
    :param window: Window length used during training (for validation of shapes).
    :type window: int
    :param device: Compute device for inference ('cpu' or 'cuda').
    :type device: str
    :param show_progress: Show a tqdm progress bar during prediction.
    :type show_progress: bool
    :return: DataFrame with ['ticker','horizon','forecast_vol_scaled'].
    :rtype: pandas.DataFrame
    """

    model.eval()
    preds = []

    tickers = df_last_windows["ticker"].unique()
    # print(f"Predicting next-day vols for {len(tickers)} tickers...")

    # Disable tqdm if nested
    iterator = tqdm(
        df_last_windows.groupby("ticker"),
        desc="Predicting next-day vols",
        disable=not show_progress,
    )

    for t, g in iterator:
        if t not in ticker_to_id:
            print(f"⚠️ Skipping {t}: not found in training ticker map.")
            continue

        t_id = torch.tensor([ticker_to_id[t]], dtype=torch.long, device=device)
        scaler = scalers[t]

        # --- Align feature sets between training and inference ---
        fitted_feats = list(getattr(scaler, "feature_names_in_", []))
        current_feats = list(g.columns)

        if fitted_feats:
            missing = [f for f in fitted_feats if f not in current_feats]
            extra = [f for f in current_feats if f not in fitted_feats]

            if missing:
                print(f"⚠️ Missing features for {t}: {missing} (filled with zeros)")
                for f in missing:
                    g[f] = 0.0
            if extra:
                g = g.drop(columns=extra, errors="ignore")

            g = g[fitted_feats]
        else:
            print(
                f"⚠️ Scaler for {t} has no feature_names_in_; using numeric columns only."
            )
            g = g.select_dtypes(include=np.number)

        # --- Scale and predict (robust alignment) ---
        g_num = g.select_dtypes(include=np.number).copy()

        if getattr(scaler, "feature_names_in_", None):
            expected_cols = [c for c in scaler.feature_names_in_ if c in g_num.columns]
            # reindex will add missing names as NaN -> fill with 0.0
            g_num = g_num.reindex(columns=expected_cols, fill_value=0.0)
        else:
            expected_len = None
            if getattr(scaler, "mean_", None) is not None:
                try:
                    expected_len = int(scaler.mean_.shape[1])
                except Exception:
                    expected_len = None

            if expected_len is not None and g_num.shape[1] != expected_len:
                print(
                    f"⚠️ scaler dim ({expected_len}) != numeric cols ({g_num.shape[1]}). "
                    "Trimming/padding numeric inputs to match."
                )
                if g_num.shape[1] > expected_len:
                    g_num = g_num.iloc[:, :expected_len]
                else:
                    for i in range(expected_len - g_num.shape[1]):
                        g_num[f"_pad_{i}"] = 0.0

        X_scaled = scaler.transform(g_num.fillna(0.0))
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device).unsqueeze(
            0
        )

        with torch.no_grad():
            yhat = model(t_id, X_tensor)

        yhat_np = np.atleast_1d(yhat.cpu().numpy().flatten())

        # --- Expand multi-horizon forecasts ---
        for i, val in enumerate(yhat_np):
            preds.append(
                {
                    "ticker": t,
                    "horizon": i + 1,  # or use cfg.horizons if available
                    "forecast_vol_scaled": float(val),
                }
            )

    # --- Create standardized DataFrame ---
    out = pd.DataFrame(preds)

    # Defensive fallback in case of unexpected output
    if not isinstance(out, pd.DataFrame):
        out = pd.DataFrame(out, columns=["ticker", "horizon", "forecast_vol_scaled"])

    return out
