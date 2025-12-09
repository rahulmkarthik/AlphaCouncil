import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from volsense_core.utils.scalers import TorchStandardScaler as StandardScaler
from dataclasses import dataclass
from typing import List, Tuple


def collate_with_dates(batch):
    """
    Collate function that stacks tensors and keeps Python datetime objects as a list.

    :param batch: Iterable of (X, y, date) triplets produced by the Dataset.
    :type batch: list[tuple[torch.Tensor, torch.Tensor, pandas.Timestamp]]
    :return: Tuple of stacked tensors and a list of dates: (X_batch, y_batch, dates).
    :rtype: tuple[torch.Tensor, torch.Tensor, list]
    """
    xs, ys, dates = zip(*batch)  # unpack triplets
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    return xs, ys, list(dates)


# ============================================================
# 1) Training configuration
# ============================================================
@dataclass
class TrainConfig:
    """
    Training configuration for BaseLSTM.

    :param window: Rolling lookback window length (timesteps).
    :type window: int
    :param horizons: Forecast horizons (e.g., [1, 5, 10]).
    :type horizons: list[int]
    :param val_start: Validation set start date (inclusive, YYYY-MM-DD).
    :type val_start: str
    :param target_col: Target column name to predict.
    :type target_col: str
    :param extra_features: Additional feature names beyond 'return' to include.
    :type extra_features: list[str], optional
    :param batch_size: Mini-batch size for DataLoader.
    :type batch_size: int
    :param epochs: Number of training epochs.
    :type epochs: int
    :param lr: Learning rate.
    :type lr: float
    :param weight_decay: L2 weight decay.
    :type weight_decay: float
    :param device: Compute device ('cpu' or 'cuda').
    :type device: str
    :param dropout: Dropout probability used in model modules.
    :type dropout: float
    :param hidden_dim: LSTM hidden size.
    :type hidden_dim: int
    :param num_layers: Number of stacked LSTM layers.
    :type num_layers: int
    :param use_layernorm: Whether to apply LayerNorm to LSTM outputs.
    :type use_layernorm: bool
    :param use_attention: Whether to apply self-attention over LSTM outputs.
    :type use_attention: bool
    :param feat_dropout_p: Feature-wise dropout probability at input.
    :type feat_dropout_p: float
    :param residual_head: Whether to add a residual linear projection to outputs.
    :type residual_head: bool
    :param output_activation: Output activation ('none' | 'softplus' | 'relu').
    :type output_activation: str
    :param num_workers: DataLoader workers.
    :type num_workers: int
    :param pin_memory: DataLoader pin_memory flag.
    :type pin_memory: bool
    """

    window: int = 30
    horizons: List[int] = None
    val_start: str = None
    target_col: str = "realized_vol"
    extra_features: list[str] | None = None

    batch_size: int = 128
    epochs: int = 15
    lr: float = 5e-4
    weight_decay: float = 1e-5
    device: str = "cpu"

    dropout: float = 0.2
    hidden_dim: int = 128
    num_layers: int = 3
    use_layernorm: bool = True
    use_attention: bool = True
    feat_dropout_p: float = 0.1
    residual_head: bool = True
    output_activation: str = "none"  # "none" | "softplus" | "relu"

    num_workers: int = 0
    pin_memory: bool = False


# ============================================================
# 2) Dataset
# ============================================================
class MultiVolDataset(Dataset):
    """
    Windowed multi-horizon dataset for single- or multi-ticker realized vol data.

    Expected columns in df:
        ["date", "ticker", <features...>, target_col]
    Behavior:
        • Sorts by (ticker, date)
        • Standardizes features with StandardScaler (train fit, val transform)
        • Standardizes target using train set y_mean/y_std; reuses for val
        • Drops constant features to avoid StandardScaler NaNs
        • Builds rolling windows of length `cfg.window` with multi-horizon labels
    """

    def __init__(
        self, df: pd.DataFrame, cfg: TrainConfig, scaler: StandardScaler = None
    ):
        """
        Build a rolling-window dataset and (optionally) fit a StandardScaler on features.

        :param df: Long-form DataFrame sorted or sortable by ['ticker','date'] with features and target.
        :type df: pandas.DataFrame
        :param cfg: Training configuration (window, horizons, target_col, etc.).
        :type cfg: TrainConfig
        :param scaler: Optional pre-fitted StandardScaler (use None for training split to fit).
        :type scaler: sklearn.preprocessing.StandardScaler, optional
        :raises KeyError: If required 'date' or 'ticker' columns are missing.
        :raises ValueError: If there are not enough rows to form at least one sample.
        :return: None
        :rtype: None
        """
        df = df.copy()

        if "date" not in df.columns:
            if "Date" in df.columns:
                df = df.rename(columns={"Date": "date"})
            else:
                raise KeyError("Expected a 'date' column in the input DataFrame.")
        if "ticker" not in df.columns:
            raise KeyError("Expected a 'ticker' column in the input DataFrame.")

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        self.window = cfg.window
        self.horizons = (
            cfg.horizons if isinstance(cfg.horizons, (list, tuple)) else [cfg.horizons]
        )
        self.scaler = scaler or StandardScaler()

        # ----- Target standardization (fit on train, reuse on val/test) -----
        self.is_train = scaler is None
        if self.is_train:
            self.y_mean = float(df[cfg.target_col].mean())
            self.y_std = float(df[cfg.target_col].std())
            # Persist onto cfg so val/test can reuse exactly the same stats
            setattr(cfg, "y_mean", self.y_mean)
            setattr(cfg, "y_std", self.y_std)
        else:
            self.y_mean = float(getattr(cfg, "y_mean"))
            self.y_std = float(getattr(cfg, "y_std"))

        # Standardize target; keep it unconstrained (can be negative)
        df[cfg.target_col] = (df[cfg.target_col] - self.y_mean) / (self.y_std + 1e-8)

        # ----- Feature prep -----
        feat_cols = [
            c for c in df.columns if c not in ["date", "ticker", cfg.target_col]
        ]
        X = (
            df[feat_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(np.float32)
        )

        const_cols = [c for c in X.columns if X[c].std() == 0]
        if const_cols:
            print(f"⚠️ Dropping constant features: {const_cols}")
            X = X.drop(columns=const_cols)
            df = df.drop(columns=const_cols)
            feat_cols = [c for c in feat_cols if c not in const_cols]

        # Fit/transform features
        if self.is_train:
            self.scaler.fit(X)
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=feat_cols)
        df[feat_cols] = X_scaled

        # ----- Build samples -----
        self.feat_cols = feat_cols
        self.samples: List[Tuple[np.ndarray, List[float]]] = []
        self.sample_dates: List[pd.Timestamp] = []  # <--- added for chronological eval
        max_h = max(self.horizons)

        if len(df) < self.window + max_h:
            raise ValueError(
                f"Insufficient samples: need at least {self.window + max_h}, got {len(df)}."
            )

        for _, group in df.groupby("ticker"):
            arr = group[feat_cols].values.astype(np.float32)
            y_series = group[cfg.target_col].values.astype(np.float32)
            dates = group["date"].values
            for i in range(len(arr) - cfg.window - max_h + 1):
                X_window = arr[i : i + cfg.window]
                y_vals = [y_series[i + cfg.window + h - 1] for h in self.horizons]
                last_date = dates[i + cfg.window - 1]
                self.samples.append((X_window, y_vals))
                self.sample_dates.append(last_date)

    def __len__(self) -> int:
        """
        Number of samples (rolling windows) in the dataset.

        :return: Count of samples.
        :rtype: int
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a single sample containing features, labels, and the window end date.

        :param idx: Sample index.
        :type idx: int
        :return: Tuple (X_window, y_horizons, date) where:
                 - X_window: torch.FloatTensor of shape [window, features]
                 - y_horizons: torch.FloatTensor of shape [H] or scalar when H=1
                 - date: pandas.Timestamp of the last row in the window
        :rtype: tuple[torch.Tensor, torch.Tensor, pandas.Timestamp]
        """
        X, y_vals = self.samples[idx]
        date = self.sample_dates[idx]
        y = torch.tensor(y_vals, dtype=torch.float32)
        if y.numel() == 1:
            y = y.squeeze(0)
        return torch.tensor(X, dtype=torch.float32), y, date  # <--- now returns date


# ============================================================
# 3) Model
# ============================================================
class FeatureDropout(nn.Module):
    """Feature-wise dropout mask shared across time steps (stochastic feature erasing)."""

    def __init__(self, p: float = 0.1):
        """
        Initialize FeatureDropout.

        :param p: Drop probability for feature channels (0 <= p < 1).
        :type p: float
        """
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel-wise dropout with a single mask per sample across time.

        :param x: Input tensor of shape [B, W, F].
        :type x: torch.Tensor
        :return: Tensor with dropped channels, scaled by 1/(1-p) during training.
        :rtype: torch.Tensor
        """
        if not self.training or self.p <= 0:
            return x
        B, W, F = x.shape
        mask = torch.empty(B, 1, F, device=x.device).bernoulli_(1 - self.p)
        return x * mask / (1 - self.p)


class BaseLSTM(nn.Module):
    """
    LSTM-based multi-horizon forecaster with optional self-attention and residual head.

    Input:  (B, W, F)   where W=window, F=features
    Output: (B, H)      where H=len(horizons)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        n_horizons: int = 1,
        use_layernorm: bool = True,
        use_attention: bool = True,
        feat_dropout_p: float = 0.1,
        residual_head: bool = True,
        output_activation: str = "none",  # "none" | "softplus" | "relu"
    ):
        """
        Initialize BaseLSTM backbone and heads.

        :param input_dim: Number of input features per timestep.
        :type input_dim: int
        :param hidden_dim: LSTM hidden size.
        :type hidden_dim: int
        :param num_layers: Number of stacked LSTM layers.
        :type num_layers: int
        :param dropout: Dropout probability used in LSTM and MLP head.
        :type dropout: float
        :param n_horizons: Number of output horizons.
        :type n_horizons: int
        :param use_layernorm: Apply LayerNorm to LSTM outputs.
        :type use_layernorm: bool
        :param use_attention: Apply MultiheadAttention over LSTM outputs.
        :type use_attention: bool
        :param feat_dropout_p: Feature-wise input dropout probability.
        :type feat_dropout_p: float
        :param residual_head: Add residual linear projection to outputs.
        :type residual_head: bool
        :param output_activation: Output activation ('none' | 'softplus' | 'relu').
        :type output_activation: str
        """
        super().__init__()
        self.use_attention = use_attention
        self.residual_head = residual_head
        self.n_horizons = n_horizons

        self.feat_dropout = FeatureDropout(feat_dropout_p)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.ln = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()
        self.attn = (
            nn.MultiheadAttention(
                hidden_dim, num_heads=2, dropout=dropout, batch_first=True
            )
            if use_attention
            else None
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_horizons),
        )
        self.res_proj = nn.Linear(hidden_dim, n_horizons) if residual_head else None

        if output_activation == "softplus":
            self.out_act = nn.Softplus(beta=1.0)
        elif output_activation == "relu":
            self.out_act = nn.ReLU()
        else:
            self.out_act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feature-dropout, LSTM, optional attention, and heads.

        :param x: Input tensor of shape [B, W, F].
        :type x: torch.Tensor
        :return: Predictions of shape [B, H] (or [B] when H==1).
        :rtype: torch.Tensor
        """
        x = self.feat_dropout(x)
        out, _ = self.lstm(x)
        if self.attn is not None:
            out, _ = self.attn(out, out, out)
        out = self.ln(out)
        last = out[:, -1, :]
        yhat = self.head(last)
        if self.res_proj is not None:
            yhat = yhat + self.res_proj(last)
        yhat = self.out_act(yhat)
        return yhat.squeeze(-1) if self.n_horizons == 1 else yhat


# ============================================================
# 4) Training / evaluation utilities
# ============================================================
def build_splits(
    df: pd.DataFrame, cfg: TrainConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into train/validation sets by date.

    :param df: Input feature DataFrame with a 'date' column.
    :type df: pandas.DataFrame
    :param cfg: Training configuration with 'val_start'.
    :type cfg: TrainConfig
    :return: Tuple of (train_df, val_df).
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame]
    """
    val_start = pd.to_datetime(cfg.val_start)
    train_df = df[df["date"] < val_start].copy()
    val_df = df[df["date"] >= val_start].copy()
    return train_df, val_df


def build_dataloaders(df: pd.DataFrame, cfg: TrainConfig):
    """
    Create training and validation DataLoaders and return feature columns.

    :param df: Input feature DataFrame.
    :type df: pandas.DataFrame
    :param cfg: Training configuration (window, horizons, batch size, etc.).
    :type cfg: TrainConfig
    :return: (train_loader, val_loader, feature_columns)
    :rtype: tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, list[str]]
    """
    train_df, val_df = build_splits(df, cfg)

    train_ds = MultiVolDataset(train_df, cfg)
    val_ds = MultiVolDataset(val_df, cfg, scaler=train_ds.scaler)

    # Use custom collate_fn to safely handle datetime objects
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_with_dates,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=collate_with_dates,
    )

    return train_loader, val_loader, train_ds.feat_cols


def train_baselstm(df: pd.DataFrame, cfg: TrainConfig):
    """
    Train the BaseLSTM model and return the model, history, and loaders.

    :param df: Input feature DataFrame with required columns and target.
    :type df: pandas.DataFrame
    :param cfg: Training configuration.
    :type cfg: TrainConfig
    :return: Tuple (model, history, (train_loader, val_loader)).
    :rtype: tuple[torch.nn.Module, dict[str, list[float]], tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]]
    """
    train_loader, val_loader, feat_cols = build_dataloaders(df, cfg)
    setattr(cfg, "features", feat_cols)
    setattr(cfg, "extra_features", [f for f in feat_cols if f != "return"])

    model = BaseLSTM(
        input_dim=len(feat_cols),
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        n_horizons=len(cfg.horizons),
        use_layernorm=cfg.use_layernorm,
        use_attention=cfg.use_attention,
        feat_dropout_p=cfg.feat_dropout_p,
        residual_head=cfg.residual_head,
        output_activation=cfg.output_activation,
    )

    device = cfg.device
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs, eta_min=1e-6
    )
    crit = nn.MSELoss(reduction="mean")

    hist = {"train": [], "val": []}

    for ep in range(cfg.epochs):
        # ---------- Train ----------
        model.train()
        tr_sum, n_train = 0.0, 0
        for batch in train_loader:  # ✅ supports (X, y, date)
            if len(batch) == 3:
                X, y, _ = batch
            else:
                X, y = batch
            X, y = X.to(device), y.to(device)
            opt.zero_grad()

            preds = model(X)
            preds = torch.nan_to_num(preds, nan=0.0, posinf=1e6, neginf=-1e6)
            y = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)

            loss = crit(preds, y)
            if torch.isnan(loss):
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = X.size(0)
            tr_sum += loss.item() * bs
            n_train += bs

        tr_loss = tr_sum / max(1, n_train)

        # ---------- Validate ----------
        model.eval()
        val_sum, n_val = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:  # ✅ supports (X, y, date)
                if len(batch) == 3:
                    X, y, _ = batch
                else:
                    X, y = batch
                X, y = X.to(device), y.to(device)
                preds = model(X)
                preds = torch.nan_to_num(preds, nan=0.0, posinf=1e6, neginf=-1e6)
                loss = crit(preds, y)

                bs = X.size(0)
                val_sum += loss.item() * bs
                n_val += bs

        val_loss = val_sum / max(1, n_val)
        sched.step()

        hist["train"].append(tr_loss)
        hist["val"].append(val_loss)
        cur_lr = opt.param_groups[0]["lr"]
        print(
            f"Epoch {ep+1}/{cfg.epochs} | LR: {cur_lr:.2e} | Train: {tr_loss:.6f} | Val: {val_loss:.6f}"
        )

    return model, hist, (train_loader, val_loader)


def evaluate_baselstm(
    model: nn.Module, loader: DataLoader, cfg: TrainConfig, device: str = "cpu"
):
    """
    Evaluate model on a DataLoader and return de-standardized arrays.

    If cfg contains y_mean/y_std (set during training), both predictions and targets
    are transformed back to the original volatility scale.

    :param model: Trained PyTorch model to evaluate.
    :type model: torch.nn.Module
    :param loader: DataLoader providing batches from the evaluation split.
    :type loader: torch.utils.data.DataLoader
    :param cfg: Training config containing target standardization stats.
    :type cfg: TrainConfig
    :param device: Compute device for inference ('cpu' or 'cuda').
    :type device: str
    :return: Tuple of (preds, actuals) each shaped (N, H).
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    model.eval()
    preds, actuals, dates = [], [], []
    with torch.no_grad():
        for batch in loader:  # ✅ supports (X, y, date)
            if len(batch) == 3:
                X, y, d = batch
                dates.extend(d)
            else:
                X, y = batch
            X = X.to(device)
            yhat = model(X).cpu().numpy()
            preds.append(yhat)
            actuals.append(y.numpy())

    preds = np.vstack(preds)
    actuals = np.vstack(actuals)

    if hasattr(cfg, "y_mean") and hasattr(cfg, "y_std"):
        preds = preds * cfg.y_std + cfg.y_mean
        actuals = actuals * cfg.y_std + cfg.y_mean

    return preds, actuals
