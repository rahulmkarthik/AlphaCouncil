import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import copy
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Union
from torch.utils.data import DataLoader, TensorDataset
from volsense_core.utils.scalers import TorchStandardScaler

# --------------------------------------------------------------------
# 🔧 Config (Updated for Per-Ticker Scaling & Feature Dropout)
# --------------------------------------------------------------------
@dataclass
class VolNetXConfig:
    window: int = 65
    horizons: List[int] = field(default_factory=lambda: [1, 5, 10])
    input_size: int = 16
    hidden_dim: int = 128      # Balanced capacity
    emb_dim: int = 16
    num_layers: int = 2        # Stability setting
    dropout: float = 0.25      # Balanced dropout
    feat_dropout: float = 0.1  # <--- NEW: Feature-level dropout
    
    use_transformer: bool = True
    use_ticker_embedding: bool = True
    use_feature_attention: bool = True
    separate_heads: bool = True
    use_layernorm: bool = True
    
    loss_horizon_weights: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.3])
    device: str = "cpu"
    early_stop: bool = True
    patience: int = 15
    lr: float = 3e-4
    cosine_schedule: bool = True
    grad_clip: float = 0.5
    weight_decay: float = 1e-3
    loss_type: str = "mse" # 'mse' or 'huber'
    batch_size: int = 128
    epochs: int = 50
    
    val_mode: str = "causal"
    val_start: Optional[str] = None
    val_end: Optional[str] = None
    embargo_days: int = 30  # drop these days on either side of val window from train
    target_col: str = "realized_log_vol"  # target column for training
    extra_features: Optional[List[str]] = None

# --------------------------------------------------------------------
# 🧠 Helper: Feature Dropout (Ported from Global Baseline)
# --------------------------------------------------------------------
class FeatureDropout(nn.Module):
    """Drops entire feature channels to force robust learning."""
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p <= 0: return x
        # x: [B, T, F]
        mask = torch.bernoulli(torch.ones(x.size(0), 1, x.size(2), device=x.device) * (1 - self.p))
        return x * mask / (1 - self.p)

# --------------------------------------------------------------------
# 🧠 Model Architecture
# --------------------------------------------------------------------
class VolNetX(nn.Module):
    def __init__(self, n_tickers, window, input_size, horizons, 
                 hidden_dim=128, emb_dim=16, num_layers=2, dropout=0.25,
                 feat_dropout=0.1, use_transformer=True, 
                 use_ticker_embedding=True, use_feature_attention=True, 
                 separate_heads=True, use_layernorm=True):
        super().__init__()
        self.n_tickers = n_tickers
        self.horizons = horizons
        self.use_ticker_embedding = use_ticker_embedding
        self.use_feature_attention = use_feature_attention
        self.separate_heads = separate_heads
        self.use_transformer = use_transformer
        self.window = window
        

        # 1. Feature Regularization (Critical for Transformers)
        self.feat_do = FeatureDropout(feat_dropout)

        # 2. Embeddings
        self.emb = nn.Embedding(n_tickers, emb_dim) if use_ticker_embedding else None
        final_input_dim = input_size + (emb_dim if use_ticker_embedding else 0)

        # 3. Gated Feature Selection
        if use_feature_attention:
            self.feat_gate = nn.Sequential(
                nn.Linear(final_input_dim, final_input_dim),
                nn.Sigmoid()
            )

        # 4. LSTM Backbone
        self.lstm = nn.LSTM(
            input_size=final_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        
        # 5. Transformer Context
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim*4, 
                dropout=dropout, batch_first=True, norm_first=True
            )
            # Disable nested tensor to silence warnings
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1, enable_nested_tensor=False)

        self.ln = nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()

        # 6. Heads
        out_dim = len(horizons)
        if separate_heads:
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 1)
                ) for _ in horizons
            ])
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, out_dim)
            )

        # 7. Residual Highway
        self.residual_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, tidx, x):
        # x: [B, W, F]
        
        # Apply Feature Dropout FIRST
        x = self.feat_do(x)
        
        if self.use_ticker_embedding and self.emb is not None:
            emb = self.emb(tidx).unsqueeze(1).expand(-1, x.size(1), -1)
            x = torch.cat([x, emb], dim=-1)

        if self.use_feature_attention:
            gate = self.feat_gate(x) 
            x = x * gate

        x, _ = self.lstm(x)

        if hasattr(self, "transformer"):
            x = self.transformer(x)

        last_step = self.ln(x[:, -1, :])

        if self.separate_heads:
            y_nonlinear = torch.cat([h(last_step) for h in self.heads], dim=-1)
        else:
            y_nonlinear = self.head(last_step)

        y_residual = self.residual_head(last_step)
        
        return y_nonlinear + y_residual

# --------------------------------------------------------------------
# 🛠️ Dataset Preparation (CORRECTED: Per-Ticker Scaling)
# --------------------------------------------------------------------
def build_volnetx_dataset(
    df: pd.DataFrame,
    features: List[str],
    target_col: str = "realized_vol_log",
    config: VolNetXConfig = None,
    scaler: Optional[Dict] = None, # <--- CHANGED: Expects a Dict of scalers
) -> Tuple[Dict[str, int], DataLoader, DataLoader, TensorDataset, TensorDataset, Dict]:
    
    window = config.window if config else 65
    horizons = config.horizons if config else [1, 5, 10]
    batch_size = config.batch_size if config else 128
    
    df = df.sort_values(["ticker", "date"]).copy()
    tickers = df["ticker"].unique().tolist()
    ticker_to_id = {t: i for i, t in enumerate(tickers)}
    df["tidx"] = df["ticker"].map(ticker_to_id)

    # Split Logic
    val_mask = pd.Series(False, index=df.index)
    if config and config.val_start:
         val_mask = df["date"] >= pd.to_datetime(config.val_start)
    df["split"] = val_mask.astype(int)

    # --- 🚀 NEW: Per-Ticker Scaling Logic ---
    if scaler is None:
        print("   ⚖️ Fitting PER-TICKER scalers (Train split only)...")
        scaler = {} # Dictionary to hold one scaler per ticker
        
        # We must iterate tickers to fit scalers individually
        for t in tickers:
            t_scaler = TorchStandardScaler()
            # Fit only on this ticker's training data
            train_subset = df[(df["ticker"] == t) & (df["split"] == 0)][features]
            if not train_subset.empty:
                t_scaler.fit(train_subset)
            scaler[t] = t_scaler
    
    # Apply transform per ticker
    scaled_dfs = []
    for t in tickers:
        sub_df = df[df["ticker"] == t].copy()
        if t in scaler:
            # fillna(0) handles cases where a feature might be NaN for a specific day
            sub_df[features] = scaler[t].transform(sub_df[features].fillna(0.0))
        scaled_dfs.append(sub_df)
    
    df = pd.concat(scaled_dfs).sort_values(["ticker", "date"])
    # ---------------------------------------

    train_X, train_Y, train_T = [], [], []
    val_X, val_Y, val_T = [], [], []

    # Windowing Logic
    for t in tickers:
        g = df[df["ticker"] == t].reset_index(drop=True)
        if len(g) < window + max(horizons): continue

        X_arr = g[features].values.astype(np.float32)
        y_arr = g[target_col].values.astype(np.float32)
        split_arr = g["split"].values
        
        valid_indices = np.arange(window, len(g) - max(horizons))
        if len(valid_indices) == 0: continue

        X_windows = np.lib.stride_tricks.sliding_window_view(X_arr, window, axis=0)
        X_windows = X_windows[valid_indices - window] 
        
        y_targets = np.stack([y_arr[valid_indices + h - 1] for h in horizons], axis=1)
        t_ids = np.full(len(valid_indices), ticker_to_id[t], dtype=np.int64)
        
        target_indices = valid_indices + horizons[0] - 1
        is_val = split_arr[target_indices] == 1
        is_embargo = split_arr[target_indices] == -1  # Exclude embargo samples
        
        # Filter out embargo samples from both train and val
        is_train = (~is_val) & (~is_embargo)
        
        if np.sum(is_train) > 0:
            train_X.append(X_windows[is_train])
            train_Y.append(y_targets[is_train])
            train_T.append(t_ids[is_train])
        if np.sum(is_val) > 0:
            val_X.append(X_windows[is_val])
            val_Y.append(y_targets[is_val])
            val_T.append(t_ids[is_val])

    def to_tensor_dataset(X_list, Y_list, T_list):
        if not X_list: return None
        X = torch.tensor(np.concatenate(X_list))
        if X.ndim == 3 and X.shape[-1] == window: X = X.permute(0, 2, 1)
        Y = torch.tensor(np.concatenate(Y_list))
        T = torch.tensor(np.concatenate(T_list), dtype=torch.long)
        return TensorDataset(X, T, Y)

    train_ds = to_tensor_dataset(train_X, train_Y, train_T)
    val_ds = to_tensor_dataset(val_X, val_Y, val_T)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True) if train_ds else None
    val_loader = DataLoader(val_ds, batch_size=batch_size) if val_ds else None

    # (train_volnetx matches previous code, no changes needed there)
    
    return ticker_to_id, train_loader, val_loader, train_ds, val_ds, scaler


def train_volnetx(config: VolNetXConfig, train_loader, val_loader, n_tickers: int):
    # 1. Initialize Model
    model = VolNetX(
        n_tickers=n_tickers,
        window=config.window,
        input_size=config.input_size,
        horizons=config.horizons,
        hidden_dim=config.hidden_dim,
        emb_dim=config.emb_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        feat_dropout=config.feat_dropout,
        use_transformer=config.use_transformer,
        use_ticker_embedding=config.use_ticker_embedding,
        use_feature_attention=config.use_feature_attention,
        separate_heads=config.separate_heads,
        use_layernorm=config.use_layernorm
    ).to(config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    
    scheduler = None
    if getattr(config, "cosine_schedule", False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    # 2. Setup Loss Function
    loss_type = getattr(config, "loss_type", "mse").lower()
    
    # We use 'none' reduction to handle per-horizon weighting manually
    if loss_type == "huber":
        criterion = nn.HuberLoss(delta=1.0, reduction='none')
    else:
        # Standard MSE (placeholder, calculation handled manually for weighting)
        criterion = nn.MSELoss(reduction='none')

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    # 3. Training Loop
    for epoch in range(config.epochs):
        model.train()
        train_losses = []
        for x, tidx, y in train_loader:
            x, tidx, y = x.to(config.device), tidx.to(config.device), y.to(config.device)
            optimizer.zero_grad()
            preds = model(tidx, x)
            loss = 0
            for i, w in enumerate(config.loss_horizon_weights):
                if loss_type == "huber":
                    # Huber: Handles outliers natively
                    raw_loss = criterion(preds[:, i], y[:, i])
                    loss += w * raw_loss.mean()
                else:
                    # MSE: Apply manual High-Vol Penalty
                    raw_loss = (preds[:, i] - y[:, i]) ** 2
                    weight = 1.0 + (y[:, i] > -1.0).float() * 1.5 
                    loss += w * (weight * raw_loss).mean()
            
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            optimizer.step()
            train_losses.append(loss.item())

        if scheduler: scheduler.step()
        
        # 4. Validation Loop
        model.eval()
        val_losses = []
        with torch.no_grad():
            if val_loader:
                for x, tidx, y in val_loader:
                    x, tidx, y = x.to(config.device), tidx.to(config.device), y.to(config.device)
                    preds = model(tidx, x)
                    batch_loss = 0
                    for i, w in enumerate(config.loss_horizon_weights):
                        if loss_type == "huber":
                            raw_loss = criterion(preds[:, i], y[:, i])
                        else:
                            # Validation usually uses standard MSE to track performance
                            # (No extra weighting) to match previous behavior
                            raw_loss = (preds[:, i] - y[:, i]) ** 2
                        
                        batch_loss += w * raw_loss.mean()
                    val_losses.append(batch_loss.item())
        
        val_loss = np.mean(val_losses) if val_losses else 0.0
        print(f"Epoch {epoch+1}/{config.epochs} | Train: {np.mean(train_losses):.5f} | Val: {val_loss:.5f}")
        
        # 5. Snapshot Logic
        if config.early_stop:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(f"⏹️ Early stopping triggered at Epoch {epoch+1}.")
                    break
    
    if best_model_state is not None:
        print(f"🔙 Restoring best model weights (Loss: {best_val_loss:.5f})...")
        model.load_state_dict(best_model_state)
    
    return model
