# volsense_core/utils/checkpoint_utils.py
# ============================================================
# 🧩 VolSense Checkpoint Utility (Final)
# Auto-detects architecture, extracts constructor args,
# and builds reconstructible meta and bundle files.
# ============================================================

"""
Checkpoint utilities for VolSense models.

This module provides helpers to build metadata, construct portable
bundles, and save/load model artifacts in multiple formats.

Public functions
----------------
build_meta_from_model(model, cfg=None, ticker_to_id=None, features=None)
    Build a reconstructible metadata dictionary from a trained model.

build_bundle(model, meta, cfg=None)
    Create a portable bundle dict containing state_dict + meta.

save_checkpoint(model, cfg=None, version="model", save_dir="models", ...)
    Persist the model in three VolSense-supported formats.

load_meta(model_version, checkpoints_dir="models")
    Quick inspector to load a model's meta.json.
"""

import os
import json
import pickle
import torch
import numpy as np
import inspect
from typing import Any, Dict

# ============================================================
# 🛠️ Helper: JSON Sanitizer
# ============================================================
def _to_json_safe(obj):
    """Recursively convert Tensors/Arrays to JSON-friendly types (lists/floats)."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(x) for x in obj]
    return obj

# ============================================================
# 🔹 Meta Builder (auto-detects architecture)
# ============================================================

def build_meta_from_model(
    model: Any, cfg: Any = None, ticker_to_id=None, features=None
) -> Dict:
    """
    Build a standardized, reconstructible metadata dictionary for a VolSense model.

    The returned metadata is suitable for saving as "<stem>.meta.json" and
    contains architecture name/module, selected constructor arguments and
    high-level training/configuration fields.

    :param model: Trained model object (e.g., GlobalVolForecaster or BaseLSTM).
    :type model: Any
    :param cfg: Training/config object used to produce the model (optional).
    :type cfg: Any or None
    :param ticker_to_id: Optional mapping of ticker -> embedding id (optional).
    :type ticker_to_id: dict or None
    :param features: Explicit list of features used during training (optional).
    :type features: list[str] or None
    :returns: Metadata dictionary containing keys `arch`, `module_path`, `features`,
              `extra_features`, `horizons`, `ticker_to_id`, and `arch_params`.
    :rtype: dict
    :raises TypeError: If model lacks an inspectable constructor (rare).
    """
    name = model.__class__.__name__
    module_path = model.__module__

    # --- Explicit feature list construction ---
    # Both VolNetX and GlobalVolForecaster use ["return"] + extra_features
    # We reconstruct this explicitly to avoid ambiguity during loading
    explicit_features = features
    if explicit_features is None and cfg is not None:
        extra_feats = getattr(cfg, "extra_features", None) or []
        explicit_features = ["return"] + extra_feats if extra_feats else ["return"]

    meta = {
        "arch": name,
        "module_path": module_path,
        "features": explicit_features or [],
        "extra_features": getattr(cfg, "extra_features", []),
        "horizons": getattr(cfg, "horizons", []),
        "ticker_to_id": ticker_to_id or {},
    }

    # Inspect constructor and extract matching attributes
    sig = inspect.signature(model.__init__)
    arch_params = {}
    
    # Generic attribute extraction
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        if hasattr(model, param_name):
            val = getattr(model, param_name)
            if isinstance(val, (int, float, bool, str, list, tuple, dict, type(None))):
                arch_params[param_name] = val

    # --- Model-specific augmentations ---
    if name.lower().startswith("glob"):  # GlobalVolForecaster
        extra_keys = [
            "emb_dim", "hidden_dim", "num_layers", "dropout",
            "separate_heads", "use_layernorm",
        ]
        for k in extra_keys:
            if hasattr(model, k):
                arch_params[k] = getattr(model, k)

    elif name.lower().startswith("volnet"): # VolNetX (NEW)
        extra_keys = [
            "input_size", "hidden_dim", "emb_dim", "num_layers",
            "dropout", "use_transformer", "use_ticker_embedding",
            "use_feature_attention", "separate_heads", "use_layernorm"
        ]
        for k in extra_keys:
            # Try getting from model first, fallback to cfg
            if hasattr(model, k):
                arch_params[k] = getattr(model, k)
            elif cfg is not None and hasattr(cfg, k):
                arch_params[k] = getattr(cfg, k)

    elif name.lower().startswith("base"):  # BaseLSTM
        extra_keys = [
            "input_dim", "hidden_dim", "num_layers", "dropout",
            "n_horizons", "use_layernorm", "use_attention",
            "feat_dropout_p", "residual_head", "output_activation",
        ]
        for k in extra_keys:
            if hasattr(model, k):
                arch_params[k] = getattr(model, k)

    # Remove cfg-only fields not accepted by constructors
    # Note: 'window' is intentionally stored at meta root (not arch_params)
    # because it's extracted directly from model attributes via generic
    # attribute extraction (lines 96-103). Both VolNetX and GlobalVolForecaster
    # store self.window, which gets captured automatically.
    for bad_key in ["horizons", "window", "stride", "val_start", "target_col"]:
        arch_params.pop(bad_key, None)

    meta["arch_params"] = arch_params

    # --- Minimal JSON-safe config extraction ---
    target_col = None
    if cfg is not None:
        if isinstance(cfg, dict):
            target_col = cfg.get("target_col") or (cfg.get("config") or {}).get("target_col")
        else:
            target_col = getattr(cfg, "target_col", None)
            cfg_config = getattr(cfg, "config", None)
            if isinstance(cfg_config, dict):
                target_col = target_col or cfg_config.get("target_col")
            else:
                target_col = target_col or getattr(cfg_config, "target_col", None)

    meta["config"] = {"target_col": target_col} if target_col is not None else {}

    return meta


# ============================================================
# 🔹 Bundle Builder & Saver (Unchanged but included for context)
# ============================================================

def build_bundle(model: Any, meta: Dict, cfg: Any = None) -> Dict:
    """
    Construct a portable bundle for pickle-based saving.

    The bundle contains:
      - state_dict: model.state_dict() (CPU tensors)
      - meta: metadata dict produced by :func:`build_meta_from_model`
      - arch, module_path, arch_params, ticker_to_id, features, extra_features, horizons
      - config (optional): cfg as a dict when provided

    :param model: Trained model object with `.state_dict()` method.
    :type model: Any
    :param meta: Metadata dictionary (see :func:`build_meta_from_model`).
    :type meta: dict
    :param cfg: Training configuration object (optional).
    :type cfg: Any or None
    :returns: Serializable bundle suitable for writing with pickle.
    :rtype: dict
    """
    safe_state = model.state_dict()
    bundle = {
        "state_dict": safe_state,
        "meta": meta,
        "arch": meta.get("arch"),
        "module_path": meta.get("module_path"),
        "arch_params": meta.get("arch_params", {}),
        "ticker_to_id": meta.get("ticker_to_id", {}),
        "features": meta.get("features", []),
        "extra_features": meta.get("extra_features", []),
        "horizons": meta.get("horizons", []),
    }
    if cfg is not None:
        bundle["config"] = meta.get("config", {})
    return bundle

def save_checkpoint(
    model: Any, cfg: Any = None, version: str = "model",
    save_dir: str = "models", ticker_to_id=None, features=None, scalers=None
):
    """
    Save model in all supported VolSense formats.

    The function writes three artifact styles for maximal compatibility:
      1. <save_dir>/<version>.full.pkl    (pickled model object)
      2. <save_dir>/<version>_bundle.pkl  (pickle'd bundle dict)
      3. <save_dir>/<version>.pth + .meta.json (portable torch state + meta)

    :param model: Trained model object to serialize.
    :type model: Any
    :param cfg: Training config object (optional).
    :type cfg: Any or None
    :param version: Version tag or stem used to name files.
    :type version: str
    :param save_dir: Directory to write artifacts to.
    :type save_dir: str
    :param ticker_to_id: Optional ticker->id mapping to include in metadata.
    :type ticker_to_id: dict or None
    :param features: Feature list to embed into metadata.
    :type features: list[str] or None
    :param scalers: Optional dict of feature scalers to persist in the metadata.
    :type scalers: dict[str, TorchStandardScaler] or None
    :returns: The meta dictionary that was written to <version>.meta.json.
    :rtype: dict
    :raises OSError: If the save directory cannot be created or files cannot be written.
    """
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.join(save_dir, version)

    meta = build_meta_from_model(model, cfg, ticker_to_id=ticker_to_id, features=features)

    # Persist scaler states into meta
    if scalers is not None:
        meta["scalers"] = {}
        for name, sc in scalers.items():
            try:
                raw_state = sc.state_dict()
            except Exception:
                raw_state = sc.__dict__
            # 🛡️ CRITICAL: Convert Tensors to Lists for JSON safety
            meta["scalers"][name] = _to_json_safe(raw_state)
    
    bundle = build_bundle(model, meta, cfg)

    # Save artifacts
    with open(base + ".full.pkl", "wb") as f: pickle.dump(model, f)
    with open(base + "_bundle.pkl", "wb") as f: pickle.dump(bundle, f)
    torch.save(model.state_dict(), base + ".pth")
    with open(base + ".meta.json", "w") as f: json.dump(meta, f, indent=2)

    print(f"💾 Saved artifacts for {version}")
    return meta

def load_meta(model_version: str, checkpoints_dir: str = "models") -> Dict:
    """
    Load metadata only for quick inspection.

    :param model_version: Version stem used to locate "<model_version>.meta.json".
    :type model_version: str
    :param checkpoints_dir: Directory where model artifacts are stored.
    :type checkpoints_dir: str
    :returns: Parsed metadata dictionary.
    :rtype: dict
    :raises FileNotFoundError: If the referenced meta.json file does not exist.
    """
    meta_path = os.path.join(checkpoints_dir, model_version + ".meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found for {model_version}")
    with open(meta_path, "r") as f:
        return json.load(f)
