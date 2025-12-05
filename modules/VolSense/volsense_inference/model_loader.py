# ============================================================
# volsense_inference/model_loader.py
# Unified VolSense Loader
# Supports:
#   - *.full.pkl        → legacy pickled model
#   - *_bundle.pkl      → combined dict
#   - .meta.json + .pth → modern portable
# ============================================================

"""
Model loader utilities for VolSense inference.

This module provides a single entrypoint `load_model()` which locates
and loads model artifacts saved in one of three supported formats:

- legacy ".full.pkl" (pickled model or bundle)
- combined "<stem>_bundle.pkl" (dict with state + meta)
- portable "<stem>.meta.json" + "<stem>.pth" (state_dict + metadata)

All loaders return a tuple:
    (model, meta, scalers, ticker_to_id, features)

The module focuses on safe CPU-device loading and conservative defaults
so artifacts saved on different environments (Colab, local) can be loaded
reliably.

Sphinx-style documentation is provided on public helpers.
"""

import os
import json
import pickle
import importlib
import torch
import numpy as np
from volsense_core.utils.scalers import TorchStandardScaler


# ------------------------------------------------------------
# 🔹 Resolve checkpoint path
# ------------------------------------------------------------
def _resolve_ckpt_path(model_version: str, checkpoints_dir: str) -> str:
    """
    Resolve the absolute base path for a model checkpoint stem.

    The function searches for `checkpoints_dir` in the current working
    directory and the parent directory, preferring the parent if both
    exist. The returned string is the absolute path joined with the
    provided `model_version` (no suffix).

    :param model_version: Version tag or stem used for checkpoint filenames
                          (e.g. "v507").
    :type model_version: str
    :param checkpoints_dir: Directory name (relative to cwd) to search for
                            model artifacts (e.g. "models").
    :type checkpoints_dir: str
    :return: Absolute path to the checkpoint stem (without suffix).
    :rtype: str
    :raises FileNotFoundError: If no matching checkpoints directory is found.
    """
    cwd = os.getcwd()
    local_ckpt = os.path.abspath(os.path.join(cwd, checkpoints_dir))
    parent_ckpt = os.path.abspath(os.path.join(cwd, "..", checkpoints_dir))

    # Prefer the parent-level models dir if both exist
    if os.path.exists(parent_ckpt):
        ckpt_dir = parent_ckpt
    elif os.path.exists(local_ckpt):
        ckpt_dir = local_ckpt
    else:
        raise FileNotFoundError(
            f"Could not locate checkpoint directory: {checkpoints_dir}"
        )

    return os.path.join(ckpt_dir, model_version)


def _import_class(module_path: str, class_name: str):
    """
    Dynamically import and return a class given its module path and name.

    :param module_path: Full python module path (e.g.
                        "volsense_core.models.global_vol_forecaster").
    :type module_path: str
    :param class_name: Class name to import from the module.
    :type class_name: str
    :return: Imported class object.
    :rtype: type
    :raises (ImportError, AttributeError): If module or attribute cannot be found.
    """
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _reconstruct_scalers(meta_scalers: dict) -> dict:
    """
    Reconstruct TorchStandardScaler objects from metadata dictionary.
    
    :param meta_scalers: Dictionary of {ticker: scaler_state_dict} from meta.json.
    :return: Dictionary of {ticker: TorchStandardScaler}.
    """
    if not meta_scalers:
        return None
        
    scalers = {}
    for ticker, state in meta_scalers.items():
        sc = TorchStandardScaler()
        # Rehydrate state (converting lists back to tensors)
        if "mean_" in state and state["mean_"] is not None:
            sc.mean_ = torch.tensor(state["mean_"], dtype=torch.float32)
        if "std_" in state and state["std_"] is not None:
            sc.std_ = torch.tensor(state["std_"], dtype=torch.float32)
        if "feature_names_in_" in state:
            sc.feature_names_in_ = state["feature_names_in_"]
        scalers[ticker] = sc
    return scalers


# ------------------------------------------------------------
# 🔹 1️⃣ Legacy full.pkl loader (v109)
# ------------------------------------------------------------
def _load_full_pickle(path: str, device: str):
    """
    Load a legacy ".full.pkl" artifact.

    The ".full.pkl" file may contain either:
      - a bundle dict: {"model": <model>, "meta": {...}, "ticker_to_id": {...}, "features": [...]}
      - a direct pickled model object

    This loader extracts model metadata and ensures the model is moved to
    the requested device and set to eval().

    :param path: Path to the ".full.pkl" file.
    :type path: str
    :param device: Torch device specifier accepted by model.to() (e.g. "cpu" or "cuda").
    :type device: str
    :return: (model, meta, scalers, ticker_to_id, features)
    :rtype: tuple
    """
    with open(path, "rb") as f:
        bundle = pickle.load(f)

    # Case A: dict bundle
    if isinstance(bundle, dict):
        model = bundle["model"]
        meta = bundle.get("meta", {}) or {}
        ticker_to_id = bundle.get("ticker_to_id", {}) or {}
        features = bundle.get("features", []) or []
    else:
        # Case B: direct model object
        model = bundle
        meta = {}
        if hasattr(model, "window"):
            meta["window"] = int(getattr(model, "window"))
        if hasattr(model, "horizons"):
            meta["horizons"] = list(getattr(model, "horizons"))
        if hasattr(model, "config") and isinstance(model.config, dict):
            meta["config"] = {
                k: v
                for k, v in model.config.items()
                if isinstance(v, (int, float, bool, str, list))
            }
        ticker_to_id = getattr(model, "ticker_to_id", {}) or {}
        features = (
            getattr(model, "features", None)
            or getattr(model, "feature_names_", None)
            or []
        )

    # Fallbacks
    if "window" not in meta:
        meta["window"] = 40
    if "horizons" not in meta:
        meta["horizons"] = [1]

    # Enrich from sidecar meta.json (single source of truth)
    sidecar = path.replace(".full.pkl", ".meta.json")
    scalers = None
    if os.path.exists(sidecar):
        try:
            with open(sidecar, "r") as f:
                side_meta = json.load(f)
            meta.update(side_meta)
            features = side_meta.get("features", features)
            ticker_to_id = side_meta.get("ticker_to_id", ticker_to_id)
            if hasattr(model, "input_size"):
                model.input_size = len(features)
            
            # Reconstruct scalers if present
            if "scalers" in side_meta:
                scalers = _reconstruct_scalers(side_meta["scalers"])
                
        except Exception:
            pass

    model.to(device).eval()
    return model, meta, scalers, ticker_to_id, features


# ------------------------------------------------------------
# 🔹 2️⃣ Bundle pickle loader (combined dict)
# ------------------------------------------------------------
def _load_bundle_pickle(path: str, device: str):
    """
    Load model from portable (.meta.json + .pth) format.

    The meta.json should contain enough information to reconstruct the
    model architecture (module_path, arch, arch_params, features, ticker_to_id).
    The .pth is expected to be a torch.state_dict saved via torch.save().

    :param base: Checkpoint stem (path without suffix).
    :type base: str
    :param device: Device for torch.load map_location and model.to().
    :type device: str
    :return: (model, meta, scalers, ticker_to_id, features)
    :rtype: tuple
    :raises FileNotFoundError: if the meta or pth file is missing.
    """
    with open(path, "rb") as f:
        bundle = pickle.load(f)

    # --- Extract bundle metadata ---
    meta = bundle.get("meta", {})
    module_path = bundle.get(
        "module_path",
        meta.get("module_path", "volsense_core.models.global_vol_forecaster"),
    )
    arch = bundle.get("arch", meta.get("arch", "GlobalVolForecaster"))
    arch_params = dict(bundle.get("arch_params", meta.get("arch_params", {})))

    # --- Fallback defaults for safety ---
    ticker_to_id = bundle.get("ticker_to_id", meta.get("ticker_to_id", {}))
    n_tickers = len(ticker_to_id) if ticker_to_id else 1
    horizons = meta.get("horizons", [1])
    n_horizons = len(horizons) if isinstance(horizons, list) else int(horizons)
    features = bundle.get("features", meta.get("features", []))
    if not features:
        features = ["return"] + meta.get("extra_features", [])

    defaults = {
        "n_tickers": n_tickers,
        "window": meta.get("window", 40),
        "n_horizons": n_horizons,
        "emb_dim": 16,
        "hidden_dim": 160,
        "num_layers": 3,
        "dropout": 0.3,
        "input_size": len(features),
        "attention": True,
        "residual_head": True,
        "use_layernorm": True,
        "separate_heads": True,
        "feat_dropout_p": 0.1,
        "variational_dropout_p": 0.1,
    }
    
    # VolNetX specific defaults
    if "VolNetX" in arch:
        defaults["horizons"] = horizons # VolNetX expects list
        # Remove n_horizons if present to avoid confusion/errors if not expected
        if "n_horizons" in defaults:
             del defaults["n_horizons"]

    # Fill missing keys in arch_params from defaults
    for k, v in defaults.items():
        arch_params.setdefault(k, v)

    # --- Build model ---
    ModelClass = _import_class(module_path, arch)
    model = ModelClass(**arch_params)
    model.load_state_dict(bundle["state_dict"], strict=False)
    model.to(device).eval()

    # --- Final consistency fix ---
    model.input_size = len(features)
    
    # Reconstruct scalers
    scalers = _reconstruct_scalers(meta.get("scalers"))

    return model, meta, scalers, ticker_to_id, features


# ------------------------------------------------------------
# 🔹 3️⃣ Meta + PTH loader (modern portable)
# ------------------------------------------------------------
def _load_meta_pth(base: str, device: str):
    """
    Load model from portable (.meta.json + .pth) format.

    The meta.json should contain enough information to reconstruct the
    model architecture (module_path, arch, arch_params, features, ticker_to_id).
    The .pth is expected to be a torch.state_dict saved via torch.save().

    :param base: Checkpoint stem (path without suffix).
    :type base: str
    :param device: Device for torch.load map_location and model.to().
    :type device: str
    :return: (model, meta, scalers, ticker_to_id, features)
    :rtype: tuple
    :raises FileNotFoundError: if the meta or pth file is missing.
    """
    meta_path = base + ".meta.json"
    pth_path = base + ".pth"

    with open(meta_path, "r") as f:
        meta = json.load(f)

    module_path = meta.get("module_path", "volsense_core.models.global_vol_forecaster")
    arch = meta.get("arch", "GlobalVolForecaster")
    arch_params = dict(meta.get("arch_params", {}))

    # --- Fallback defaults for safety ---
    ticker_to_id = meta.get("ticker_to_id", {})
    n_tickers = len(ticker_to_id) if ticker_to_id else 1
    horizons = meta.get("horizons", [1])
    
    # Convert horizons to appropriate format based on model architecture:
    # - GlobalVolForecaster expects n_horizons (int) in its constructor
    # - VolNetX expects horizons (list) in its constructor
    # The loader converts list→int for GlobalVol, but VolNetX-specific
    # handling passes the list directly (see lines 339-343)
    n_horizons = len(horizons) if isinstance(horizons, list) else int(horizons)
    
    features = meta.get("features", [])
    if not features:
        features = ["return"] + meta.get("extra_features", [])

    defaults = {
        "n_tickers": n_tickers,
        "window": meta.get("window", 40),
        "n_horizons": n_horizons,
        "emb_dim": 16,
        "hidden_dim": 160,
        "num_layers": 3,
        "dropout": 0.3,
        "input_size": len(features),
        "attention": True,
        "residual_head": True,
        "use_layernorm": True,
        "separate_heads": True,
        "feat_dropout_p": 0.1,
        "variational_dropout_p": 0.1,
    }
    
    # VolNetX specific defaults
    if "VolNetX" in arch:
        defaults["horizons"] = horizons  # VolNetX expects list
        # Remove GlobalVolForecaster-specific params
        for param in ["n_horizons", "attention", "residual_head", "feat_dropout_p", "variational_dropout_p"]:
            defaults.pop(param, None)
        # Add VolNetX-specific params (matching its constructor)
        defaults.setdefault("feat_dropout", 0.1)
        defaults.setdefault("use_transformer", True)
        defaults.setdefault("use_ticker_embedding", True)
        defaults.setdefault("use_feature_attention", True)

    # Fill missing keys from defaults
    for k, v in defaults.items():
        arch_params.setdefault(k, v)

    # --- Build and load model ---
    ModelClass = _import_class(module_path, arch)
    model = ModelClass(**arch_params)
    state_dict = torch.load(pth_path, map_location=device)
    
    # --- Critical validation: check state_dict compatibility ---
    # For VolNetX and GlobalVol, the first layer should match input_size
    # This catches cases where meta.json features are wrong but .pth is correct
    if hasattr(model, "input_size") and "VolNetX" in arch:
        # Check if embedding layer or first LSTM input size matches
        if "lstm.weight_ih_l0" in state_dict:
            # LSTM input weight shape: [hidden*4, input_size + emb_dim]
            saved_input_dim = state_dict["lstm.weight_ih_l0"].shape[1]
            expected_input_dim = model.input_size
            if model.emb is not None:
                expected_input_dim += model.emb.embedding_dim
            
            if saved_input_dim != expected_input_dim:
                raise ValueError(
                    f"State dict dimension mismatch for VolNetX: "
                    f"saved weights expect {saved_input_dim} input dims, "
                    f"but model reconstructed with {expected_input_dim} dims "
                    f"(input_size={model.input_size}, features={features}). "
                    f"This indicates corrupted meta.json with wrong feature list."
                )
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    # --- Secondary validation: ensure dimensions match ---
    if hasattr(model, "input_size"):
        actual_input_size = getattr(model, "input_size")
        expected_input_size = len(features)
        if actual_input_size != expected_input_size:
            raise ValueError(
                f"Model input_size mismatch: model expects {actual_input_size} features, "
                f"but meta.json specifies {expected_input_size} features {features}. "
                f"This indicates a corrupted checkpoint or feature list reconstruction error."
            )
    
    if hasattr(model, "n_tickers"):
        actual_n_tickers = getattr(model, "n_tickers")
        if actual_n_tickers != n_tickers:
            raise ValueError(
                f"Model n_tickers mismatch: model has {actual_n_tickers} tickers, "
                f"but meta.json specifies {n_tickers} tickers. "
                f"This model cannot be used with the provided ticker_to_id mapping."
            )
    
    # --- Final consistency fix (safe after validation) ---
    model.input_size = len(features)
    
    # Reconstruct scalers
    scalers = _reconstruct_scalers(meta.get("scalers"))

    return model, meta, scalers, ticker_to_id, features


# ------------------------------------------------------------
# 🔹 Entrypoint
# ------------------------------------------------------------
def load_model(
    model_version: str, checkpoints_dir: str = "models", device: str = "cpu"
):
    """
    Universal VolSense model loader.

    This function attempts to locate and load artifacts for `model_version`
    from `checkpoints_dir`. Supported artifact layouts (priority order):

      1. <stem>.full.pkl
      2. <stem>_bundle.pkl
      3. <stem>.meta.json + <stem>.pth

    The returned tuple contains:
      (model, meta, scalers, ticker_to_id, features)

    :param model_version: Version identifier matching checkpoint stems (e.g. "global_vol_forecaster_multi_v507").
    :type model_version: str
    :param checkpoints_dir: Relative directory name containing model artifacts.
    :type checkpoints_dir: str
    :param device: Torch device to move the loaded model to (default "cpu").
    :type device: str
    :return: Tuple with loaded model and auxiliary metadata/assets.
    :rtype: tuple
    :raises FileNotFoundError: If no supported artifact layout is found for the version.
    :example:
        >>> model, meta, scalers, t2i, features = load_model("global_vol_forecaster_multi_v507", "models", device="cpu")
    """
    base = _resolve_ckpt_path(model_version, checkpoints_dir)

    # meta.json + pth (portable)
    meta_path = base + ".meta.json"
    pth_path = base + ".pth"
    if os.path.exists(meta_path) and os.path.exists(pth_path):
        return _load_meta_pth(base, device)

        # bundle.pkl (combined)
    bundle_path = base + "_bundle.pkl"
    if os.path.exists(bundle_path):
        return _load_bundle_pickle(bundle_path, device)

    # full.pkl (legacy)
    full_path = base + ".full.pkl"
    if os.path.exists(full_path):
        return _load_full_pickle(full_path, device)

    raise FileNotFoundError(
        f"No valid checkpoint found for {model_version} in {checkpoints_dir}"
    )
