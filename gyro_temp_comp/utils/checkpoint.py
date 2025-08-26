from typing import Any, Dict, Optional


def save_checkpoint(
    path: str,
    *,
    model_state: Dict[str, Any],
    optimizer_g_state: Optional[Dict[str, Any]] = None,
    optimizer_d_state: Optional[Dict[str, Any]] = None,
    epoch: Optional[int] = None,
    best_metric: Optional[float] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    import torch  # local import to avoid hard dependency during non-training imports

    payload: Dict[str, Any] = {
        "model_state": model_state,
        "optimizer_g_state": optimizer_g_state,
        "optimizer_d_state": optimizer_d_state,
        "epoch": epoch,
        "best_metric": best_metric,
        "meta": meta or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: str) -> Dict[str, Any]:
    import torch  # local import

    return torch.load(path, map_location="cpu")