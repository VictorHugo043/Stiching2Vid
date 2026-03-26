"""Runtime helpers for optional Method B dependencies."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence


class MethodBBackendError(RuntimeError):
    """Error carrying structured diagnostics for optional Method B backends."""

    def __init__(self, message: str, diagnostics: Optional[Dict[str, object]] = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def probe_method_b_dependencies() -> Dict[str, Dict[str, object]]:
    """Probe optional Method B modules without importing them eagerly."""

    out: Dict[str, Dict[str, object]] = {}
    for name in ("torch", "kornia", "lightglue"):
        spec = importlib.util.find_spec(name)
        out[name] = {
            "available": bool(spec),
            "origin": getattr(spec, "origin", None) if spec else None,
        }
    return out


def resolve_method_b_device(
    requested_device: Optional[str] = None,
    force_cpu: bool = False,
) -> Dict[str, object]:
    """Resolve runtime device for Method B backends."""

    diagnostics = {
        "requested_device": requested_device or "auto",
        "force_cpu": bool(force_cpu),
        "resolved_device": None,
        "resolution_reason": None,
        "torch_available": False,
        "cuda_available": False,
        "mps_built": False,
        "mps_available": False,
    }
    deps = probe_method_b_dependencies()
    diagnostics["torch_available"] = bool(deps["torch"]["available"])

    if force_cpu:
        diagnostics["resolved_device"] = "cpu"
        diagnostics["resolution_reason"] = "force_cpu"
        return diagnostics

    if not deps["torch"]["available"]:
        diagnostics["resolution_reason"] = "torch_missing"
        return diagnostics

    import torch  # type: ignore

    cuda_available = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    mps_built = bool(mps_backend.is_built()) if mps_backend is not None else False
    mps_available = bool(mps_backend.is_available()) if mps_backend is not None else False
    diagnostics["cuda_available"] = cuda_available
    diagnostics["mps_built"] = mps_built
    diagnostics["mps_available"] = mps_available

    requested = (requested_device or "auto").strip().lower()
    if requested == "auto":
        if cuda_available:
            diagnostics["resolved_device"] = "cuda"
            diagnostics["resolution_reason"] = "auto_cuda"
        elif mps_available:
            diagnostics["resolved_device"] = "mps"
            diagnostics["resolution_reason"] = "auto_mps"
        else:
            diagnostics["resolved_device"] = "cpu"
            diagnostics["resolution_reason"] = "auto_cpu"
        return diagnostics

    if requested.startswith("cuda") and not cuda_available:
        diagnostics["resolved_device"] = "cpu"
        diagnostics["resolution_reason"] = "cuda_unavailable_fallback_cpu"
        return diagnostics

    if requested == "mps" and not mps_available:
        diagnostics["resolved_device"] = "cpu"
        diagnostics["resolution_reason"] = "mps_unavailable_fallback_cpu"
        return diagnostics

    diagnostics["resolved_device"] = requested_device or requested
    diagnostics["resolution_reason"] = "requested"
    return diagnostics


def ensure_method_b_dependencies(
    backend_name: str,
    required_modules: Iterable[str],
) -> Dict[str, Dict[str, object]]:
    """Fail fast if required optional modules are unavailable."""

    diagnostics = probe_method_b_dependencies()
    missing = [name for name in required_modules if not diagnostics.get(name, {}).get("available")]
    if missing:
        raise MethodBBackendError(
            (
                f"Backend '{backend_name}' requires optional dependencies: "
                f"{', '.join(missing)}"
            ),
            diagnostics={
                "backend_name": backend_name,
                "missing_modules": missing,
                "dependency_status": diagnostics,
            },
        )
    return diagnostics


def resolve_method_b_weights(
    *,
    backend_name: str,
    weights_dir: Optional[str],
    candidate_names: Sequence[str],
) -> Dict[str, object]:
    """Resolve optional explicit weights path for a Method B backend."""

    diagnostics = {
        "backend_name": backend_name,
        "weights_dir": weights_dir,
        "weights_source": "package_default",
        "weights_path": None,
        "candidate_names": list(candidate_names),
    }
    if not weights_dir:
        return diagnostics

    path = Path(weights_dir).expanduser()
    if not path.exists():
        raise MethodBBackendError(
            f"weights_dir does not exist for backend '{backend_name}': {weights_dir}",
            diagnostics=diagnostics,
        )

    if path.is_file():
        diagnostics["weights_source"] = "explicit_file"
        diagnostics["weights_path"] = str(path)
        return diagnostics

    matches = []
    for candidate in candidate_names:
        matches.extend(path.rglob(candidate))
    if not matches:
        raise MethodBBackendError(
            (
                f"No supported weights file found for backend '{backend_name}' under "
                f"{weights_dir}"
            ),
            diagnostics=diagnostics,
        )

    selected = sorted(matches)[0]
    diagnostics["weights_source"] = "weights_dir"
    diagnostics["weights_path"] = str(selected)
    return diagnostics


def maybe_load_state_dict(module, weights_path: Optional[str]) -> Dict[str, object]:
    """Optionally load an explicit state dict into a torch module."""

    diagnostics = {
        "weights_loaded": False,
        "missing_keys": [],
        "unexpected_keys": [],
    }
    if not weights_path:
        return diagnostics

    import torch  # type: ignore

    checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if isinstance(checkpoint.get("state_dict"), dict):
            checkpoint = checkpoint["state_dict"]
        elif isinstance(checkpoint.get("model"), dict):
            checkpoint = checkpoint["model"]
    if not isinstance(checkpoint, dict):
        raise MethodBBackendError(
            f"Unsupported checkpoint structure in weights file: {weights_path}",
            diagnostics={"weights_path": weights_path},
        )

    missing_keys, unexpected_keys = module.load_state_dict(checkpoint, strict=False)
    diagnostics["weights_loaded"] = True
    diagnostics["missing_keys"] = list(missing_keys)
    diagnostics["unexpected_keys"] = list(unexpected_keys)
    return diagnostics
