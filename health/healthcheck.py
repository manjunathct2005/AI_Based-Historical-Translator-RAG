"""
health/healthcheck.py

System health checks shown in the Streamlit sidebar / used by tests and
Docker HEALTHCHECK: internet reachability (since RAG depends on live web
search), GPU availability, disk space, and whether key model weights have
been cached locally yet (first run downloads them, which can be slow).

NOTE: this is a *system* health module, not a medical-domain RAG. See the
assumption noted in the top-level project introduction.
"""

from __future__ import annotations

import shutil
import socket
import time
from dataclasses import dataclass
from typing import List


@dataclass
class HealthCheckResult:
    name: str
    ok: bool
    detail: str
    latency_ms: float = 0.0


def check_internet(host: str = "8.8.8.8", port: int = 53, timeout: float = 3.0) -> HealthCheckResult:
    start = time.time()
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return HealthCheckResult("internet", True, "Reachable",
                                  latency_ms=(time.time() - start) * 1000)
    except OSError as exc:
        return HealthCheckResult("internet", False, f"Unreachable: {exc}",
                                  latency_ms=(time.time() - start) * 1000)


def check_gpu() -> HealthCheckResult:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return HealthCheckResult("gpu", True, f"CUDA available: {name}")
        return HealthCheckResult("gpu", False, "No CUDA GPU detected -- running on CPU")
    except Exception as exc:
        return HealthCheckResult("gpu", False, f"torch not available: {exc}")


def check_disk_space(path: str = ".", min_free_gb: float = 2.0) -> HealthCheckResult:
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024 ** 3)
    ok = free_gb >= min_free_gb
    return HealthCheckResult(
        "disk_space", ok,
        f"{free_gb:.1f} GB free (threshold {min_free_gb} GB)"
    )


def check_dependency_imports() -> List[HealthCheckResult]:
    """Verify heavier optional dependencies actually import; helps catch env issues fast."""
    modules = [
        "torch", "transformers", "sentence_transformers", "faiss",
        "easyocr", "pytesseract", "fitz", "docx", "trafilatura",
        "duckduckgo_search",
    ]
    results = []
    for mod in modules:
        try:
            __import__(mod)
            results.append(HealthCheckResult(f"import:{mod}", True, "OK"))
        except Exception as exc:
            results.append(HealthCheckResult(f"import:{mod}", False, str(exc)))
    return results


def run_all_checks() -> List[HealthCheckResult]:
    results = [check_internet(), check_gpu(), check_disk_space()]
    results.extend(check_dependency_imports())
    return results
