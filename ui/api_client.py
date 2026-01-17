# mikebot/ui/api_client.py
"""
Shared API client for Mikebot Studio UI.

Features:
    - BASE_URL from environment variable MIKEBOT_API_URL
    - Robust retries (3 attempts, exponential backoff: 100ms → 300ms → 900ms)
    - Persistent error logging to mikebot/ui/api_errors.log (no rotation)
    - All functions return a dict
        - On success: parsed JSON dict
        - On failure: { "error": "message" }

This module is intentionally simple, explicit, and sweep‑friendly.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict

import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL: str = os.getenv("MIKEBOT_API_URL", "http://127.0.0.1:8000")

TIMEOUT_SECONDS: float = 0.75

# Retry delays: 100ms → 300ms → 900ms
RETRY_DELAYS = [0.1, 0.3, 0.9]

# Log file (no rotation)
LOG_PATH = Path(__file__).resolve().parent / "api_errors.log"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log_error(msg: str) -> None:
    """
    Append an error message to api_errors.log.
    No rotation, no truncation — user deletes manually when desired.
    """
    try:
        LOG_PATH.write_text(
            LOG_PATH.read_text(encoding="utf-8") + msg + "\n",
            encoding="utf-8"
        )
    except FileNotFoundError:
        LOG_PATH.write_text(msg + "\n", encoding="utf-8")
    except Exception:
        # As a last resort, swallow logging errors — UI must never crash.
        pass


# ---------------------------------------------------------------------------
# Core request helper
# ---------------------------------------------------------------------------

def _request_json(endpoint: str) -> Dict[str, Any]:
    """
    Perform a GET request with robust retries and consistent error handling.
    Always returns a dict.
    """
    url = f"{BASE_URL}{endpoint}"

    for attempt, delay in enumerate(RETRY_DELAYS, start=1):
        try:
            resp = requests.get(url, timeout=TIMEOUT_SECONDS)
            if resp.status_code != 200:
                msg = f"HTTP {resp.status_code} for {url}"
                _log_error(msg)
                return {"error": msg}

            try:
                return resp.json()
            except json.JSONDecodeError:
                msg = f"Invalid JSON from {url}"
                _log_error(msg)
                return {"error": msg}

        except Exception as exc:
            msg = f"Attempt {attempt} failed for {url}: {exc}"
            _log_error(msg)
            time.sleep(delay)

    # Final failure after all retries
    final_msg = f"All retries failed for {url}"
    _log_error(final_msg)
    return {"error": final_msg}


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------

def get_system_state() -> Dict[str, Any]:
    return _request_json("/system/state")


def get_strategy_state() -> Dict[str, Any]:
    return _request_json("/strategy/state")


def get_experience_state() -> Dict[str, Any]:
    return _request_json("/experience/state")


def get_models_state() -> Dict[str, Any]:
    return _request_json("/models/state")


def get_health_state() -> Dict[str, Any]:
    return _request_json("/health")


def get_training_state() -> Dict[str, Any]:
    return _request_json("/training/state")