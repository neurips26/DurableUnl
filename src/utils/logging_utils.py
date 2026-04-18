"""
Logging utilities with timestamps for DurableUn.
All loggers include date + time in every message.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Timestamp helpers
# ─────────────────────────────────────────────────────────────────────────────

def now_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Return current datetime as a formatted string."""
    return datetime.now().strftime(fmt)


def file_ts() -> str:
    """Timestamp safe for use in file/dir names: 2026-03-26_14-30-00"""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# ─────────────────────────────────────────────────────────────────────────────
# Logger factory
# ─────────────────────────────────────────────────────────────────────────────

_CONFIGURED: set = set()

def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Return a named logger that writes [YYYY-MM-DD HH:MM:SS] prefixed messages
    to both stdout and optionally a file.
    """
    logger = logging.getLogger(name)
    if name in _CONFIGURED:
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    _CONFIGURED.add(name)
    return logger


def setup_root_logger(log_dir: str = "logs") -> str:
    """
    Configure root logger + file handler. Returns path to log file.
    Called once at the start of each experiment script.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"phase0_{file_ts()}.log")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        root.addHandler(ch)

    # File
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    return log_path


# ─────────────────────────────────────────────────────────────────────────────
# ResultLogger — optional W&B / CSV event logging
# ─────────────────────────────────────────────────────────────────────────────

class ResultLogger:
    """
    Lightweight event logger that writes rows to a CSV.
    Pass to BaseUnlearner for per-step metric tracking.
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._writer = None
        self._file = None
        self._headers_written = False

    def log(self, row: dict):
        import csv
        row["timestamp"] = now_str()
        os.makedirs(os.path.dirname(os.path.abspath(self.csv_path)), exist_ok=True)
        write_header = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def close(self):
        pass
