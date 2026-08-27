"""Unified run logging -- logging configuration (stderr + optional run.log file).

Convention: stdout carries only the command's **results** (acceptance
numbers/tables/paths); process events go through the ``atap`` logger →
stderr (visible in the terminal) + ``runs/<name>/run.log`` (persisted,
attached automatically by runtime.run_config). ``setup_logging`` is
idempotent (no handler accumulation across multiple runs in one process);
``attach_run_log`` attaches the file handler in replace mode.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOGGER_NAME = "atap"
_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a child logger under the ``atap`` namespace (e.g. atap.runtime / atap.cli)."""
    return logging.getLogger(f"{_LOGGER_NAME}.{name}" if name else _LOGGER_NAME)


def setup_logging(*, verbose: bool = False) -> logging.Logger:
    """Initialize the ``atap`` logger (stderr handler; verbose enables DEBUG)."""
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    return logger


def attach_run_log(path: str | Path) -> None:
    """Attach a run.log file handler to the ``atap`` logger (replacing the
    old file handler).

    Called by runtime.run_config for each run: compare's multiple runs swap
    files in sequence, so each run.log only contains logs from its own run
    onward.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.level in (logging.NOTSET, logging.WARNING, logging.ERROR,
                        logging.CRITICAL):
        logger.setLevel(logging.INFO)   # library-style calls (without setup_logging) still get persisted
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)
            h.close()
    fh = logging.FileHandler(p, encoding="utf-8")
    fh.setFormatter(logging.Formatter(_FMT, datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
