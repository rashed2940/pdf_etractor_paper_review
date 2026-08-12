"""Structured logging setup (AGENTS.md §26).

Single import side-effect: `import config.logging` configures the root
logger with a consistent format and level. Safe to call multiple times —
re-installation is a no-op.

Never logs API keys or secrets; uses a small filter that scrubs known
sensitive keys before they reach any handler.
"""
from __future__ import annotations

import logging
import logging.config
import re
from typing import Any

from config.settings import get_settings

# Patterns considered sensitive — replaced with *** in log records.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*\S+"),
)


class _SecretScrubber(logging.Filter):
    """Filter that masks obvious secret-looking substrings in log messages.

    Replaces patterns like ``api_key=abc123`` with ``api_key=***`` so the
    key name is preserved (useful in debugging) but the secret value is
    masked.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        msg = record.getMessage()
        for pat in _SECRET_PATTERNS:
            msg = pat.sub(lambda m: re.sub(r"=\S+", "=***", m.group(0)), msg)
        record.msg = msg
        record.args = ()
        return True


_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    """Configure root logger. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    log_level = (level or settings.log_level).upper()

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {
                "scrub_secrets": {"()": "config.logging._SecretScrubber"},
            },
            "formatters": {
                "standard": {
                    "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "filters": ["scrub_secrets"],
                    "level": log_level,
                },
            },
            "loggers": {
                # Quiet down noisy third-party loggers by default
                "httpx": {"level": "WARNING"},
                "chromadb": {"level": "WARNING"},
                # posthog is invoked by chromadb 0.5+ with an API signature
                # mismatch ("capture() takes 1 positional argument but 3
                # were given"). It's pure noise — silence it.
                "chromadb.telemetry.product.posthog": {"level": "CRITICAL"},
                "urllib3": {"level": "WARNING"},
            },
            "root": {
                "handlers": ["console"],
                "level": log_level,
            },
        }
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger, configuring logging on first call."""
    setup_logging()
    return logging.getLogger(name)


__all__ = ["setup_logging", "get_logger"]
