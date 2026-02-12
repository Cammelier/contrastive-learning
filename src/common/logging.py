from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

from rich.logging import RichHandler


def setup_logger(
    *,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s: %(message)s",
    date_fmt: str = "%H:%M:%S",
    console: bool = True,
    log_file: Optional[str] = "resources/logs",
    file_level: Optional[int] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 3,
) -> logging.Logger:
    """
    Configure the root logger:

      - level:       root logger level
      - fmt:         format string for all handlers
      - date_fmt:     timestamp format
      - console:     whether to add a RichHandler for stdout
      - log_file:    path to a rolling file log (optional)
      - file_level:  level for the file handler (defaults to `level`)
      - max_bytes:   rotate when log exceeds this size (in bytes)
      - backup_count: how many old log files to keep

    Handlers are only added once, even if you call this multiple times.
    """
    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)

    if console and not any(isinstance(h, RichHandler) for h in root.handlers):
        rich_handler = RichHandler(
            rich_tracebacks=True, show_time=False, show_path=False, markup=True
        )
        rich_handler.setFormatter(formatter)
        rich_handler.setLevel(level)
        root.addHandler(rich_handler)

    if log_file:
        file_level = file_level or level

        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        # avoid adding twice for the same filename
        existing = [
            h
            for h in root.handlers
            if isinstance(h, RotatingFileHandler)
            and getattr(h, "baseFilename", "") == log_file
        ]
        if not existing:
            fh = RotatingFileHandler(
                filename=log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            fh.setFormatter(formatter)
            fh.setLevel(file_level)
            root.addHandler(fh)

    return root
