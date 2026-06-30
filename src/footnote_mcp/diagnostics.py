from __future__ import annotations

import logging
import os
import sys


def get_logger() -> logging.Logger:
    logger = logging.getLogger("footnote")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False

    level_name = os.environ.get("FOOTNOTE_LOG_LEVEL", "WARNING").upper()
    logger.setLevel(getattr(logging, level_name, logging.WARNING))
    return logger


log = get_logger()
