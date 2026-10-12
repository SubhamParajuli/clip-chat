"""Central logging configuration for application events."""

import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_FILE = Path("app.log")
APPLICATION_LOGGER = "src"
_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    """Configure application logging."""
    global _configured
    if _configured:
        return

    # Ignore HTTPX INFO and DEBUG logs.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    application_logger = logging.getLogger(APPLICATION_LOGGER)
    application_logger.setLevel(level)
    application_logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    application_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    application_logger.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return an application logger."""
    configure_logging()
    return logging.getLogger(name)