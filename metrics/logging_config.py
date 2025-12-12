"""
Logging Configuration for T.A.R.S.

Provides structured JSON logging or human-readable text logging.
"""

import logging
import sys
from typing import Literal, Optional
from pathlib import Path
import json
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data)


def configure_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    format: Literal["json", "text"] = "json",
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Configure logging for T.A.R.S.

    Args:
        level: Logging level
        format: Log format (json or text)
        log_file: Optional log file path

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("tars")
    logger.setLevel(getattr(logging, level))

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level))

    if format == "json":
        console_handler.setFormatter(JSONFormatter())
    else:
        text_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        console_handler.setFormatter(logging.Formatter(text_format))

    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level))

        if format == "json":
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(text_format))

        logger.addHandler(file_handler)

    return logger


# Example usage
if __name__ == "__main__":
    # JSON logging
    logger = configure_logging(
        level="INFO",
        format="json",
        log_file=Path("logs/tars.log"),
    )

    logger.info("T.A.R.S. starting up")
    logger.debug("Debug message")
    logger.warning("Warning message")

    try:
        raise ValueError("Example error")
    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
