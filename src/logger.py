"""
Logger setup module.

This module provides a reusable logger that writes logs both to a file and the console.
The console output is colorized using `colorlog` for easier debugging during development.

Usage:
    from logger import get_logger
    logger = get_logger(__name__)
    logger.info("This is an info message.")
"""

import logging
from datetime import datetime
from pathlib import Path

from colorlog import ColoredFormatter

# Create logs directory if it doesn't exist
LOGS_DIR = Path("loges")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Define log file path
LOG_FILE = LOGS_DIR / f"log_{datetime.now().strftime('%Y-%m-%d')}.log"

# Create a custom logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# File handler (logs to a file)
file_handler = logging.FileHandler(LOG_FILE)
file_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_format)
logger.addHandler(file_handler)

# Console handler (colored logs)
console_handler = logging.StreamHandler()
console_format = ColoredFormatter(
    "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
)
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)


# Function to get named loggers
def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
