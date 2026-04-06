# utils/logger.py

import sys
from pathlib import Path
from loguru import logger
from env.config import LOG_DIR, LOG_LEVEL


# ─── Create Log Directory ───────────────────────────────────
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ─── Remove Default Logger ──────────────────────────────────
logger.remove()


# ─── Console Logger ─────────────────────────────────────────
logger.add(
    sys.stdout,
    level=LOG_LEVEL,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
    colorize=True,
)


# ─── File Logger ────────────────────────────────────────────
logger.add(
    LOG_DIR / "app.log",
    level=LOG_LEVEL,
    format=(
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{name}:{line} | "
        "{message}"
    ),
    rotation="10 MB",       # New file after 10MB
    retention="7 days",     # Keep logs for 7 days
    compression="zip",      # Compress old logs
)


# ─── Error Logger ───────────────────────────────────────────
logger.add(
    LOG_DIR / "error.log",
    level="ERROR",
    format=(
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{name}:{line} | "
        "{message}"
    ),
    rotation="10 MB",
    retention="30 days",    # Keep errors longer
    compression="zip",
)


# ─── Helper Functions ───────────────────────────────────────
def get_logger(name: str):
    """Get a named logger."""
    return logger.bind(name=name)


def log_episode_start(episode_id: str, task_level: str):
    """Log episode start."""
    logger.info(f"Episode started | {episode_id} | task={task_level}")


def log_episode_end(episode_id: str, score: float, passed: bool):
    """Log episode end."""
    logger.info(
        f"Episode ended | {episode_id} | "
        f"score={score} | passed={passed}"
    )


def log_step(episode_id: str, step: int, action: str, reward: float):
    """Log a single step."""
    logger.debug(
        f"Step | {episode_id} | "
        f"step={step} | action={action} | reward={reward}"
    )


def log_error(message: str, error: Exception):
    """Log an error."""
    logger.error(f"{message} | error={str(error)}")