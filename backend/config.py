import logging
import logging.handlers
import os
import threading
import time
from dataclasses import dataclass
from dotenv import load_dotenv, find_dotenv, dotenv_values

# Load environment variables from .env file
_ENV_PATH = find_dotenv()
load_dotenv(_ENV_PATH)

@dataclass
class Config:
    """Configuration settings for the RAG system"""
    # Anthropic API settings
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    #ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    ANTHROPIC_MODEL: str = "claude-sonnet-4-5-20250929"

    # Embedding model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Document processing settings
    CHUNK_SIZE: int = 800       # Size of text chunks for vector storage
    CHUNK_OVERLAP: int = 100     # Characters to overlap between chunks
    MAX_RESULTS: int = 5         # Maximum search results to return
    MAX_HISTORY: int = 2         # Number of conversation messages to remember

    # Database paths
    CHROMA_PATH: str = "./chroma_db"  # ChromaDB storage location

    # Logging settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")  # DEBUG, INFO, WARNING, ERROR
    LOG_FILE: str = os.getenv("LOG_FILE", "./logs/app.log")  # Rotating log file location
    LOG_MAX_BYTES: int = 5 * 1024 * 1024  # Roll over after 5 MB
    LOG_BACKUP_COUNT: int = 5              # Keep up to 5 rolled-over backups

config = Config()

# Configure logging for the whole backend. This module is imported before any
# other backend module, so setting the root logger up here ensures every
# `logging.getLogger(__name__)` call elsewhere picks up this configuration.
_log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
_log_formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_log_formatter)

_log_dir = os.path.dirname(config.LOG_FILE)
if _log_dir:
    os.makedirs(_log_dir, exist_ok=True)

_file_handler = logging.handlers.RotatingFileHandler(
    config.LOG_FILE,
    maxBytes=config.LOG_MAX_BYTES,
    backupCount=config.LOG_BACKUP_COUNT,
)
_file_handler.setFormatter(_log_formatter)

logging.basicConfig(level=_log_level, handlers=[_console_handler, _file_handler])

# Third-party libraries emit very verbose DEBUG output (full HTTP payloads,
# connection pool chatter) that drowns out this app's own DEBUG logs. Cap
# them at WARNING regardless of LOG_LEVEL so DEBUG stays focused on our code.
for _noisy_logger in (
    "urllib3",
    "httpx",
    "httpcore",
    "chromadb",
    "sentence_transformers",
    "huggingface_hub",
    "posthog",
    "anthropic",
):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

# Watch the .env file for LOG_LEVEL changes and apply them to the root logger
# without requiring an app restart. Polls on a daemon thread so it never
# blocks shutdown; picks up edits by mtime rather than a filesystem-event
# library to avoid an extra dependency.
_ENV_WATCH_INTERVAL_SECONDS = 2.0

def _watch_log_level(path: str, interval: float) -> None:
    watch_logger = logging.getLogger(__name__)
    last_mtime = None

    while True:
        time.sleep(interval)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue

        if last_mtime is None:
            last_mtime = mtime
            continue
        if mtime == last_mtime:
            continue
        last_mtime = mtime

        new_level_name = dotenv_values(path).get("LOG_LEVEL")
        if not new_level_name:
            continue

        new_level = getattr(logging, new_level_name.upper(), None)
        if new_level is None:
            watch_logger.warning("Ignoring invalid LOG_LEVEL %r in %s", new_level_name, path)
            continue

        root_logger = logging.getLogger()
        if new_level != root_logger.level:
            root_logger.setLevel(new_level)
            config.LOG_LEVEL = new_level_name.upper()
            watch_logger.info("LOG_LEVEL changed to %s (picked up from %s)", config.LOG_LEVEL, path)

if _ENV_PATH:
    threading.Thread(
        target=_watch_log_level,
        args=(_ENV_PATH, _ENV_WATCH_INTERVAL_SECONDS),
        daemon=True,
        name="log-level-watcher",
    ).start()

