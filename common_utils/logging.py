import logging
import sys
import traceback
from pathlib import Path

def setup_logging(
        *,
        log_dir: str,
        log_file: str,
        log_level: str,
        console_enabled: bool,
        name: str,
) -> Path:
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    log_path = log_dir_path / log_file

    root = logging.getLogger()

    # IMPORTANT: avoid duplicate handlers if called twice
    root.handlers.clear()

    level = getattr(logging, log_level.upper(), logging.INFO)
    root.setLevel(level)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # File handler (local file)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Optional console handler
    if console_enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    logging.getLogger(name).info("Logging initialized. file=%s level=%s console=%s", log_path, log_level,
                                     console_enabled)
    return log_path


def install_global_exception_logging(logger: logging.Logger) -> None:
    def handle_exception(exc_type, exc, tb):
        # Keep KeyboardInterrupt clean
        if issubclass(exc_type, KeyboardInterrupt):
            logger.warning("Interrupted by user (KeyboardInterrupt).")
            sys.__excepthook__(exc_type, exc, tb)
            return

        logging.getLogger().critical(
            "Uncaught exception:\n%s",
            "".join(traceback.format_exception(exc_type, exc, tb)),
        )
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = handle_exception