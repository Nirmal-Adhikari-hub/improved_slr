import logging, sys
from pathlib import Path

def setup_logger(save_dir: str, name: str = "train", level=logging.INFO) -> logging.Logger:
    """
    Creates and returns a configured logger that writes log messages to both the
    console (with optional color formatting) and a log file

    Args:
        save_dir (str): Directory where the log file will be stored.
        name (str, optional): Logger name and log file prefix. Defaults to "train".
        level (int, optional): Logging level threshold (eg "logging.INFO", 
        "logging.DEBUG"). Defaults to logging.INFO.

    Returns:
        logging.Logger: Configure logger instance
    """
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    try:
        import colorlog
        cfmt = "%(log_color)s" + fmt
        ch.setFormatter(colorlog.ColoredFormatter(cfmt))
    except Exception:
        ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)

    # File Handler
    fh = logging.FileHandler(Path(save_dir)/f"{name}.log")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)

    logger.propagate = False
    return logger