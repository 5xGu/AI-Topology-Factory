'''
There exists a logging class in Discuss Data and Discuss Data Next repos.
Whether to keep both or unify needs to be decided.
'''

import logging
import os
from typing import Optional

def setup_logger(
    log_name: str,
    output_dir: str = "logs",
    level: int = logging.INFO,
    mode: str = "a", 
    encoding: str = "utf-8",
    formatter: Optional[logging.Formatter] = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
) -> logging.Logger:
    """
    Create or retrieve a logger that writes to a file into an `output_dir`.

    Args:
        output_dir: Directory to store log files, gets created if it doesn't exist already.
        log_name: Name of the log file (and associated logger). Its extension is preserved.
        level: Logging level (default is INFO).
        mode: file mode, a-ppend or w-rite.
        encoding: defaults to utf-8.
        formatter: defaults to `%(asctime)s - %(name)s - %(levelname)s - %(message)s.

    Returns:
        Configured logger instance. Reuses existing logger if `log_name` matches.

    Example:
        >>> logger = setup_logger("logs/", "app.log")
        >>> logger.info("Hello, world!")
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    logger_name = os.path.splitext(log_name)[0]
    logger = logging.getLogger(logger_name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    log_path = os.path.join(output_dir, log_name)
    file_handler = logging.FileHandler(log_path, mode=mode, encoding=encoding)

    if formatter is None:
        formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


default_logger = setup_logger(
    log_name="default.log",
    level=logging.DEBUG
)

ai_default_logger = setup_logger(
    log_name="ai_default.log",
)

io_logger = setup_logger(
    log_name="inOut.log",
    level=logging.DEBUG
)

topology_logger = setup_logger(
    log_name="topology.log",
)