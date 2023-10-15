import logging
from colorlog import ColoredFormatter

# Create a logger instance
LOGGER = logging.getLogger(__name__)

# Configure the logger (only once)
if not LOGGER.handlers:
    formatter = ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "INFO": "white",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )

    # Create a handler
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    # Set the log level and add the handler to the logger
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(handler)
