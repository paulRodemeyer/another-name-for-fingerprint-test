import logging
from typing import Optional
from logging.config import dictConfig
import os


def build_logger(name: str, log_file: str, purge: Optional[bool]) -> logging.Logger:
    """
    Create a logger with a specific name and log level that logs to a file.
    """
    # Define color codes for the log levels
    colors = {
        logging.DEBUG: '\033[94m',
        logging.INFO: '\033[92m',
        logging.WARNING: '\033[93m',
        logging.ERROR: '\033[91m',
    }

    # Log formatter for colored console output
    class ColoredFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            color = colors.get(record.levelno, '')
            message = logging.Formatter.format(self, record)
            return f"{color}{message}\033[0m"

    if purge and os.path.exists(log_file):
        os.remove(log_file)

    dictConfig(config={
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s]: %(message)s'
            },
            'colored': {
                '()': ColoredFormatter,
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'file_handler': {
                'level': logging.INFO,
                'class': 'logging.FileHandler',
                'formatter': 'standard',
                'filename': log_file
            },
            'console_handler': {
                'level': logging.DEBUG,
                'formatter': 'colored',
                'class': 'logging.StreamHandler',
            },
        },
        'loggers': {
            name: {
                'handlers': ['file_handler', 'console_handler'],
                'level': logging.DEBUG,
                'propagate': True
            }
        }
    })

    return logging.getLogger(name)