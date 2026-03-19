import sys

LOGURU = True
try:
    from loguru import logger as log
except ImportError:
    LOGURU = False

if LOGURU:
    log.remove()  # Remove default logger
    log.configure(
        handlers=[
            {
                "sink": sys.stderr,
                "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:8s}</level> | {message} | <cyan><dim>{module}:{function}</dim></cyan>",
                "colorize": True,
            },
            {
                "sink": "loguru.log",
                "format": "{time:YYYY-MM-DD HH:mm:ss} | {level:8s} | {message} | {module}:{function}",
                "rotation": "23:59",
            },
        ]
    )

__version__ = "19-Mar-2026 12:00h"
