# logger.py
import logging
import datetime
from typing import List, Dict

class LogCaptureHandler(logging.Handler):
    """Custom handler to capture logs in memory during request processing."""
    def __init__(self):
        super().__init__()
        self.logs = []
    
    def emit(self, record):
        try:
            log_entry = {
                "timestamp": datetime.datetime.fromtimestamp(record.created, datetime.timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": self.format(record)
            }
            self.logs.append(log_entry)
        except Exception:
            self.handleError(record)
    
    def get_logs(self) -> List[Dict]:
        return self.logs

def setup_logging():
    """Configures the root logger."""
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    console_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler]
    )
    logger = logging.getLogger("taxation_bot")
    logger.info("Logging initialized. Logs saved to MongoDB only.")
    return logger

# Initialize logger instance to be imported elsewhere
logger = setup_logging()