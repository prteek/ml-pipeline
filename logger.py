import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# the handler determines where the logs go: stdout/file
stream_handler = logging.StreamHandler()

log_file_name = "debug_" + datetime.now().strftime("%Y-%m-%d_%H:%M") + ".log"
# file_handler = logging.FileHandler(log_file_name)

# Set level. Highest level in logger
logger.setLevel(logging.DEBUG)
stream_handler.setLevel(logging.DEBUG)
# file_handler.setLevel(logging.INFO)

# the formatter determines what our logs will look like
stream_formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s %(funcName)s : %(message)s"
)
# file_formatter = logging.Formatter("%(levelname)s %(asctime)s %(message)s")

# here we hook everything together
stream_handler.setFormatter(stream_formatter)
# file_handler.setFormatter(file_formatter)

logger.addHandler(stream_handler)
# logger.addHandler(file_handler)
