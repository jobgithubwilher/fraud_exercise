import logging
from app.config import LOG_LEVEL


def setup_logging():
    logging.basicConfig(
        level=LOG_LEVEL,
        format=("%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
    )
    return logging.getLogger(__name__)


logger = setup_logging()
