import logging

logger = logging
logger.basicConfig(
    format='%(asctime)s - %(levelname)s - %(pathname)s - %(funcName)s - %(message)s',
    level=logging.DEBUG,
    filename='log.recoder.log'
)
