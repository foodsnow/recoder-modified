import logging

logger = logging
logger.basicConfig(
    format='%(asctime)s - %(pathname)s - %(funcName)s - %(levelname)s - %(message)s',
    level=logging.DEBUG,
    filename='log.recoder.log'
)
