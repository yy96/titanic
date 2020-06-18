import logging

from titanic_project.config import config
from titanic_project.config import logging_config

VERSION_PATH = config.PACKAGE_ROOT / 'VERSION'

# __name__ will be titanic_project
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging_config.get_console_handler())
logger.propagate = False

with open(VERSION_PATH, 'r') as version_file:
    __version__ = version_file.read().strip()