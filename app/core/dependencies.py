from app.core.config import settings
from app.core.logger import logger
from app.domain.sequential.repository import SequentialRepository

def get_settings():
    return settings

def get_logger():
    return logger

def get_sequential_repo() -> SequentialRepository:
    return SequentialRepository(settings.SEQUENTIAL_MODEL_PATH)