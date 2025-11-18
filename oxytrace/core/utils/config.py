import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class AppConfig:
    dataset_url: str = os.getenv("DATASET_URL")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    runtime_environment: str = os.getenv("RUNTIME_ENVIRONMENT", "local")
