# settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    db_host: str = "localhost"
    db_name: str = "postgres"
    db_user: str = "postgres"
    db_password: str
    # this tells pydantic to load the variables from .env
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()