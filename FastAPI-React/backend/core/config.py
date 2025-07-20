from typing import List
from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    # should match with .env file variables
    DATABASE_URL: str
    API_PREFIX: str = "/api"
    DEBUG: bool = False
    ALLOWED_ORIGINS: str = ""
    OPENAI_API_KEY: str

    @field_validator("ALLOWED_ORIGINS") # Custom validator to parse comma-separated origins
    def parse_allowed_origins(cls, v: str) -> List[str]:
        return v.split(",") if v else []
    
    class Config: # Tells pydantic to read from .env file
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

settings = Settings()        