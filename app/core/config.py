from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "FOCUSGUARD.ML"
    DEBUG: bool = True
    
    # Model Paths
    MODELS_DIR: str = "models"
    SEQUENTIAL_MODEL_PATH: str = "models/gru_model.pt"
    
    # Model Config - ACTUALIZADO para 8 features
    SEQUENCE_LENGTH: int = 10  # Máximo de sesiones históricas a analizar
    FEATURE_DIM: int = 8       # 8 features por sesión de navegación
    
    class Config:
        env_file = ".env"

settings = Settings()