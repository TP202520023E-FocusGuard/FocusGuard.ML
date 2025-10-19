import torch
import os
from .model import GRUModel
from app.core.config import settings
from app.core.logger import logger

class SequentialRepository:
    def __init__(self, model_path: str):
        self.model_path = model_path
        # ✅ FORZAR feature_dim=8 sin importar la configuración
        self.model = GRUModel(
            sequence_length=settings.SEQUENCE_LENGTH,
            feature_dim=8  # ✅ FIJO en 8, ignorando settings
        )
    
    def load_model(self) -> bool:
        try:
            # ✅ Verificar si el archivo existe y tiene dimensiones correctas
            if not os.path.exists(self.model_path):
                logger.warning("⚠️ No model file found. Using new model with 8 features.")
                return False
                
            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=True)
            
            # ✅ Verificar explícitamente las dimensiones
            if 'gru.weight_ih_l0' in checkpoint:
                saved_feature_dim = checkpoint['gru.weight_ih_l0'].shape[1]
                if saved_feature_dim != 8:
                    logger.warning(f"⚠️ Saved model has wrong feature dim: {saved_feature_dim}. Expected 8. Using new model.")
                    return False
            
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            logger.info("✅ GRU model loaded successfully with 8 features")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Could not load GRU model: {str(e)}. Using new model with 8 features.")
            # ✅ Asegurar que el modelo tenga 8 features
            self.model = GRUModel(
                sequence_length=settings.SEQUENCE_LENGTH,
                feature_dim=8  # ✅ FIJO en 8
            )
            self.model.eval()
            return False
    
    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)
        logger.info("💾 GRU model saved with 8 features")
    
    def get_model(self) -> GRUModel:
        return self.model