from app.domain.sequential.service import SequentialService
from app.domain.sequential.schemas import SequenceRequest, SequencePrediction

class PredictSequenceUseCase:
    def __init__(self, sequential_service: SequentialService):
        self.sequential_service = sequential_service
    
    def execute(self, data: SequenceRequest) -> SequencePrediction:
        """Caso de uso: Predecir secuencia de enfoque basado en navegación"""
        return self.sequential_service.predict_focus_level(data)