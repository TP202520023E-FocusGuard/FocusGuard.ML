from fastapi import APIRouter, Depends, HTTPException
from app.application.use_cases.predict_sequence import PredictSequenceUseCase
from app.domain.sequential.schemas import SequenceRequest, SequencePrediction
from app.domain.sequential.service import SequentialService
from app.domain.sequential.repository import SequentialRepository
from app.core.dependencies import get_sequential_repo

router = APIRouter(prefix="/sequential", tags=["sequential"])

def get_sequential_use_case(
    repo: SequentialRepository = Depends(get_sequential_repo)
) -> PredictSequenceUseCase:
    service = SequentialService(repo.get_model())
    return PredictSequenceUseCase(service)

@router.post("/predict", response_model=SequencePrediction)
async def predict_sequence(
    data: SequenceRequest,
    use_case: PredictSequenceUseCase = Depends(get_sequential_use_case)
):
    """Endpoint para predecir enfoque usando GRU basado en historial de navegación"""
    try:
        return use_case.execute(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@router.get("/status")
async def get_status(repo: SequentialRepository = Depends(get_sequential_repo)):
    """Estado del modelo GRU"""
    try:
        repo.load_model()
        model = repo.get_model()
        return {
            "status": "ready", 
            "model": "GRU Sequential",
            "sequence_length": model.sequence_length,
            "feature_dim": model.feature_dim,  # ✅ Esto debe mostrar 8
            "model_initialized": True,
            "description": "Analiza historial de navegación para predecir enfoque"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# ✅ NUEVO ENDPOINT PARA DEBUG
@router.get("/debug-model")
async def debug_model(repo: SequentialRepository = Depends(get_sequential_repo)):
    """Debug info del modelo"""
    model = repo.get_model()
    
    debug_info = {
        "model_feature_dim": model.feature_dim,
        "model_sequence_length": model.sequence_length,
        "gru_input_size": model.gru.input_size,
        "model_device": str(model.device),
        "state_dict_keys": list(model.state_dict().keys()) if hasattr(model, 'state_dict') else []
    }
    
    # Verificar dimensiones del GRU
    if hasattr(model.gru, 'weight_ih_l0'):
        debug_info["gru_weight_ih_shape"] = model.gru.weight_ih_l0.shape
        debug_info["expected_features"] = model.gru.weight_ih_l0.shape[1]
    
    return debug_info