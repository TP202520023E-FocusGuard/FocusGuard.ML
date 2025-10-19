from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

class NavigationSession(BaseModel):
    dominio: str
    fecha_inicio: datetime
    fecha_fin: Optional[datetime] = None
    duracion_segundos: int
    dia_semana: str
    hora_dia: int
    es_fin_semana: bool
    patron_uso: str
    contexto_anterior: str
    fue_bloqueado: bool = False
    usuario_ignoro_advertencia: bool = False

class SequenceRequest(BaseModel):
    user_id: str
    navigation_sessions: List[NavigationSession]
    current_context: str = "trabajando"

class SequencePrediction(BaseModel):
    focus_level: float
    needs_intervention: bool
    confidence: float
    predicted_duration: int
    risk_factors: List[str]
    sequence_stats: Dict[str, Any]

# Mantener los nombres antiguos para compatibilidad
SequenceData = SequenceRequest  # Alias para compatibilidad