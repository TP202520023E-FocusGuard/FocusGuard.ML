from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

# -------------------------
# Sesión individual de navegación
# -------------------------
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
    fue_bloqueado: bool
    usuario_ignoro_advertencia: bool


# -------------------------
# Request principal que enviará el backend
# -------------------------
class SequenceRequest(BaseModel):
    id_usuario: int
    historial: List[NavigationSession]


# -------------------------
# Respuesta del modelo ML
# -------------------------
class SequencePrediction(BaseModel):
    focus_level: float
    needs_intervention: bool
    confidence: float
    predicted_duration: int
    risk_factors: List[str]
    sequence_stats: Dict[str, Any]


# Alias para compatibilidad (opcional)
SequenceData = SequenceRequest
