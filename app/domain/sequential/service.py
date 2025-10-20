import torch
import numpy as np
from typing import List, Dict, Any
from .model import GRUModel
from .schemas import SequenceRequest, SequencePrediction, NavigationSession
from datetime import datetime

class SequentialService:
    def __init__(self, model: GRUModel):
        self.model = model
        self.model.eval()
        
        self.domain_categories = {
            'figma.com': 'productivo',
            'github.com': 'productivo', 
            'stackoverflow.com': 'productivo',
            'falabella.com.pe': 'ocio',
            'youtube.com': 'ocio',
            'netflix.com': 'ocio',
            'facebook.com': 'redes_sociales',
            'instagram.com': 'redes_sociales'
        }
    
    def predict_focus_level(self, request: SequenceRequest) -> SequencePrediction:
        """Predice nivel de enfoque usando el historial real de navegación"""
        try:
            # ✅ Ahora usamos 'historial' en lugar de 'navigation_sessions'
            sessions = request.historial

            # Convertir historial a features para el GRU
            features = self._sessions_to_features(sessions)
            sequences = self._preprocess_sequences(features)
            
            with torch.no_grad():
                prediction = self.model(sequences)
                focus_probability = prediction.item()
            
            # Analizar factores de riesgo
            risk_factors = self._analyze_risk_factors(sessions)
            
            return SequencePrediction(
                focus_level=focus_probability,
                needs_intervention=focus_probability < 0.6 or len(risk_factors) > 0,
                confidence=abs(focus_probability - 0.5) * 2,
                predicted_duration=self._predict_remaining_focus(focus_probability, sessions),
                risk_factors=risk_factors,
                sequence_stats=self._calculate_session_stats(sessions)
            )
        except Exception as e:
            raise ValueError(f"Prediction error: {str(e)}")
    
    def _sessions_to_features(self, sessions: List[NavigationSession]) -> List[List[float]]:
        """Convierte historial de navegación en features numéricas para el GRU"""
        features = []
        
        for session in sessions:
            session_features = [
                min(session.duracion_segundos / 3600.0, 1.0),
                self._domain_to_productivity_score(session.dominio),
                session.hora_dia / 23.0,
                1.0 if session.es_fin_semana else 0.0,
                self._pattern_to_value(session.patron_uso),
                self._context_to_value(session.contexto_anterior),
                1.0 if session.fue_bloqueado else 0.0,
                1.0 if session.usuario_ignoro_advertencia else 0.0
            ]
            features.append(session_features)
        
        return features
    
    def _domain_to_productivity_score(self, domain: str) -> float:
        for key, category in self.domain_categories.items():
            if key in domain:
                return {'productivo': 0.9, 'ocio': 0.1, 'redes_sociales': 0.3}[category]
        return 0.5
    
    def _pattern_to_value(self, pattern: str) -> float:
        return {'intenso': 0.9, 'medio': 0.5, 'bajo': 0.1}.get(pattern.lower(), 0.5)
    
    def _context_to_value(self, context: str) -> float:
        return {'estudiando': 0.9, 'trabajando': 0.8, 'descanso': 0.2, 'ocio': 0.1}.get(context.lower(), 0.5)
    
    def _preprocess_sequences(self, features: List[List[float]]) -> torch.Tensor:
        sequences_array = np.array(features, dtype=np.float32)
        reshaped_sequences = sequences_array.reshape(1, sequences_array.shape[0], sequences_array.shape[1])
        return torch.tensor(reshaped_sequences)
    
    def _analyze_risk_factors(self, sessions: List[NavigationSession]) -> List[str]:
        risk_factors = []
        if not sessions:
            return ["sin_historial"]
        
        recent = sessions[-3:]
        ocio_count = sum(1 for s in recent if self._domain_to_productivity_score(s.dominio) < 0.3)
        if ocio_count >= 2:
            risk_factors.append("muchas_sesiones_ocio_recientes")
        
        if any(s.usuario_ignoro_advertencia for s in recent):
            risk_factors.append("advertencias_ignoradas")
        
        return risk_factors
    
    def _predict_remaining_focus(self, focus_probability: float, sessions: List[NavigationSession]) -> int:
        base_duration = 1800
        if not sessions:
            return int(base_duration * focus_probability)
        
        recent = [s.duracion_segundos for s in sessions[-3:] if s.duracion_segundos > 0]
        if recent:
            avg_duration = sum(recent) / len(recent)
            adjusted_duration = (avg_duration + base_duration) / 2
        else:
            adjusted_duration = base_duration
        
        return int(adjusted_duration * focus_probability)
    
    def _calculate_session_stats(self, sessions: List[NavigationSession]) -> Dict[str, Any]:
        if not sessions:
            return {"sessions_analyzed": 0}
        
        durations = [s.duracion_segundos for s in sessions if s.duracion_segundos > 0]
        productivity_scores = [self._domain_to_productivity_score(s.dominio) for s in sessions]
        
        return {
            "sessions_analyzed": len(sessions),
            "avg_session_duration": sum(durations) / len(durations) if durations else 0,
            "avg_productivity_score": sum(productivity_scores) / len(productivity_scores),
            "productivity_trend": self._calculate_productivity_trend(productivity_scores),
            "recent_contexts": [s.contexto_anterior for s in sessions[-3:]]
        }
    
    def _calculate_productivity_trend(self, scores: List[float]) -> str:
        if len(scores) < 2:
            return "estable"
        recent_avg = sum(scores[-3:]) / min(3, len(scores))
        older_avg = sum(scores[:-3]) / max(1, len(scores) - 3) if len(scores) > 3 else scores[0]
        if recent_avg > older_avg + 0.1:
            return "mejorando"
        elif recent_avg < older_avg - 0.1:
            return "empeorando"
        else:
            return "estable"
