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
        """Predice enfoque basado en historial real de navegación"""
        try:
            # Convertir sesiones de navegación a features para el GRU
            features = self._sessions_to_features(request.navigation_sessions)
            sequences = self._preprocess_sequences(features)
            
            with torch.no_grad():
                prediction = self.model(sequences)
                focus_probability = prediction.item()
            
            # Analizar factores de riesgo
            risk_factors = self._analyze_risk_factors(request.navigation_sessions, request.current_context)
            
            return SequencePrediction(
                focus_level=focus_probability,
                needs_intervention=focus_probability < 0.6 or len(risk_factors) > 0,
                confidence=abs(focus_probability - 0.5) * 2,
                predicted_duration=self._predict_remaining_focus(focus_probability, request.navigation_sessions),
                risk_factors=risk_factors,
                sequence_stats=self._calculate_session_stats(request.navigation_sessions)
            )
        except Exception as e:
            raise ValueError(f"Prediction error: {str(e)}")
    
    def _sessions_to_features(self, sessions: List[NavigationSession]) -> List[List[float]]:
        """Convierte sesiones de navegación en features numéricas para el GRU"""
        features = []
        
        for session in sessions:
            # Feature engineering basado en tus datos reales
            session_features = [
                # 1. Duración normalizada (0-1, donde 1 = 3600 segundos = 1 hora)
                min(session.duracion_segundos / 3600.0, 1.0),
                
                # 2. Categoría del dominio (one-hot encoded)
                self._domain_to_productivity_score(session.dominio),
                
                # 3. Hora del día normalizada (0-1)
                session.hora_dia / 23.0,
                
                # 4. Es fin de semana (0 o 1)
                1.0 if session.es_fin_semana else 0.0,
                
                # 5. Patrón de uso (codificado)
                self._pattern_to_value(session.patron_uso),
                
                # 6. Contexto anterior (codificado)
                self._context_to_value(session.contexto_anterior),
                
                # 7. Fue bloqueado (0 o 1)
                1.0 if session.fue_bloqueado else 0.0,
                
                # 8. Ignoró advertencia (0 o 1)
                1.0 if session.usuario_ignoro_advertencia else 0.0
            ]
            features.append(session_features)
        
        return features
    
    def _domain_to_productivity_score(self, domain: str) -> float:
        """Convierte dominio a score de productividad"""
        for key, category in self.domain_categories.items():
            if key in domain:
                if category == 'productivo':
                    return 0.9
                elif category == 'ocio':
                    return 0.1
                elif category == 'redes_sociales':
                    return 0.3
        return 0.5  # Neutral por defecto
    
    def _pattern_to_value(self, pattern: str) -> float:
        """Convierte patrón de uso a valor numérico"""
        patterns = {
            'intenso': 0.9,
            'medio': 0.5, 
            'bajo': 0.1
        }
        return patterns.get(pattern.lower(), 0.5)
    
    def _context_to_value(self, context: str) -> float:
        """Convierte contexto a valor numérico"""
        contexts = {
            'estudiando': 0.9,
            'trabajando': 0.8,
            'descanso': 0.2,
            'ocio': 0.1
        }
        return contexts.get(context.lower(), 0.5)
    
    def _preprocess_sequences(self, features: List[List[float]]) -> torch.Tensor:
        """Preprocesa sequences para el GRU"""
        sequences_array = np.array(features, dtype=np.float32)
        
        # El GRU espera: (batch_size, sequence_length, feature_dim)
        # Si tenemos N sesiones con M features cada una
        batch_size = 1  # Una secuencia por predicción
        sequence_length = sequences_array.shape[0]  # Número de sesiones
        feature_dim = sequences_array.shape[1]      # Número de features por sesión
        
        reshaped_sequences = sequences_array.reshape(batch_size, sequence_length, feature_dim)
        
        return torch.tensor(reshaped_sequences)
    
    def _analyze_risk_factors(self, sessions: List[NavigationSession], current_context: str) -> List[str]:
        """Analiza factores de riesgo en el historial"""
        risk_factors = []
        
        if not sessions:
            return ["sin_historial"]
        
        # Últimas 3 sesiones
        recent_sessions = sessions[-3:] if len(sessions) >= 3 else sessions
        
        # Verificar patrones de riesgo
        ocio_count = sum(1 for s in recent_sessions 
                        if self._domain_to_productivity_score(s.dominio) < 0.3)
        
        if ocio_count >= 2:
            risk_factors.append("muchas_sesiones_ocio_recientes")
        
        # Verificar si viene de descanso pero el contexto actual es trabajo
        last_session = sessions[-1]
        if (last_session.contexto_anterior in ['descanso', 'ocio'] and 
            current_context in ['estudiando', 'trabajando']):
            risk_factors.append("transicion_descanso_a_trabajo")
        
        # Verificar si ignoró advertencias recientemente
        if any(s.usuario_ignoro_advertencia for s in recent_sessions):
            risk_factors.append("advertencias_ignoradas")
        
        return risk_factors
    
    def _predict_remaining_focus(self, focus_probability: float, sessions: List[NavigationSession]) -> int:
        """Predice duración restante de enfoque en segundos"""
        base_duration = 1800  # 30 minutos base
        
        if not sessions:
            return int(base_duration * focus_probability)
        
        # Ajustar basado en historial reciente
        recent_durations = [s.duracion_segundos for s in sessions[-3:] if s.duracion_segundos > 0]
        if recent_durations:
            avg_duration = sum(recent_durations) / len(recent_durations)
            adjusted_duration = (avg_duration + base_duration) / 2
        else:
            adjusted_duration = base_duration
        
        return int(adjusted_duration * focus_probability)
    
    def _calculate_session_stats(self, sessions: List[NavigationSession]) -> Dict[str, Any]:
        """Calcula estadísticas del historial"""
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
        """Calcula tendencia de productividad"""
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