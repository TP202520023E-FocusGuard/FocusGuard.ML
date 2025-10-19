import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, sequence_length: int = 10, feature_dim: int = 8, hidden_dim: int = 64):
        super(GRUModel, self).__init__()
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim  # ✅ DEBE SER 8
        
        print(f"🔥 INITIALIZING GRU MODEL WITH feature_dim: {feature_dim}")  # Debug
        
        self.gru = nn.GRU(
            input_size=feature_dim,  # ✅ 8 features
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        print(f"🔥 GRU MODEL CREATED: input_size={feature_dim}")  # Debug
    
    def forward(self, x):
        print(f"🔥 FORWARD INPUT SHAPE: {x.shape}")  # Debug
        # x debe ser: (batch_size, sequence_length, 8)
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        return self.classifier(last_output)