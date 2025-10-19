import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


# --- Red Neuronal Q ---
class DQNNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.output(x)


# --- Memoria de Repetición (Replay Buffer) ---
class ReplayMemory:
    def __init__(self, capacity: int = 10000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# --- Agente DQN ---
class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.1,
        epsilon_decay: float = 0.995,
        device: str = "cpu",
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = torch.device(device)

        self.memory = ReplayMemory()
        self.model = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        """Selecciona una acción (exploración vs explotación)."""
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def train_step(self):
        """Realiza un paso de entrenamiento con un batch de memoria."""
        if len(self.memory) < self.batch_size:
            return  # Aún no hay suficientes muestras

        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Q actual
        current_q = self.model(states).gather(1, actions)
        # Q objetivo
        next_q = self.model(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + (self.gamma * next_q * (1 - dones))

        # Pérdida
        loss = self.loss_fn(current_q, target_q.detach())

        # Optimización
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decaimiento de epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
