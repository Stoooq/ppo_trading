import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_actions: int = 3):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
        )

        self.actor_head = nn.Linear(in_features=hidden_dim, out_features=num_actions)

        self.critic_head = nn.Linear(in_features=hidden_dim, out_features=1)

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor = None):
        logits, value = self.forward(x)

        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), value.squeeze(-1)
