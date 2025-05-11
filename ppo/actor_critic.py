import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        """
        :param state_dim: размерность пространства состояний
        :param action_dim: размерность пространства действий
        """
        super(ActorCritic, self).__init__()

        # Слой актора (политика)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1),
        )
        # Слой критика (ценность)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def act(self, state: torch.Tensor) -> tuple:
        """
        Выбирает действие на основе политики.

        :param state: тензор состояния
        :return: кортеж (действие, логарифм вероятности, ценность состояния)
        """
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> tuple:
        """
        Оценивает действия и ценности состояния.

        :param state: батч состояний
        :param action: батч действий
        :return: логарифмы вероятностей, ценности состояний, энтропия
        """
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy
