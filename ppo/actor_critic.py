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

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=state_dim[2],
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Вычисляем размер выхода сверточной части
        with torch.no_grad():
            dummy_input = torch.zeros(1, *state_dim).permute(
                0, 3, 1, 2
            )  # (batch, channels, height, width)
            conv_output_size = self.conv(dummy_input).shape[1]

        # Слой актора (политика)
        self.actor = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1),
        )
        # Слой критика (ценность)
        self.critic = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def preprocess_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Переводим state в формат (batch, channels, height, width)
        """

        if len(state.shape) == 3:  # Если нет batch dimension
            state = state.permute(2, 0, 1).unsqueeze(0)
        else:
            state = state.permute(0, 3, 1, 2)

        return state

    def act(self, state: torch.Tensor) -> tuple:
        """
        Выбирает действие на основе политики.

        :param state: тензор состояния
        :return: кортеж (действие, логарифм вероятности, ценность состояния)
        """
        features = self.conv(self.preprocess_state(state))
        action_probs = self.actor(features)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(features)

        return action.detach(), action_logprob.detach(), state_val.detach()

    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> tuple:
        """
        Оценивает действия и ценности состояния.

        :param state: батч состояний
        :param action: батч действий
        :return: логарифмы вероятностей, ценности состояний, энтропия
        """
        features = self.conv(self.preprocess_state(state))
        action_probs = self.actor(features)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(features)

        return action_logprobs, state_values, dist_entropy
