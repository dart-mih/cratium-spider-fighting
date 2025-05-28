import torch
import torch.nn as nn
from tqdm import tqdm

from .buffer import RolloutBuffer
from .actor_critic import ActorCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_actor: float,
        lr_critic: float,
        gamma: float,
        K_epochs: int,
        eps_clip: float,
    ):
        """
        :param state_dim: размерность пространства состояний
        :param action_dim: размерность пространства действий
        :param lr_actor: learning rate для актера
        :param lr_critic: learning rate для критика
        :param gamma: коэффициент дисконтирования
        :param K_epochs: количество эпох обучения на 1 пакете данных из буфера
        :param eps_clip: значение клипинга для PPO (от 0.2 до 0.3 обычно)
        """
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )

        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state) -> tuple:
        """
        Выбирает действие на основе текущей политики.

        :param state: состояние среды
        :return: кортеж (действие, логарифм вероятности, ценность состояния, тензор состояния)
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state)

        return action, action_logprob, state_val, state

    def update(self, buffer: RolloutBuffer, mini_batch_size: int = 128) -> None:
        """
        Обновляет политику на основе буфера с помощью метода Монте-Карло с обучением на мини-батчах.

        :param buffer: экземпляр RolloutBuffer с собранными данными для обучения
        :param mini_batch_size: размер мини-батча для обучения
        """
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(buffer.rewards), reversed(buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Нормализация наград
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Преобразование списков в тензоры
        old_states = torch.stack(buffer.states, dim=0).detach().to(device)
        old_actions = torch.squeeze(
            torch.stack(buffer.actions, dim=0).detach().to(device)
        )
        old_logprobs = torch.squeeze(
            torch.stack(buffer.logprobs, dim=0).detach().to(device)
        )
        old_state_values = torch.squeeze(
            torch.stack(buffer.state_values, dim=0).detach().to(device)
        )

        advantages = rewards.detach() - old_state_values.detach()

        # Оптимизация политики в течение K эпох
        for _ in tqdm(range(self.K_epochs), "Эпоха обновления весов"):
            batch_size = len(buffer.states)
            indices = torch.randperm(batch_size, device=device)

            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mini_batch_indices = indices[start:end]

                # Выбираем мини-батч
                mb_states = old_states[mini_batch_indices]
                mb_actions = old_actions[mini_batch_indices]
                mb_old_logprobs = old_logprobs[mini_batch_indices]
                mb_rewards = rewards[mini_batch_indices]
                mb_advantages = advantages[mini_batch_indices]

                # Оцениваем мини-батч
                logprobs, state_values, dist_entropy = self.policy.evaluate(
                    mb_states, mb_actions
                )
                state_values = torch.squeeze(state_values)
                ratios = torch.exp(logprobs - mb_old_logprobs.detach())

                # Находим Surrogate Loss
                surr1 = ratios * mb_advantages
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * mb_advantages
                )

                loss = (
                    -torch.min(surr1, surr2)
                    + 0.5 * self.MseLoss(state_values, mb_rewards)
                    - 0.01 * dist_entropy
                )

                # Оптимизация для каждого мини-батча
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        # Обновление старой политики
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save(self, checkpoint_path: str) -> None:
        """
        Сохраняет веса модели.

        :param checkpoint_path: путь для сохранения
        """
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path: str) -> None:
        """
        Загружает веса модели.

        :param checkpoint_path: путь к файлу весов
        """
        self.policy_old.load_state_dict(torch.load(checkpoint_path))
        self.policy.load_state_dict(torch.load(checkpoint_path))
