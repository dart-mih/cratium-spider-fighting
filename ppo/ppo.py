import os
from datetime import datetime

import torch
from tqdm import tqdm
from gym import Env

from .agent import PPOAgent
from .buffer import RolloutBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPO:
    def __init__(
        self,
        env: Env,
        env_count: int,
        env_name: str,
        max_ep_len: int,
        max_train_timesteps: int,
        save_model_freq: int,
        update_timestep_mult: int = 4,
        k_epochs: int = 80,
        eps_clip: float = 0.2,
        gamma: float = 0.99,
        lr_actor: float = 0.0003,
        lr_critic: float = 0.001,
        print_freq_mult: int = 10,
        log_freq_mult: int = 2,
    ):
        """
        :param env: среда Gym, где будет происходить обучения PPO (Ожидается синхронный вектор сред)
        :param env_count: количество одинаковых сред, которые запущены параллельно.
        :param env_name: название среды (влияет на папки сохранения результатов)
        :param max_ep_len: макс. длина одного эпизода
        :param max_train_timesteps: общая длина шагов обучения (после скольки шагов обучения прекратится)
        :param save_model_freq: частота сохранения модели (в шагах обучения)
        :param update_timestep_mult: множитель частоты обновления (множитель к max_ep_len)
        :param k_epochs: количество эпох обучения на 1 пакете данных из буфера
        :param eps_clip: коэффициент клипинга PPO (обычно изменяется в пределах 0.2 до 0.3)
        :param gamma: коэффициент дисконтирования (упор на текущую или будущую выгоду)
        :param lr_actor: learning rate для актера
        :param lr_critic: learning rate для критика
        :param print_freq_mult: множитель частоты печати (множитель к max_ep_len)
        :param log_freq_mult: множитель частоты логирования (множитель к max_ep_len)
        """
        self.env_name = env_name

        if "/" in env_name:
            self.env_name = env_name.replace("/", "-")

        self.max_ep_len = max_ep_len
        self.max_training_timesteps = max_train_timesteps
        self.print_freq = max_ep_len * print_freq_mult
        self.log_freq = max_ep_len * log_freq_mult
        self.save_model_freq = save_model_freq
        self.update_timestep = max_ep_len * update_timestep_mult

        self.k_epochs = k_epochs
        self.eps_clip = eps_clip
        self.gamma = gamma

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        self.env = env
        self.env_count = env_count
        self.buffer = RolloutBuffer()

        # Получаем размерности состояния и действия среды.
        self.state_dim = env.observation_space.shape[1:]
        self.action_dim = env.action_space.nvec[0]

        # Инициалзиация PPO агента.
        self.ppo_agent = PPOAgent(
            self.state_dim,
            self.action_dim,
            self.lr_actor,
            self.lr_critic,
            self.gamma,
            self.k_epochs,
            self.eps_clip,
        )

        self.setup_log()
        self.setup_checkpoints()

    def setup_log(self) -> None:
        """
        Создает директорию и файл логов для текущего запуска.

        P.S. Для каждого запуска содается отдельный лог.
        """
        log_dir = "logs/" + self.env_name + "/"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        current_num_files = next(os.walk(log_dir))[2]
        self.run_num = len(current_num_files)
        self.log_f_name = (
            log_dir + "/PPO_" + self.env_name + "_log_" + str(self.run_num) + ".csv"
        )

    def setup_checkpoints(self) -> None:
        """
        Создает директорию и путь для сохранения модели.
        """
        directory = "checkpoints/" + self.env_name + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.checkpoint_path = directory + "PPO_{}_{}.pth".format(
            self.env_name, self.run_num
        )

    def print_train_parameters(self) -> None:
        """
        Печатает параметры тренировки.
        """
        print(
            "Текущий номер запуска для обучения на среде " + self.env_name + " : ",
            self.run_num,
        )
        print("Количество параллельно запущенных сред:", self.env_count)
        print("Логирование по пути: " + self.log_f_name)
        print("-" * 100)

        print("Путь сохранения модели: " + self.checkpoint_path)
        print("-" * 100)

        print(
            "Максимальное количество шагов всего обучения: ",
            self.max_training_timesteps,
        )
        print("Максимальное количество шагов в эпизоде: ", self.max_ep_len)
        print("Частота сохранения модели: " + str(self.save_model_freq) + " шагов")
        print("Частота логирования: " + str(self.log_freq) + " шагов")
        print(
            "Печать средней награды за эпизоды за последние: "
            + str(self.print_freq)
            + " шагов"
        )
        print("-" * 100)

        print("Размерность пространства состояний: ", self.state_dim)
        print("Размерность пространства действий: ", self.action_dim)
        print("-" * 100)

        print("Частота обновления PPO: " + str(self.update_timestep) + " шагов")
        print("Количество эпох PPO: ", self.k_epochs)
        print("Коэффициент клиппинга PPO: ", self.eps_clip)
        print("Фактор дисконтирования (гамма): ", self.gamma)
        print("-" * 100)

        print("Скорость обучения (actor): ", self.lr_actor)
        print("Скорость обучения (critic): ", self.lr_critic)
        print("-" * 100)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Загружает веса модели из файла.

        :param checkpoint_path: путь к чекпоинту
        """
        self.ppo_agent.load(checkpoint_path)

    def train(self) -> None:
        """
        Запускает процесс обучения агента PPO.
        """
        self.print_train_parameters()

        # Для отслеживания времени обучения
        start_time = datetime.now().replace(microsecond=0)
        print("Начало обучения (GMT): ", start_time)
        print("-" * 100)

        # Открытие файла логов
        log_f = open(self.log_f_name, "w+")
        log_f.write("episode,timestep,reward\n")

        # Переменные для логов и печати
        print_running_reward = 0
        print_running_episodes = 0

        log_running_reward = 0
        log_running_episodes = 0

        i_episode = 0

        # Цикл обучения
        states, _ = self.env.reset()
        for time_step in tqdm(range(self.max_training_timesteps), "Шаг"):
            current_ep_reward = [0, 0, 0, 0]

            # Выбор действия с использованием политики
            actions = []
            actions_logprob = []
            state_vals = []
            states_t = []
            for state in states:
                action, action_logprob, state_val, state_t = (
                    self.ppo_agent.select_action(state)
                )

                actions.append(action)
                actions_logprob.append(action_logprob)
                state_vals.append(state_val)
                states_t.append(state_t)
            states, rewards, terms, truncs, _ = self.env.step(
                [action.item() for action in actions]
            )
            dones = [term or trunc for term, trunc in zip(terms, truncs)]

            # Сохраняем итерацию в буфер для обучения
            self.buffer.states.extend(states_t)
            self.buffer.actions.extend(actions)
            self.buffer.logprobs.extend(actions_logprob)
            self.buffer.state_values.extend(state_vals)
            self.buffer.rewards.extend(rewards)
            self.buffer.is_terminals.extend(dones)

            time_step += 1

            for i in range(self.env_count):
                current_ep_reward[i] += rewards[i]

            # Обновляем агента PPO
            if time_step % self.update_timestep == 0:
                self.ppo_agent.update(self.buffer, self.env_count)
                self.buffer.clear()

            # Логгируем среднюю награду
            if time_step % self.log_freq == 0:
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write("{},{},{}\n".format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # Печать средней награды
            if time_step % self.print_freq == 0:
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print(
                    f"Эпизод : {i_episode} \t\t Шаг : {time_step} \t\t Средняя награда : {print_avg_reward}"
                )

                print_running_reward = 0
                print_running_episodes = 0

            # Сохранение модели
            if time_step % self.save_model_freq == 0:
                print("Сохраняем модель по пути: " + self.checkpoint_path)
                self.ppo_agent.save(self.checkpoint_path)
                print("Модель успешно сохранена")
                print(
                    "Прошедшее время: ",
                    datetime.now().replace(microsecond=0) - start_time,
                )

            for i, done in enumerate(dones):
                # Обновление счётчиков наград
                if done:
                    print_running_reward += current_ep_reward[i]
                    print_running_episodes += 1

                    log_running_reward += current_ep_reward[i]
                    log_running_episodes += 1

                    i_episode += 1

        log_f.close()

        # Завершение обучения
        end_time = datetime.now().replace(microsecond=0)
        print("Начало обучения (GMT): ", start_time)
        print("Окончание обучения (GMT): ", end_time)
        print("Общее время обучения: ", end_time - start_time)

    def test(self, total_test_episodes: int = 10) -> None:
        """
        Тестирует обученного агента в среде.

        :param total_test_episodes: количество тестовых эпизодов
        """
        test_running_reward = 0

        for ep in range(1, total_test_episodes + 1):
            ep_reward = [0 for _ in range(self.env_count)]
            states, _ = self.env.reset()

            for _ in range(1, self.max_ep_len + 1):
                actions = []
                for state in states:
                    action, _, _, _ = self.ppo_agent.select_action(state)
                    actions.append(action.item())

                states, rewards, terms, truncs, _ = self.env.step(actions)
                dones = [term or trunc for term, trunc in zip(terms, truncs)]

                for i in range(self.env_count):
                    ep_reward[i] += rewards[i]

                if any(dones):
                    break

            for i, done in enumerate(dones):
                if done:
                    test_running_reward += ep_reward[i]
                    print(f"Эпизод: {ep} \t\t Средняя награда: {round(ep_reward, 2)}")

                    break

        print("-" * 100)

        avg_test_reward = test_running_reward / (total_test_episodes * self.env_count)
        avg_test_reward = round(avg_test_reward, 2)
        print("Средняя награда в тестировании: " + str(avg_test_reward))
        print("-" * 100)
