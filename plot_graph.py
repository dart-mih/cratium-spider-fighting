import os
import pandas as pd
import matplotlib.pyplot as plt


def save_graph(env_name: str, fig_width: int = 10, fig_height: int = 6) -> None:
    """
    Строит и сохраняет график средней награды по эпизодам из логов.

    :param env_name: название среды (например, 'CartPole-v1')
    :param fig_width: ширина сохраняемой фигуры
    :param fig_height: высота сохраняемой фигуры
    """
    print("-" * 100)

    if "/" in env_name:
        env_name = env_name.replace("/", "-")

    # Задание параметров сглаживания
    window_len_smooth = 20
    min_window_len_smooth = 1
    linewidth_smooth = 1.5
    alpha_smooth = 1

    window_len_var = 5
    min_window_len_var = 1
    linewidth_var = 2
    alpha_var = 0.1

    # Цвета линий
    colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "olive",
        "brown",
        "magenta",
        "cyan",
        "crimson",
        "gray",
        "black",
    ]

    # Создание директории для сохранения графиков
    figures_dir = "graphs"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Создание поддиректории с именем среды
    figures_dir = os.path.join(figures_dir, env_name)
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    fig_save_path = os.path.join(figures_dir, f"PPO_{env_name}_fig.png")

    # Путь к логам
    log_dir = os.path.join("logs", env_name)
    current_num_files = next(os.walk(log_dir))[2]
    num_runs = len(current_num_files)

    all_runs = []

    # Загрузка данных из всех логов
    for run_num in range(num_runs):
        log_f_name = os.path.join(log_dir, f"PPO_{env_name}_log_{run_num}.csv")
        print("Загружается лог-файл: " + log_f_name)
        data = pd.read_csv(log_f_name)
        data = pd.DataFrame(data)

        print("Размер данных: ", data.shape)

        all_runs.append(data)
        print("-" * 100)

    ax = plt.gca()

    # Усреднение данных по всем запускам
    df_concat = pd.concat(all_runs)
    df_concat_groupby = df_concat.groupby(df_concat.index)
    data_avg = df_concat_groupby.mean()

    # Сглаживание наград
    data_avg["reward_smooth"] = (
        data_avg["reward"]
        .rolling(
            window=window_len_smooth,
            win_type="triang",
            min_periods=min_window_len_smooth,
        )
        .mean()
    )
    data_avg["reward_var"] = (
        data_avg["reward"]
        .rolling(
            window=window_len_var, win_type="triang", min_periods=min_window_len_var
        )
        .mean()
    )

    # Построение графиков
    data_avg.plot(
        kind="line",
        x="timestep",
        y="reward_smooth",
        ax=ax,
        color=colors[0],
        linewidth=linewidth_smooth,
        alpha=alpha_smooth,
    )
    data_avg.plot(
        kind="line",
        x="timestep",
        y="reward_var",
        ax=ax,
        color=colors[0],
        linewidth=linewidth_var,
        alpha=alpha_var,
    )

    # Оставить в легенде только сглаженную награду
    handles, _ = ax.get_legend_handles_labels()
    ax.legend([handles[0]], [f"средняя_награда_{len(all_runs)}_запусков"], loc=2)

    ax.grid(color="gray", linestyle="-", linewidth=1, alpha=0.2)
    ax.set_xlabel("Шаги", fontsize=12)
    ax.set_ylabel("Награды", fontsize=12)

    plt.title(env_name, fontsize=14)

    fig = plt.gcf()
    fig.set_size_inches(fig_width, fig_height)

    print("-" * 100)
    plt.savefig(fig_save_path)
    print("График сохранён по пути:", fig_save_path)
    print("-" * 100)


if __name__ == "__main__":
    env_name = "CartPole-v1"

    save_graph(env_name)
