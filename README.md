# cratium-spider-fighting

_Поддерживается обучения только в средах с дискретными пространствами действий_

1. Для запуска обучения используется скрипт main.py

- Необходимо заменить env_name на ту среду, на которой должна обучаться модель.
- Описание параметров обучения см. в реализации init класса PPO.

2. Для построения графиков используется plot_graph.py

- Необходимо заменить env_name на ту среду, на которой должна обучаться модель.
- Он строит график по всем запускам модели на определенной среде.

3. Для запуска визуализации работы обученной модели в среде используется vizualize.py

- Необходимо заменить env_name на ту среду, на которой должна обучаться модель.
- Необходимо отредактировать путь к весам обученной модели.
