import os
import shutil


def delete_minetest_run_folders(root_dir):
    """
    Удаляет папки прошлых запусков minetest.

    :param root_dir: Путь к директории, в которой нужно искать и удалять папки
    """
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path) and item.startswith("minetest-run-"):
            try:
                shutil.rmtree(item_path)
                print(f"Успешно удалено: {item_path}")
            except Exception as e:
                print(f"Ошибка при удалении {item_path}: {e}")
