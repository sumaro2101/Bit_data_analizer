from os import path
from re import match

from table_config import WHITE, RESET


def validate_file_path(input_str):
    """
    Проверяет, является ли введенная строка путем к существующему файлу.

    Args:
        input_str (str): Введенная пользователем строка

    Returns:
        str or None: Корректный путь к файлу или None
    """
    # Убираем кавычки
    cleaned_path = input_str.strip('"\'').strip()

    # Регулярное выражение для проверки пути к файлу
    path_pattern = r'^([a-zA-Z]:\\|\\\\)(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]+$'

    if match(path_pattern, cleaned_path) and path.exists(cleaned_path):
        return cleaned_path

    return None


def do_formatted_action(action: str) -> str:
    formatted_action = '\n'.join(
        (f"{WHITE}{line.strip()}{RESET}"
         for line
         in action.split('\n'))
        )
    return formatted_action
