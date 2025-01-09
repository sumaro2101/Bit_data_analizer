# ANSI коды для цветов
GREEN = '\033[92m'
RED = '\033[91m'
CYAN = '\033[96m'
WHITE = '\033[97m'
RESET = '\033[0m'
BOLD = '\033[1m'
BLUE = '\033[94m'
YELLOW = '\033[93m'

# Символы для определения границ таблицы
BORDER_CHARS = '═║╒╓╔╕╖╗╘╙╚╛╜╝╞╟╠╡╢╣╤╥╦╧╨╩╪╫╬─│┌┐└┘├┤┬┴┼'

# Максимальная длина для каждой колонки
COLUMN_WIDTHS = {
    'step': 3,  # Номер шага
    'action': 100,  # Действие
    'event_name': 40,  # Имя ивента
    'parameters': 50000,  # Максимальная длина параметров
    "MAX_PARAMETER_LINE_LENGTH": 60,  # Максимальная длина строки параметров
    'result': 1,  # Результат проверки
    'differences': 300  # Различия
}

# Заголовки колонок
HEADERS = ["Шаг", "Действие", "Имя ивента", "Параметры", "Результат проверки", "Различия с логами"]

# Формат таблицы (можно изменить на другой поддерживаемый tabulate)
TABLE_FORMAT = "fancy_grid"

# Цвет таблицы (ANSI-код)
TABLE_COLOR = '\033[96m'  # Бирюзовый для всей таблицы

# Жирный шрифт для заголовков (ANSI-код)
HEADER_STYLE = '\033[1m'


# Функция для раскрашивания рамки таблицы
def colorize_table_borders(table_output: str) -> str:
    """
    Раскрашивает рамку таблицы в бирюзовый цвет.

    Args:
        table_output: исходная таблица

    Returns:
        таблица с бирюзовыми границами
    """
    colored_table = ""
    for char in table_output:
        if char in BORDER_CHARS:
            colored_table += f"{CYAN}{char}{RESET}"
        else:
            colored_table += char
    return colored_table
