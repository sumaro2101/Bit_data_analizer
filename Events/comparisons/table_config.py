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
    'differences': 300,  # Различия
    'timestamp': 20  # Временная метка
}

# Заголовки колонок
HEADERS = ["Шаг", "Действие", "Имя ивента",
           "Параметры", "Результат проверки",
           "Различия с логами", "Временная метка",
           ]

IGNORED_EVENTS = {'QuestCompleted', 'EventAppeared'}

HTML_TEMPLATE = '''<!DOCTYPE html>
                <html>
                <head>
                <meta charset="utf-8">
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
                <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
                <style>
                .table-container {{ margin: 20px }}
                .table {{ table-layout: fixed; width: 100% }}
                .table td {{ white-space: normal !important; word-wrap: break-word; vertical-align: top }}
                .parameter-cell {{ overflow-x: auto; max-height: none !important; white-space: normal !important }}
                .table th:nth-child(1) {{ width: 5% }}
                .table th:nth-child(2) {{ width: 15% }}
                .table th:nth-child(3) {{ width: 15% }}
                .table th:nth-child(4) {{ width: 30% }}
                .table th:nth-child(5) {{ width: 5% }}
                .table th:nth-child(6) {{ width: 20% }}
                .table th:nth-child(7) {{ width: 10% }}
                .text-success {{ color: #3cb371 !important; font-weight: bold }}
                .text-danger {{ color: #dc3545 !important; font-weight: bold }}
                .text-warning {{ color: #CCCC00 !important; font-weight: bold }}  <!-- Добавлен стиль для желтого -->
                .parameter-value-mismatch {{ color: #dc3545 !important; font-weight: bold }}
                .table th {{ text-align: left; background-color: #f8f9fa; padding: 10px }}
                .user-id {{ color: #0d6efd; font-size: 1.5em; padding: 10px }}
                </style>
                </head>
                <body>
                <div class="table-container">
                <div class="user-id">UserID: {user_id}</div>
                {table_html}
                </div>
                </body>
                </html>'''

# Формат таблицы (можно изменить на другой поддерживаемый tabulate)
TABLE_FORMAT = "fancy_grid"

# Цвет таблицы (ANSI-код)
TABLE_COLOR = '\033[96m'  # Бирюзовый для всей таблицы

# Жирный шрифт для заголовков (ANSI-код)
HEADER_STYLE = '\033[1m'

NAME_CHECK_LIST = 'User_story.xlsx'

TIME_MASK = '%d.%m.%Y %H:%M:%S'


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
