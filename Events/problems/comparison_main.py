"""
Скрипт для сравнения фактических игровых логов с ожидаемыми событиями из чек-листа.
Анализирует последовательность событий, их параметры и выявляет несоответствия.

Основные функции:
- Валидация чек-листа
- Парсинг логов
- Сравнение ожидаемых и фактических событий
- Генерация отчета о несоответствиях
"""

from typing import Dict, List, Optional, Tuple, Any, cast
import re
from dataclasses import dataclass
from datetime import datetime
import webbrowser
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from enum import Enum
import logging
import os
import sys
from tabulate import tabulate
from table_config import (COLUMN_WIDTHS, HEADERS, TABLE_FORMAT, GREEN, RED, BLUE, YELLOW,
                          WHITE, RESET, BOLD, colorize_table_borders)
import time


class ChecklistValidator:
    """
    Класс для валидации данных чек-листа.

    Выполняет всестороннюю проверку DataFrame с данными чек-листа, включая:
    - Наличие всех обязательных столбцов
    - Корректность типов данных
    - Последовательность номеров шагов
    - Формат параметров
    - Отсутствие пустых значений в обязательных полях

    Attributes:
        logger (logging. Logger): Логгер для вывода информационных и отладочных сообщений
        errors (List[str]): Список обнаруженных ошибок при валидации
    """

    # Словарь требуемых столбцов с их типами и допустимостью null-значений
    REQUIRED_COLUMNS = {
        'Номер шага': {'type': 'numeric', 'allow_null': True},
        'Действие': {'type': 'string', 'allow_null': True},
        'Проверяем наличие ивента(ов)': {'type': 'string', 'allow_null': True},
        'Параметры через запятую': {'type': 'string', 'allow_null': True}
    }

    def __init__(self, logger):
        """
        Инициализирует экземпляр ChecklistValidator.

        Args:
            logger (logging. Logger): Логгер для вывода сообщений о валидации
        """
        self.logger = logger
        self.errors: List[str] = []

    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        Выполняет комплексную проверку DataFrame чек-листа.

        Процесс валидации включает несколько последовательных этапов:
        1. Проверка наличия всех необходимых столбцов
        2. Валидация типов данных и пустых значений
        3. Проверка последовательности номеров шагов
        4. Проверка корректности параметров

        Args:
            df (pd. DataFrame): DataFrame для проверки

        Returns:
            bool: True, если чек-лист прошел все проверки, False при наличии ошибок

        Raises:
            ValueError: При критических ошибках в структуре данных
        """
        self.errors = []

        # Проверка наличия всех необходимых столбцов
        missing_columns = set(self.REQUIRED_COLUMNS.keys()) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Отсутствуют обязательные столбцы: {', '.join(missing_columns)}")

        # Проверка типов данных и пустых значений
        for column, rules in self.REQUIRED_COLUMNS.items():
            self._validate_column(df, column, rules)

        # Проверка последовательности номеров шагов
        self._validate_step_numbers(df)

        # Проверка корректности параметров
        self._validate_parameters(df)

        if self.errors:
            self.logger.warning("❌ Чек-лист не прошел валидацию:")
            for error in self.errors:
                self.logger.warning(f"  - {error}")
            return False

        self.logger.info(f"{GREEN}✓ Чек-лист успешно прошел валидацию{RESET}")
        return True

    @staticmethod
    def parse_step_number(step_value: str) -> Optional[int]:
        """
        Преобразует значение номера шага в число, учитывая альтернативные шаги.

        Args:
            step_value: Строка с номером шага (например, '124' или '124, 125 ALT')

        Returns:
            Optional[int]: Номер шага или None для пустых значений
        """
        if pd.isna(step_value):
            return None

        if isinstance(step_value, (int, float)):
            if pd.isna(step_value):  # Дополнительная проверка для float
                return None
            return int(step_value)

        # Если это строка, убираем пробелы и проверяем на ALT
        step_str = str(step_value).strip()
        if not step_str:
            return None

        if 'ALT' in step_str:
            # Берем первый номер из альтернативной последовательности
            return int(step_str.split(',')[0].strip())

        try:
            return int(float(step_str))
        except (ValueError, TypeError):
            return None

    def _validate_column(self, df: DataFrame, column: str, rules: dict) -> None:
        """
        Проверяет отдельный столбец DataFrame на соответствие заданным правилам валидации.

        Метод выполняет комплексную проверку столбца, включая:
        - Анализ наличия пустых значений
        - Проверку соответствия правилам заполнения
        - Идентификацию индексов строк с нарушениями

        Args:
            df (DataFrame): DataFrame, содержащий данные для проверки
            column (str): Название столбца для валидации
            rules (dict): Словарь правил проверки столбца:
                - 'allow_null' (bool): Разрешены ли пустые значения
                - 'type' (str, optional): Ожидаемый тип данных (не используется в текущей реализации)

        Raises:
            Exception: При возникновении ошибок в процессе валидации

        Side Effects:
            При обнаружении ошибок добавляет сообщения в self.errors
        """
        try:
            # Создаем маску для непустых строк
            notna_mask = df.notna().any(axis=1)
            non_empty_rows = df.loc[notna_mask]

            # Проверка на пустые значения только в значимых строках
            if not rules['allow_null']:
                column_data = non_empty_rows[column]
                null_mask = column_data.isnull()
                null_indices = non_empty_rows.index[null_mask].tolist()

                if null_indices:
                    self.errors.append(
                        f"Пустые значения в столбце '{column}' в строках: {null_indices}"
                    )
        except Exception as e:
            self.logger.error(f"Ошибка при валидации столбца {column}: {str(e)}")
            raise

    def _validate_step_numbers(self, df: DataFrame) -> None:
        """
        Проверяет корректность последовательности номеров шагов в чек-листе.

        Метод выполняет следующие проверки:
        - Фильтрует строки с непустыми номерами шагов
        - Преобразует номера шагов с учетом альтернативных путей (ALT)
        - Проверяет правильность последовательности основных шагов (не ALT)
        - Выявляет дублирующиеся номера шагов

        Args:
            df (DataFrame): DataFrame с данными чек-листа для проверки

        Raises:
            Exception: При возникновении ошибок в процессе валидации.
                      Ошибка логируется перед повторным возбуждением.

        Примечание:
            Метод добавляет сообщения об ошибках в self.errors при обнаружении:
            - Нарушений последовательности номеров шагов
            - Дублирующихся номеров шагов (исключая ALT)
        """
        try:
            # Фильтруем только строки с номерами шагов
            steps_df = df[df['Номер шага'].notna()]

            # Преобразуем номера шагов с учетом ALT
            def parse_step(val: Any) -> Optional[int]:
                """
                Преобразует значение шага в целое число с учетом различных форматов и типов данных.

                Алгоритм преобразования:
                1. Проверяет, является ли значение `val` Not a Number (NaN). Если да, возвращает `None`.
                2. Если `val` является целым числом, возвращает его.
                3. Если `val` является числом с плавающей точкой, преобразует его в целое число и возвращает.
                4. Если `val` является строкой:
                   - Удаляет начальные и конечные пробелы.
                   - Если строка содержит 'ALT', извлекает первый номер шага перед 'ALT', преобразует его в целое число и возвращает.
                   - В противном случае, пытается преобразовать строку в число с плавающей точкой, затем в целое число и возвращает.
                5. Если ни одно из вышеперечисленных условий не выполняется, возвращает `None`.

                Args:
                    val (Any): Значение для преобразования. Может быть числом, строкой или содержать 'ALT'

                Returns:
                    Optional[int]: Целое число - номер шага, или None если преобразование невозможно

                Examples:
                    >>> parse_step(5)
                    5
                    >>> parse_step("123, 124 ALT")
                    123
                    >>> parse_step("invalid")
                    None
                    >>> parse_step(np.nan)
                    None
                """
                if pd.isna(val):
                    return None
                if isinstance(val, (int, float, np.integer, np.floating)):
                    return int(val)
                if isinstance(val, (str, np.str_)):
                    val_str = str(val).strip()
                    if 'ALT' in val_str:
                        return int(val_str.split(',')[0].strip())
                    try:
                        return int(float(val_str))
                    except (ValueError, TypeError):
                        return None
                return None

            # Безопасное преобразование с явным приведением к Series
            parsed_steps = pd.Series(steps_df['Номер шага']).apply(parse_step)
            steps_df = steps_df.assign(parsed_step=parsed_steps)

            # Проверяем последовательность только для основных шагов (не ALT)
            main_steps_mask = ~steps_df['Номер шага'].astype(str).str.contains('ALT', na=False)
            main_steps = cast(DataFrame, steps_df[main_steps_mask])

            step_numbers = cast(List[int], main_steps['parsed_step'].dropna().tolist())

            if step_numbers != sorted(step_numbers):
                self.errors.append("Нарушена последовательность номеров шагов")

            # Проверяем дубликаты
            duplicates = cast(DataFrame, main_steps[main_steps.duplicated(['parsed_step'], keep=False)])

            if not duplicates.empty:
                duplicate_steps = cast(List[int], duplicates['parsed_step'].unique().tolist())
                self.errors.append(
                    f"Обнаружены повторяющиеся номера шагов (не ALT): {sorted(duplicate_steps)}"
                )
        except Exception as e:
            self.logger.error(f"Ошибка при валидации номеров шагов: {str(e)}")
            raise

    def _validate_parameters(self, df: DataFrame) -> None:
        """
        Проверяет корректность форматирования параметров в чек-листе.

        Метод выполняет следующие проверки:
        - Фильтрует строки с непустыми параметрами
        - Проверяет корректность разделителя параметров ('|')
        - Валидирует формат каждого параметра (ключ|значение)
        - Проверяет наличие непустых значений в обеих частях параметра

        Args:
            df (DataFrame): DataFrame с данными чек-листа для проверки

        Raises:
            Exception: При возникновении ошибок в процессе валидации.
                      Ошибка логируется перед повторным возбуждением.

        Примечание:
            Метод добавляет сообщения об ошибках в self.errors при обнаружении:
            - Некорректного формата параметров (отсутствие разделителя '|')
            - Пустых значений в любой части параметра
            - Неправильного количества частей параметра (должно быть 2)
        """
        try:
            param_rows = cast(DataFrame, df[df['Параметры через запятую'].notna()])

            for idx, row in enumerate(param_rows.itertuples(), start=1):  # Используем enumerate
                params = row[1]  # Используем номер столбца вместо названия (например, 'Параметры через запятую')
                if not isinstance(params, str):
                    continue

                for param in str(params).split(','):
                    param = param.strip()
                    if param:
                        if '|' in param:
                            parts = param.split('|')
                            if len(parts) != 2 or not all(p.strip() for p in parts):
                                self.errors.append(
                                    f"Некорректный формат параметра '{param}' в строке {idx + 1}"  # `idx` уже число
                                )
        except Exception as e:
            self.logger.error(f"Ошибка при валидации параметров: {str(e)}")
            raise


class DiscrepancyType(Enum):
    """
    Перечисление типов возможных несоответствий между ожидаемыми
    и фактическими событиями.

    Attributes:
        MISSING_EVENT: Событие отсутствует в логах
        PARAMETER_MISMATCH: Параметры события не соответствуют ожидаемым
        WRONG_ORDER: Нарушена последовательность событий
    """
    MISSING_EVENT = "missing_event"
    PARAMETER_MISMATCH = "parameter_mismatch"
    WRONG_ORDER = "wrong_order"


@dataclass
class ExpectedEvent:
    """
    Класс для представления ожидаемого события из чек-листа.

    Содержит полную информацию о событии, которое должно произойти
    в определенной последовательности с заданными параметрами.

    Attributes:
        step_number (Any): Номер шага, к которому относится событие
        name (str): Название события
        parameters (Dict[str, Optional[str]]): Словарь параметров события
        action (str): Описание действия для события
        flexible_order (bool): Флаг возможности произвольного порядка события
        order_group (int): Группа для событий с гибким порядком
        is_optional: bool = False  # Флаг, указывающий, является ли событие опциональным
    """
    step_number: Any
    name: str
    parameters: Dict[str, Optional[str]]  # None означает, что параметр должен существовать без конкретного значения
    action: str
    flexible_order: bool = False  # Флаг для событий с произвольным порядком
    order_group: int = 0  # Группа событий с произвольным порядком
    is_optional: bool = False  # Флаг, является ли событие опциональным

    def __hash__(self):
        """
        Генерирует хеш объекта для использования в словарях и множествах.

        Returns:
            int: Хеш-значение объекта
        """
        return hash((self.step_number, self.name, tuple(sorted(self.parameters.items())),
                     self.action, self.flexible_order, self.order_group, self.is_optional))

    def __eq__(self, other):
        """
        Проверяет равенство двух объектов ExpectedEvent.

        Args:
            other: Объект для сравнения

        Returns:
            bool: True, если объекты эквивалентны, иначе False
        """
        if not isinstance(other, ExpectedEvent):
            return False
        return (self.step_number == other.step_number and
                self.name == other.name and
                self.parameters == other.parameters and
                self.action == other.action and
                self.flexible_order == other.flexible_order and
                self.order_group == other.order_group and
                self.is_optional == other.is_optional)


@dataclass
class ActualEvent:
    """
    Класс для представления фактического события из логов.

    Содержит полную информацию о событии, зафиксированном в логах,
    включая его название, параметры и временную метку.

    Attributes:
        name (str): Название события
        parameters (Dict[str, str]): Словарь параметров события
        timestamp (datetime): Временная метка события
    """
    name: str
    parameters: Dict[str, str]
    timestamp: datetime


class LogAnalyzer:
    """
    Основной класс для анализа соответствия фактических логов ожидаемым событиям.

    Выполняет полный цикл анализа:
    - Чтение и валидация чек-листа
    - Чтение и парсинг файла логов
    - Сравнение ожидаемых и фактических событий
    - Генерация отчета о несоответствиях

    Attributes:
        checklist_path (str): Путь к файлу чек-листа
        log_path (str): Путь к файлу логов
        expected_events (List[ExpectedEvent]): Список ожидаемых событий
        actual_events (List[ActualEvent]): Список фактических событий
        logger (logging.Logger): Логгер для вывода информационных сообщений
    """

    # Список событий, которые нужно игнорировать при анализе
    IGNORED_EVENTS = {'QuestCompleted', 'EventAppeared'}

    def __init__(self, log_path: str):
        """
        Инициализирует экземпляр LogAnalyzer.

        Выполняет следующие действия:
        1. Сохраняет пути к файлам чек-листа и логов
        2. Инициализирует списки ожидаемых и фактических событий
        3. Проверяет наличие файла логов
        4. Настраивает логирование (файловый и консольный обработчики)

        Args:
            log_path (str): Путь к файлу логов

        Raises:
            FileNotFoundError: Если файл логов не найден по указанному пути
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.checklist_path = os.path.join(current_dir, 'checklist', 'User_story.xlsx')
        self.log_path = log_path
        self.expected_events: List[ExpectedEvent] = []
        self.actual_events: List[ActualEvent] = []
        self.user_id: Optional[str] = None

        # Проверяем существование файла логов
        if not os.path.exists(self.log_path):
            raise FileNotFoundError(f"Файл лога не найден по пути: {self.log_path}")

        # Сначала очищаем все существующие обработчики
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Настройка логирования
        log_format = '%(asctime)s - %(levelname)s - %(message)s'

        # Создаем файловый обработчик
        file_handler = logging.FileHandler('analysis.log', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format))

        # Создаем консольный обработчик
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_format)

        # Настраиваем корневой логгер
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)

        self.logger = logging.getLogger(__name__)

    def generate_html_report(self, table_data: List[List[str]], headers: List[str]) -> None:
        """
        Генерирует HTML отчет на основе данных таблицы.

        Args:
            table_data (List[List[str]]): Двумерный список с данными для таблицы
            headers (List[str]): Список заголовков для столбцов таблицы

        Raises:
            Exception: При ошибках в процессе генерации отчета
        """
        try:
            if not table_data:
                self.logger.warning("Нет данных для создания HTML отчета")
                return

            # Очищаем ANSI-коды и создаем данные для DataFrame
            df_data = []
            for row in table_data:
                clean_row = []
                for idx, cell in enumerate(row):
                    # Очищаем от ANSI-кодов
                    clean_cell = re.sub(r'\033\[\d+m', '', str(cell))
                    clean_cell = clean_cell.replace('\n', '<br>')

                    if idx == 3:
                        if clean_cell == "Нет ожидаемых параметров":
                            clean_cell = f'<span class="text-success">{clean_cell}</span>'
                    if idx == 4:
                        if clean_cell == "✓":
                            clean_cell = f'<span class="text-success">✓</span>'
                        elif clean_cell == "✗":
                            clean_cell = f'<span class="text-danger">✗</span>'
                        elif clean_cell == "Ивент не найден":
                            clean_cell = f'<span class="text-danger">Ивент не найден</span>'
                    elif idx == 5:
                        if clean_cell:
                            clean_cell = f'<span class="text-danger">{clean_cell}</span>'

                    clean_row.append(clean_cell)
                df_data.append(clean_row)

            clean_headers = [re.sub(r'\033\[\d+m', '', str(header)) for header in headers]
            df = pd.DataFrame(df_data)
            df.columns = clean_headers

            html_template = '''<!DOCTYPE html>
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
                .table th:nth-child(4) {{ width: 35% }}
                .table th:nth-child(5) {{ width: 5% }}
                .table th:nth-child(6) {{ width: 25% }}
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

            table_html = df.to_html(
                classes='table table-striped table-bordered',
                escape=False,
                index=False
            )

            table_html = table_html.replace('<td>', '<td class="parameter-cell">')
            current_dir = os.path.dirname(os.path.abspath(__file__))
            report_path = os.path.join(current_dir, 'HTMLreport.html')

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_template.format(
                    table_html=table_html,
                    user_id=self.user_id if self.user_id else "Не найден"
                ))

        except Exception as e:
            self.logger.error(f"Ошибка при создании HTML отчета: {str(e)}")
            raise

    @staticmethod
    def _check_event_match(expected: ExpectedEvent, actual: ActualEvent) -> Tuple[bool, List[str]]:
        """
        Проверяет соответствие между ожидаемым и фактическим событием.
        """
        names_match = (actual.name.lower() == expected.name.lower() or
                       actual.name.lower().replace('_', '') == expected.name.lower().replace('_', ''))

        if not names_match:
            return False, []

        mismatches = []
        actual_params_lower = {k.lower(): k for k in actual.parameters}

        # Проверяем наличие и значения ожидаемых параметров
        for param_name, expected_value in expected.parameters.items():
            param_name_lower = param_name.lower()
            if param_name_lower not in actual_params_lower:
                mismatches.append(f"Отсутствует параметр: {param_name}")
            else:
                actual_param_name = actual_params_lower[param_name_lower]
                if expected_value is not None:
                    actual_value = actual.parameters[actual_param_name]
                    if actual_value != expected_value:
                        mismatches.append(
                            f"Несоответствие параметра {param_name}: "
                            f"ожидалось '{expected_value}', получено '{actual_value}'"
                        )

        # Проверяем наличие лишних параметров
        for actual_param in actual.parameters:
            actual_param_lower = actual_param.lower()
            if not any(exp_param.lower() == actual_param_lower for exp_param in expected.parameters):
                mismatches.append(f"Лишний параметр: {actual_param}")

        # Возвращаем True только если имена совпадают, иначе всегда False
        # Список несоответствий будет пустым только если все параметры соответствуют ожидаемым
        is_match = True  # Если имена совпали, считаем события совпадающими, даже если есть несоответствия в параметрах
        return is_match, mismatches

    def _get_group_events(self, current_group: int) -> List[ExpectedEvent]:
        """
        Возвращает список событий, принадлежащих указанной группе гибкого порядка.

        Args:
            current_group (int): Номер группы гибкого порядка

        Returns:
            List[ExpectedEvent]: Список событий в указанной группе
        """
        return [event for event in self.expected_events if event.order_group == current_group]

    @staticmethod
    def is_not_na(value) -> bool:
        """
        Безопасно преобразует различные типы в булево значение, указывающее на отсутствие NA (Not Available).

        Args:
            value: Входное значение для проверки

        Returns:
            bool: True, если значение не является NA, иначе False
        """
        # Обработка Pandas Series или NumPy массивов
        if isinstance(value, (pd.Series, np.ndarray)):
            # Преобразуем в Python bool с использованием .any() или .all() в зависимости от контекста
            return bool(value.any()) if value is not None else False

        # Обработка других типов с использованием Pandas' notna()
        return bool(pd.notna(value))

    def parse_checklist(self) -> None:
        """
        Читает и парсит файл чек-листа для подготовки к анализу.

        Выполняет следующие ключевые операции:
        1. Чтение Excel-файла чек-листа
        2. Фильтрация и очистка данных
        3. Обработка номеров шагов и действий
        4. Извлечение ожидаемых событий с их параметрами
        5. Подготовка списка событий для последующего анализа

        Метод обрабатывает сложные случаи:
        - Альтернативные пути (ALT)
        - Пропущенные значения
        - Гибкий порядок событий

        Raises:
            Exception: При ошибках чтения или обработки чек-листа
        """
        try:
            self.logger.info(f"{WHITE}Начинаем чтение чек-листа из Excel{RESET}")

            # Читаем Excel файл
            df_check = pd.read_excel(self.checklist_path)
            self.logger.debug(
                f"Первые несколько строк DataFrame после чтения:\n{df_check.head()}")  # Отладка: вывод первых строк DataFrame

            # Создаем копию DataFrame для обработки
            df_check_filtered = df_check.copy()

            # Безопасное преобразование номеров шагов
            def parse_step_safely(val):
                """
                Внутренняя функция для безопасного преобразования номеров шагов.
                """
                try:
                    if pd.isna(val):
                        return None
                    # Преобразование для строк с ALT
                    if isinstance(val, str) and 'ALT' in val:
                        return val
                    # Преобразование числовых значений
                    return int(float(str(val)))
                except (ValueError, TypeError):
                    return None

            df_check_filtered['Номер шага'] = df_check['Номер шага'].apply(parse_step_safely)

            # Сохраняем текущий номер шага для строк, где он пустой
            last_valid_step = None
            last_valid_action = None

            # Заполняем пропущенные номера шагов и действия предыдущими значениями
            for idx, row in df_check_filtered.iterrows():
                # Безопасное получение значения шага
                step_value = row['Номер шага']

                # Проверка на NA с использованием улучшенного метода
                if self.is_not_na(step_value):
                    if isinstance(step_value, (pd.Series, np.ndarray)):
                        step_value = step_value.item()  # Извлекаем скалярное значение

                    last_valid_step = step_value
                    last_valid_action = row['Действие']
                else:
                    # Заполнение пустых значений последним валидным шагом
                    df_check_filtered.at[idx, 'Номер шага'] = last_valid_step

                    # Заполнение действия, если оно отсутствует
                    if bool(pd.isna(row['Действие'])):
                        df_check_filtered.at[idx, 'Действие'] = last_valid_action

            # Фильтруем только строки, где есть события
            def is_valid_event(val):
                """
                Проверяет, является ли значение валидным событием.
                """
                # Проверка, что значение не является NA и не является пустым/служебным
                return (pd.notna(val) and
                        str(val).strip() not in ['ПРЕДУСЛОВИЯ', '-', '', 'nan'])

            df_check_filtered = df_check_filtered[
                df_check_filtered['Проверяем наличие ивента(ов)'].apply(is_valid_event)
            ]

            # Заменяем оригинальный DataFrame на отфильтрованный
            df_check = df_check_filtered

            current_step = None
            current_action = None
            current_order_group = 0
            flexible_group_active = False  # Флаг, показывающий, активна ли текущая группа гибкого порядка

            # Сбрасываем список ожидаемых событий
            self.expected_events = []

            # Проходим по всем строкам
            for index, row in df_check.iterrows():
                event_value = row['Проверяем наличие ивента(ов)']
                self.logger.debug(f"Raw event value from Excel (row {index}): '{event_value}'")

                # Безопасное получение значений
                def safe_str_convert(val) -> str:
                    """
                    Безопасно преобразует значение в строку, обрабатывая пустые значения.
                    """
                    return str(val).strip() if pd.notna(val) else ""

                step_value = row['Номер шага']
                event_value = safe_str_convert(row['Проверяем наличие ивента(ов)'])
                self.logger.debug(
                    f"Raw event value from Excel (row {index}): '{event_value}'")  # Отладка: вывод значения ячейки "Проверяем наличие ивента(ов)"
                action_value = safe_str_convert(row['Действие'])
                params_value = safe_str_convert(row['Параметры через запятую'])

                # Обновляем текущий шаг и действие, если есть номер шага
                if self.is_not_na(step_value):
                    if isinstance(step_value, (pd.Series, np.ndarray)):
                        step_value = step_value.item()

                    if isinstance(step_value, str) and 'ALT' in step_value:
                        current_step = step_value
                        current_action = f"{action_value} (Альтернативный путь)"
                    else:
                        current_step = step_value
                        current_action = action_value

                # Определяем тип гибкого порядка
                is_optional = False  # Добавлено: Инициализация флага опциональности
                if '[?]' in event_value:  # Проверяем наличие метки опциональности
                    is_optional = True
                    event_value = event_value.replace('[?]', '').strip()  # Убираем метку из имени

                is_flexible = False
                current_event_str_raw_lines = event_value.split('\n')
                for current_event_str_raw in current_event_str_raw_lines:
                    current_event_str_trimmed = current_event_str_raw.strip()
                    if not current_event_str_trimmed:
                        continue

                    current_event_str = current_event_str_trimmed

                    if '[↓]' in current_event_str:
                        is_flexible = True
                        flexible_group_active = True  # Активируем группу гибкого порядка
                        current_event_str = current_event_str.replace('[↓]', '').strip()
                        current_order_group += 1
                    elif '[↑]' in current_event_str:
                        is_flexible = True
                        current_event_str = current_event_str.replace('[↑]', '').strip()
                        if '[↓]' not in event_value:  # Если это не начало, это отдельное гибкое событие или конец последовательности
                            if not flexible_group_active:
                                current_order_group += 1
                            flexible_group_active = False  # Считаем группу завершенной

                    self.logger.debug(
                        f"Processing event: '{current_event_str}', is_flexible: {is_flexible}, group: {current_order_group}")  # Отладка: вывод имени события и флага гибкости

                    # Создаем событие
                    event = ExpectedEvent(
                        step_number=current_step,
                        name=current_event_str,
                        parameters=self._extract_parameters(params_value),
                        action=current_action or "",
                        flexible_order=is_flexible,
                        order_group=current_order_group if is_flexible else 0,
                        is_optional=is_optional  # Устанавливаем флаг опциональности
                    )

                    # Добавляем событие в список
                    self.expected_events.append(event)

            self.logger.info(f"{WHITE}Успешно обработано {len(self.expected_events)} ожидаемых событий{RESET}")

        except Exception as e:
            self.logger.error(f"Ошибка при чтении чек-листа: {str(e)}")
            raise

    def parse_logs(self) -> None:
        """
        Читает и парсит файл логов для последующего анализа.

        Выполняет детальный разбор текстового файла логов с извлечением:
        - UserID из первой строки
        - Названий событий
        - Параметров событий
        - Временных меток

        Ключевые особенности парсинга:
        - Обработка многострочных параметров
        - Восстановление контекста событий
        - Обработка особых форматов логирования

        Raises:
            Exception: При ошибках чтения или парсинга файла логов
        """
        try:
            self.logger.info(f"{WHITE}Начинаем чтение лог-файла{RESET}")
            with open(self.log_path, 'r', encoding='utf-8') as f:
                # Читаем первую строку для получения UserID
                first_line = f.readline().strip()
                if "UserID:" in first_line:
                    self.user_id = first_line.split("UserID:")[1].strip()

                # Перемещаемся в начало файла для полного парсинга
                f.seek(0)

                current_event = None
                current_time = None
                current_params = {}
                current_param_line = ""

                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Если строка начинается с даты, это новое событие
                    if re.match(r'\d{2}\.\d{2}\.\d{4}', line):
                        # Сохраняем предыдущее событие, если оно есть
                        if current_event is not None:
                            # Проверяем, не находится ли событие в списке игнорируемых
                            if current_event not in self.IGNORED_EVENTS:
                                if current_param_line:
                                    self._process_param_line(current_param_line, current_params)

                                event = ActualEvent(
                                    name=current_event,
                                    parameters=current_params.copy(),
                                    timestamp=current_time or datetime.now()
                                )
                                self.actual_events.append(event)
                                self.logger.debug(f"Добавлено событие: {current_event}")

                        # Парсим новое событие
                        parts = line.split('|')
                        if len(parts) >= 2:
                            current_time = datetime.strptime(parts[0].strip(), '%d.%m.%Y %H:%M:%S')
                            event_part = parts[1].strip()

                            if 'EventName:' in event_part:
                                current_event = event_part.split('EventName:')[1].strip()
                            else:
                                current_event = event_part.strip()

                            current_params = {}
                            current_param_line = ""

                    # Обрабатываем строки с параметрами
                    elif line.startswith('['):
                        current_param_line += line.strip()

                # Добавляем последнее событие, если оно не в списке игнорируемых
                if current_event is not None and current_event not in self.IGNORED_EVENTS:
                    if current_param_line:
                        self._process_param_line(current_param_line, current_params)

                    event = ActualEvent(
                        name=current_event,
                        parameters=current_params.copy(),
                        timestamp=current_time or datetime.now()
                    )
                    self.actual_events.append(event)

            self.logger.info(f"{WHITE}Успешно прочитано {len(self.actual_events)} фактических событий{RESET}")

        except Exception as e:
            self.logger.error(f"Ошибка при чтении лог-файла: {str(e)}")
            raise

    def _process_param_line(self, param_line: str, params: dict) -> None:
        """
        Обрабатывает строку с параметрами и добавляет их в словарь параметров.

        Выполняет разбор строки параметров с учетом специфического формата:
        - Извлечение параметров в квадратных скобках
        - Обработка параметров с разделителем '|'
        - Безопасное добавление параметров в словарь

        Args:
            param_line (str): Строка с параметрами
            params (dict): Словарь параметров для обновления

        Raises:
            Exception: При ошибках разбора параметров
        """
        try:
            pattern = r'\[(.*?)\]'
            param_matches = re.findall(pattern, param_line)

            for param_str in param_matches:
                if '|' in param_str:
                    key, value = param_str.split('|', 1)
                    key = key.strip()
                    value = value.strip()
                    params[key] = value
                    self.logger.debug(f"Обработан параметр: {key} = {value}")
                else:
                    self.logger.warning(f"Пропущен некорректный параметр: {param_str}")

        except Exception as e:
            self.logger.error(f"Ошибка при обработке параметров: {str(e)}")
            self.logger.error(f"Исходная строка: {param_line}")
            raise

    def _extract_parameters(self, params_str: str) -> Dict[str, Optional[str]]:
        """
        Извлекает параметры из строки параметров чек-листа.

        Обрабатывает различные форматы входных параметров:
        - Параметры с разделителем '|'
        - Параметры без значения
        - Обработка пустых и NA значений

        Args:
            params_str (str): Строка параметров для извлечения

        Returns:
            Dict[str, Optional[str]]: Словарь извлеченных параметров
        """
        parameters = {}

        def safe_isna(val) -> bool:
            """
            Безопасно проверяет, является ли значение NA (Not Available).
            Обрабатывает различные типы данных, включая pandas.Series.

            Args:
                val: Значение для проверки

            Returns:
                bool: True если значение является NA, иначе False
            """
            result = pd.isna(val)
            return result.item() if hasattr(result, 'item') else result

        if not params_str or (isinstance(params_str, pd.Series) and safe_isna(params_str)):
            return parameters

        param_items = [p.strip() for p in str(params_str).split(',')]

        for item in param_items:
            if '|' in item:
                parts = item.split('|', 1)
                if len(parts) == 2:
                    key, value = parts
                    parameters[key.strip()] = value.strip()
            else:
                if item.strip():
                    parameters[item.strip()] = None

        self.logger.debug(f"Извлеченные параметры: {parameters}")
        return parameters

    @property
    def _do_compare_events(self) -> Tuple[List[Dict], List[List[str]], List[List[str]]]:
        """
        Выполняет сравнение фактических событий с ожидаемыми на основе данных чек-листа.

        Метод проходит по шагам, определенным в чек-листе, и сравнивает ожидаемые события
        с фактическими событиями из логов. Учитывает порядок событий, а также события
        с гибким порядком. Формирует отчет о расхождениях и подготавливает данные для
        консольного и HTML отчетов.

        Returns:
            Tuple[List[Dict], List[List[str]], List[List[str]]]:
                - Список словарей, содержащих информацию о расхождениях.
                - Список строк для консольной таблицы.
                - Список строк для HTML таблицы.

        Raises:
            Exception: В случае возникновения ошибок в процессе сравнения.
        """
        discrepancies = []
        console_table_data: List[List[str]] = []
        html_table_data: List[List[str]] = []
        used_events = set()
        unexpected_events_count = 0
        actual_event_index = 0  # Отслеживаем текущий индекс в списке фактических событий

        try:
            # Загрузка данных из чек-листа для информации о шагах
            df_check = pd.read_excel(self.checklist_path)
            df_check['Номер шага'] = pd.to_numeric(df_check['Номер шага'], errors='coerce')

            # Словарь для информации о шагах
            steps_info = {}
            current_step = None

            # Заполняем информацию о шагах
            for _, row in df_check.iterrows():
                step_value = row['Номер шага']

                def safe_notna(val) -> bool:
                    result = pd.notna(val)
                    return result.item() if hasattr(result, 'item') else result

                if safe_notna(step_value):
                    current_step = int(step_value)
                    action_value = row['Действие']
                    current_action = str(action_value) if safe_notna(action_value) else ""
                    if current_step not in steps_info:
                        steps_info[current_step] = {
                            'action': current_action,
                            'has_events': False
                        }

                if current_step is not None:
                    event_value = row['Проверяем наличие ивента(ов)']
                    event_str = str(event_value) if safe_notna(event_value) else ""
                    if event_str and event_str != '-' and event_str != 'nan':
                        steps_info[current_step]['has_events'] = True

            # Группируем ожидаемые события по шагам
            step_events = {}
            for event in self.expected_events:
                if event.step_number not in step_events:
                    step_events[event.step_number] = []
                step_events[event.step_number].append(event)

            # Обрабатываем каждый шаг по порядку
            for step_number in sorted(steps_info.keys()):
                step_info = steps_info[step_number]
                formatted_action = '\n'.join(f"{WHITE}{line.strip()}{RESET}"
                                             for line in step_info['action'].split('\n'))

                if not step_info['has_events']:
                    console_row = [
                        f"{WHITE}{str(step_number)}{RESET}",
                        formatted_action,
                        f"{GREEN}Нет ожидаемых ивентов{RESET}",
                        f"{GREEN}Нет ожидаемых параметров{RESET}",
                        f"{GREEN}✓{RESET}",
                        f"{WHITE}{RESET}"
                    ]
                    html_row = [
                        str(step_number),
                        step_info['action'],
                        "Нет ожидаемых ивентов",
                        "Нет ожидаемых параметров",
                        "✓",
                        ""
                    ]
                    console_table_data.append(console_row)
                    html_table_data.append(html_row)
                    continue

                if step_number not in step_events:
                    continue

                first_in_step = True
                events = step_events[step_number]

                # Обрабатываем группы событий с гибким порядком
                processed_expecteds = set()  # Отслеживаем уже обработанные ожидаемые события
                for expected in events:
                    if expected.flexible_order and expected not in processed_expecteds:
                        flexible_group = [e for e in events if e.order_group == expected.order_group]

                        # Пытаемся сопоставить все события в гибкой группе
                        for exp_flex in flexible_group:
                            best_match_flex = None
                            best_mismatches_flex = []
                            matched_actual_index_flex = -1

                            for i in range(len(self.actual_events)):
                                if i not in used_events and self.actual_events[i].name not in self.IGNORED_EVENTS:
                                    matches, mismatches = self._check_event_match(exp_flex, self.actual_events[i])
                                    if matches:
                                        best_match_flex = self.actual_events[i]
                                        best_mismatches_flex = mismatches
                                        matched_actual_index_flex = i
                                        break

                            console_row, html_row = self._add_event_to_table(
                                step_number,
                                formatted_action if first_in_step else "",
                                exp_flex,
                                first_in_step,
                                best_match_flex is not None,
                                best_match_flex,
                                best_mismatches_flex,
                                is_flexible=True  # Указываем, что событие относится к гибкому порядку
                            )
                            console_table_data.append(console_row)
                            html_table_data.append(html_row)
                            first_in_step = False

                            if best_match_flex:
                                used_events.add(matched_actual_index_flex)
                                self._add_discrepancy(
                                    discrepancies,
                                    step_number,
                                    step_info['action'],
                                    exp_flex,
                                    True,
                                    best_match_flex,
                                    best_mismatches_flex
                                )
                            else:
                                self._add_discrepancy(
                                    discrepancies,
                                    step_number,
                                    step_info['action'],
                                    exp_flex,
                                    False,
                                    None,
                                    []
                                )
                            processed_expecteds.add(exp_flex)

                    elif not expected.flexible_order:
                        found_match = False
                        best_match = None
                        best_mismatches = []
                        potential_unexpected_events = []

                        for i in range(actual_event_index, len(self.actual_events)):
                            actual = self.actual_events[i]
                            if i not in used_events and actual.name not in self.IGNORED_EVENTS:
                                matches, mismatches = self._check_event_match(expected, actual)
                                if matches and not expected.is_optional:  # Проверяем, что событие не опциональное
                                    found_match = True
                                    best_match = actual
                                    best_mismatches = mismatches
                                    used_events.add(i)
                                    actual_event_index = i + 1
                                    break
                                elif matches and expected.is_optional:
                                    found_match = True
                                    best_match = actual
                                    best_mismatches = mismatches
                                    used_events.add(i)
                                    break
                                else:
                                    # Если не соответствует текущему ожидаемому, добавляем в список неожиданных
                                    potential_unexpected_events.append(actual)

                        # Обработка неожиданных событий перед выводом текущего ожидаемого
                        if potential_unexpected_events and not expected.is_optional:  # Добавлено условие
                            for unexpected_event in potential_unexpected_events:
                                console_row, html_row = self._add_event_to_table(
                                    step_number,
                                    formatted_action if first_in_step else "",
                                    expected,  # передаем expected для структуры
                                    first_in_step,
                                    True,  # found_match=True для неожиданных событий
                                    unexpected_event,
                                    [],  # пустой список несоответствий
                                    is_unexpected=True  # помечаем как неожиданное событие
                                )
                                console_table_data.append(console_row)
                                html_table_data.append(html_row)
                                unexpected_events_count += 1

                                discrepancies.append({
                                    'step': step_number,
                                    'action': step_info['action'],
                                    'event': unexpected_event.name,
                                    'type': 'unexpected_event',
                                    'details': [f"Неожиданный ивент с параметрами: {unexpected_event.parameters}"],
                                    'timestamp': unexpected_event.timestamp
                                })
                        # Логика добавления в таблицы и дикрепанси
                        console_row, html_row = self._add_event_to_table(
                            step_number,
                            formatted_action,
                            expected,
                            first_in_step,
                            found_match,
                            best_match,
                            best_mismatches,
                            is_flexible=expected.flexible_order
                        )
                        console_table_data.append(console_row)
                        html_table_data.append(html_row)

                        if not found_match or best_mismatches:
                            self._add_discrepancy(
                                discrepancies,
                                step_number,
                                step_info['action'],
                                expected,
                                found_match,
                                best_match,
                                best_mismatches
                            )

                        first_in_step = False

            # Выводим UserID перед таблицей
            if self.user_id:
                self.logger.info(f"\n{BOLD}UserID: {WHITE}{self.user_id}{RESET}\n")

            # Применяем форматирование к таблице
            headers = [f"{BOLD}{header}{RESET}" for header in HEADERS]
            table_output = tabulate(console_table_data, headers, tablefmt=TABLE_FORMAT)
            colored_table = colorize_table_borders(table_output)
            self.logger.info(colored_table)

            return discrepancies, console_table_data, html_table_data

        except Exception as e:
            self.logger.error(f"Ошибка в процессе сравнения: {str(e)}")
            raise

    @staticmethod
    def _add_event_to_table(step_number: int, formatted_action: str,
                            expected: ExpectedEvent, first_in_step: bool, found_match: bool,
                            best_match: Optional[ActualEvent], best_mismatches: List[str],
                            is_unexpected: bool = False, is_flexible: bool = False) -> Tuple[List[str], List[str]]:
        """
        Формирует строки данных для консольной и HTML таблиц на основе сравнения событий.

        Метод определяет, как будет отображаться информация о каждом ожидаемом событии
        в таблицах отчетов. Учитывает, было ли найдено соответствие в логах,
        есть ли расхождения в параметрах, является ли событие опциональным или
        относится к группе с гибким порядком.

        Args:
            step_number (int): Номер шага в чек-листе.
            formatted_action (str): Отформатированное описание действия для шага.
            expected (ExpectedEvent): Объект ожидаемого события.
            first_in_step (bool): Флаг, указывающий, является ли событие первым для данного шага.
            found_match (bool): Флаг, указывающий, было ли найдено соответствие события в логах.
            best_match (Optional[ActualEvent]): Объект наилучшего соответствия фактического события (если есть).
            best_mismatches (List[str]): Список расхождений в параметрах.
            is_unexpected (bool): Флаг, указывающий, является ли событие неожиданным.
            is_flexible (bool): Флаг, указывающий, относится ли событие к группе с гибким порядком.

        Returns:
            Tuple[List[str], List[str]]: Кортеж, содержащий строки для консольной и HTML таблиц.
        """

        def get_visible_length(s: str) -> int:
            """Вычисляет видимую длину строки без учета ANSI-кодов форматирования."""
            return len(re.sub(r'\033\[\d+m', '', s))

        def truncate_parameter(text_to_truncate: str, max_length: int) -> str:
            """Усекает строку параметра до указанной максимальной длины с сохранением форматирования."""
            visible_length = get_visible_length(text_to_truncate)
            if visible_length > max_length:
                visible_text = re.sub(r'\033\[\d+m', '', text_to_truncate)
                truncated_text = visible_text[:max_length - 3] + "..."
                has_red = RED in text_to_truncate
                has_white = WHITE in text_to_truncate
                current_color = RED if has_red else (WHITE if has_white else "")
                return f"{current_color}{truncated_text}{RESET}"
            return text_to_truncate

        # Обработка неожиданного события
        if is_unexpected and best_match:
            params_str = ', '.join([f'{k}={v}' for k, v in best_match.parameters.items()])
            console_params = truncate_parameter(f"{RED}{params_str}{RESET}",
                                                COLUMN_WIDTHS['MAX_PARAMETER_LINE_LENGTH'])

            console_row = [
                "" if not first_in_step else f"{WHITE}{str(step_number)}{RESET}",
                "" if not first_in_step else formatted_action,
                f"{RED}{best_match.name}{RESET}",
                console_params,
                f"{RED}✗{RESET}",
                f"{RED}Неожиданный ивент{RESET}"
            ]

            html_row = [
                str(step_number) if first_in_step else "",
                formatted_action.replace(WHITE, "").replace(RESET, "") if first_in_step else "",
                f'<span class="text-danger">{best_match.name}</span>',
                f'<span class="text-danger">{params_str}</span>',
                "✗",
                "Неожиданный ивент"
            ]
            return console_row, html_row

        # Получаем список проблемных параметров
        missing_params = set()
        mismatch_params = set()
        for msg in best_mismatches:
            if "Отсутствует параметр:" in msg:
                param = msg.split("Отсутствует параметр:")[1].strip()
                missing_params.add(param)
            elif "Несоответствие параметра" in msg:
                param = msg.split("Несоответствие параметра ")[1].split(":")[0].strip()
                mismatch_params.add(param)

        # Форматируем параметры для консоли и HTML
        console_params = []
        html_params = []

        if found_match and best_match:
            # Форматируем параметры для найденного события
            all_params = []
            for param_name, actual_value in best_match.parameters.items():
                if param_name in expected.parameters:
                    if param_name in mismatch_params or param_name in missing_params:
                        param_str = f"{WHITE}{param_name}={RED}{actual_value}{RESET}"
                    else:
                        param_str = f"{WHITE}{param_name}={actual_value}{RESET}"
                else:
                    param_str = f"{RED}{param_name}={actual_value}{RESET}"

                all_params.append(truncate_parameter(param_str,
                                                     COLUMN_WIDTHS['MAX_PARAMETER_LINE_LENGTH']))

            # Проверяем общую длину
            total_length = sum(get_visible_length(p) + 1 for p in all_params)
            if total_length > COLUMN_WIDTHS['parameters']:
                truncated_params = []
                current_length = 0
                for param in all_params:
                    param_length = get_visible_length(param) + 1
                    if current_length + param_length > COLUMN_WIDTHS['parameters']:
                        truncated_params.append(f"{WHITE}...{RESET}")
                        break
                    truncated_params.append(param)
                    current_length += param_length
                console_params = truncated_params
            else:
                console_params = all_params

            # Форматируем параметры для HTML
            for param_name, actual_value in best_match.parameters.items():
                if param_name in expected.parameters:
                    if param_name in mismatch_params or param_name in missing_params:
                        html_params.append(
                            f'{param_name}=<span class="parameter-value-mismatch">{actual_value}</span>'
                        )
                    else:
                        html_params.append(f"{param_name}={actual_value}")
                else:
                    html_params.append(
                        f'<span class="text-danger">{param_name}={actual_value}</span>'
                    )
        else:
            # Для ненайденного события параметры в зависимости от опциональности
            for param_name, expected_value in expected.parameters.items():
                value_str = expected_value if expected_value is not None else ''
                # Используем жёлтый цвет для параметров опционального события
                color = YELLOW if expected.is_optional else RED
                param_str = f"{color}{param_name}={value_str}{RESET}"
                console_params.append(truncate_parameter(param_str,
                                                         COLUMN_WIDTHS['MAX_PARAMETER_LINE_LENGTH']))
                html_color_class = "text-warning" if expected.is_optional else "text-danger"
                html_params.append(f'<span class="{html_color_class}">{param_name}={value_str}</span>')

        # Форматируем имя события и маркеры
        event_name = expected.name
        flexible_marker = f" [{BLUE}↕{WHITE}]" if is_flexible else ""
        html_flexible_marker = ' <span style="color: #6495ED;">[↕]</span>' if is_flexible else ""

        optional_marker = ""
        html_optional_marker = ""
        status_color = RED
        if expected.is_optional:
            optional_marker = f" [{YELLOW}?{WHITE}]"
            html_optional_marker = ' <span style="color: #CCCC00;">[?]</span>'
            if found_match:
                status_color = GREEN if not best_mismatches else RED
            else:
                status_color = YELLOW
        elif found_match:
            status_color = GREEN if not best_mismatches else RED

        status_text = '✓' if (expected.is_optional and found_match and not best_mismatches) or (not expected.is_optional and found_match and not best_mismatches) else \
                      '✓' if (expected.is_optional and not found_match) else '✗' if not found_match else '✗'

        differences = list(best_mismatches)

        # Обработка опциональных событий
        if expected.is_optional and not found_match:
            differences = [f"{YELLOW}Опциональный ивент не найден, но это нормально{RESET}"]
        elif not found_match:
            status_text = 'Ивент не найден'

        # Создаем строки для консоли
        console_row = [
            f"{WHITE}{str(step_number)}{RESET}" if first_in_step else "",
            formatted_action if first_in_step else "",
            f"{WHITE}{event_name}{flexible_marker}{optional_marker}{RESET}",
            f"{WHITE}\n{RESET}".join(console_params),
            f"{status_color}{status_text}{RESET}",
            f"{RED}{chr(10).join(differences)[:COLUMN_WIDTHS['differences']]}{RESET}" if differences else ""
        ]

        # Создаем строки для HTML
        html_row = [
            str(step_number) if first_in_step else "",
            formatted_action.replace(WHITE, "").replace(RESET, "") if first_in_step else "",
            f"{event_name}{html_flexible_marker}{html_optional_marker}",
            "<br>".join(html_params),
            # Упрощенное определение класса для статуса
            f'<span class="{"text-warning" if expected.is_optional and not found_match else "text-success" if status_color == GREEN else "text-danger"}">{status_text}</span>',
            "<br>".join([
                f'<span class="text-warning">{d.replace(YELLOW, "").replace(RED, "").replace(RESET, "")}</span>'
                if "Опциональный ивент не найден" in d
                else f'<span class="text-danger">{d.replace(RED, "").replace(RESET, "")}</span>'
                for d in differences
            ]) if differences else ""
        ]

        return console_row, html_row

    @staticmethod
    def _add_discrepancy(discrepancies: List[Dict], step_number: int, action: str,
                         expected: ExpectedEvent, found_match: bool, best_match: Optional[ActualEvent],
                         best_mismatches: List[str]) -> None:
        """
        Добавляет информацию о несоответствии между ожидаемым и фактическим событием в список расхождений.

        Метод формирует словарь, описывающий расхождение, и добавляет его в общий список.
        Фиксируются такие аспекты, как номер шага, описание действия, название события,
        тип расхождения (отсутствие события или несоответствие параметров) и детали.

        Args:
            discrepancies (List[Dict]): Список, в который добавляется информация о расхождениях.
            step_number (int): Номер шага, на котором произошло расхождение.
            action (str): Описание действия, выполняемого на данном шаге.
            expected (ExpectedEvent): Объект ожидаемого события.
            found_match (bool): Флаг, указывающий, было ли найдено соответствующее событие в логах.
            best_match (Optional[ActualEvent]): Объект наилучшего соответствия фактического события (если есть).
            best_mismatches (List[str]): Список деталей несоответствия параметров.
        """
        if not found_match:
            if not expected.is_optional:  # Добавлено условие: не опциональный
                discrepancy_type = DiscrepancyType.MISSING_EVENT.value
                details = ['Ивент не найден']
                discrepancies.append({
                    'step': step_number,
                    'action': action,
                    'event': expected.name,
                    'type': discrepancy_type,
                    'details': details,
                    'timestamp': best_match.timestamp if best_match else None
                })
        else:
            if best_mismatches:  # Добавляем несоответствие только если есть best_mismatches
                discrepancy_type = DiscrepancyType.PARAMETER_MISMATCH.value
                details = best_mismatches
                discrepancies.append({
                    'step': step_number,
                    'action': action,
                    'event': expected.name,
                    'type': discrepancy_type,
                    'details': details,
                    'timestamp': best_match.timestamp if best_match else None
                })

    def analyze(self) -> None:
        """Выполняет полный анализ логов и выводит результаты."""
        self.logger.info(f"{WHITE}Начинаем анализ...{RESET}")

        # Парсим чек-лист и логи
        self.parse_checklist()
        self.parse_logs()

        # Сравниваем события и получаем несоответствия и данные таблиц
        discrepancies, _, html_table_data = self._do_compare_events

        # Генерируем HTML отчет с полными данными
        try:
            self.generate_html_report(html_table_data, HEADERS)
            self.logger.info(
                f"\n{GREEN}HTML отчет сохранен в файл: {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'HTMLreport.html')}{RESET}")
        except Exception as e:
            self.logger.error(f"Ошибка при создании HTML отчета: {str(e)}")

        # Разделяем несоответствия по типам
        if len(discrepancies) == 0:
            self.logger.info(f"\n{GREEN}✓ Все события соответствуют ожиданиям!{RESET}")
        else:
            param_mismatches = [d for d in discrepancies if d['type'] == DiscrepancyType.PARAMETER_MISMATCH.value]
            missing_events = [d for d in discrepancies if d['type'] == DiscrepancyType.MISSING_EVENT.value]
            unexpected_events = [d for d in discrepancies if d['type'] == 'unexpected_event']

            # Выводим общую статистику
            self.logger.warning(f"\n{RED}❌ Найдено несоответствий:{RESET}")
            self.logger.warning(f"  - Отсутствующих событий: {len(missing_events)}")
            self.logger.warning(f"  - Количество шагов с несоответствием параметров: {len(param_mismatches)}")
            if unexpected_events:
                self.logger.warning(f"  - Неожиданных ивентов: {len(unexpected_events)}")

            # Выводим информацию о каждом несоответствии параметров
            for disc in param_mismatches:
                self.logger.warning(f"\nШаг {disc['step']} - {disc['event']}")
                self.logger.warning(f"Действие: {disc['action']}")
                self.logger.warning(f"Тип: parameter_mismatch")
                if disc['timestamp']:
                    self.logger.warning(f"Временная метка: {disc['timestamp']}")
                self.logger.warning("Детали:")
                for detail in disc['details']:
                    self.logger.warning(f"  - {detail}")

            # Выводим информацию о неожиданных событиях
            if unexpected_events:
                self.logger.warning("\nНеожиданные события:")
                for disc in unexpected_events:
                    self.logger.warning(f"\nШаг {disc['step']} - {disc['event']}")
                    if disc['timestamp']:
                        self.logger.warning(f"Временная метка: {disc['timestamp']}")
                    for detail in disc['details']:
                        self.logger.warning(f"  {detail}")

            # В конце выводим список шагов с отсутствующими событиями
            if missing_events:
                missing_steps = sorted(list(set(d['step'] for d in missing_events)))
                self.logger.warning(f"\nНе найдены ивенты в шагах: {', '.join(map(str, missing_steps))}")


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

    if re.match(path_pattern, cleaned_path) and os.path.exists(cleaned_path):
        return cleaned_path

    return None


def main():
    """
    Главная функция скрипта для анализа соответствия игровых логов чек-листу.

    Выполняет следующие действия:
    1. Проверка наличия файла чек-листа
    2. Настройка логирования
    3. Валидация чек-листа
    4. Запуск основного цикла работы программы:
       - Запрос пути к файлу логов
       - Анализ указанного файла логов
       - Вывод меню дальнейших действий
       - Обработка выбора пользователя (открытие HTML отчета, анализ другого файла, выход)

    Скрипт обрабатывает возможные ошибки и исключения на всех этапах выполнения,
    включая проверку наличия файлов, валидацию чек-листа, анализ логов,
    некорректный ввод пользователя и другие исключительные ситуации.

    При возникновении критических ошибок выполнение скрипта прерывается
    с выводом соответствующего сообщения и кода завершения.

    В случае успешного выполнения скрипт завершается с кодом 0.
    """

    def print_menu():
        """
        Выводит меню действий с гарантированным отображением и двойной проверкой.

        Функция выполняет следующие шаги:
        1. Очищает буфер вывода для гарантированного отображения меню
        2. Выводит разделитель и заголовок меню
        3. Выводит пункты меню с вариантами действий
        4. Использует принудительный сброс буфера вывода и задержки для корректного отображения
        5. Выводит приглашение к вводу выбора пользователя

        Меню предоставляет пользователю следующие варианты:
        - Просмотр HTML отчета
        - Обработка другого файла логов
        - Выход из программы
        """
        # Очищаем буфер вывода
        sys.stdout.flush()

        # Первый вывод разделителя
        print("\n" + "=" * 50)
        time.sleep(0.2)  # Небольшая задержка после разделителя

        # Основное меню с промежуточными задержками
        print("Выберите дальнейшее действие:")
        sys.stdout.flush()
        time.sleep(0.2)

        print("-" * 50)
        print("1. Просмотреть HTML отчёт - нажмите 'y'")
        print("2. Обработать другие логи - введите путь")
        print("3. Выйти из программы - нажмите 'q'")
        print("=" * 50)

        # Финальный сброс и задержка
        sys.stdout.flush()
        time.sleep(0.3)

        # Приглашение к вводу с гарантированным отображением
        print("Ваш выбор: ", end='')
        sys.stdout.flush()
        time.sleep(0.3)  # Финальная задержка перед вводом

    try:
        # Проверяем чек-лист
        current_dir = os.path.dirname(os.path.abspath(__file__))
        checklist_path = os.path.join(current_dir, 'checklist', 'User_story.xlsx')

        if not os.path.exists(checklist_path):
            print(f"Ошибка: файл чек-листа не найден по пути: {checklist_path}")
            sys.exit(1)

        # Настраиваем логирование
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        logger = logging.getLogger(__name__)

        # Валидируем чек-лист
        try:
            df_check = pd.read_excel(checklist_path)
            validator = ChecklistValidator(logger)
            if not validator.validate_dataframe(df_check):
                sys.exit(1)
        except Exception as e:
            print(f"Ошибка при валидации чек-листа: {str(e)}")
            sys.exit(1)

        # Основной цикл работы программы
        while True:
            # Запрос пути к логам
            time.sleep(0.3)  # Задержка перед запросом
            print("\nПожалуйста введите путь к файлу с очищенными логами (или 'q' для выхода):")
            log_path = input().strip()
            log_path = log_path.strip('"\'')  # Удаляем кавычки из ввода

            if log_path.lower() == 'q':
                print("Выход из программы...")
                sys.exit(0)

            if os.path.exists(log_path):
                # Попытка анализа логов
                try:
                    # Анализ логов
                    analyzer = LogAnalyzer(log_path)
                    analyzer.analyze()

                    # Меню дальнейших действий
                    while True:
                        print_menu()
                        choice = input().strip()  # Убрали вывод приглашения из input()
                        choice = choice.strip('"\'')

                        # Если введен существующий путь - начинаем анализ нового файла
                        if os.path.exists(choice):
                            log_path = choice
                            try:
                                # Сразу запускаем анализ новых логов
                                analyzer = LogAnalyzer(log_path)
                                analyzer.analyze()
                                continue  # Продолжаем внутренний цикл с меню выбора действий
                            except Exception as e:
                                print(f"Ошибка при анализе логов: {str(e)}")
                            continue  # В случае ошибки тоже остаемся во внутреннем цикле

                        # Просмотр HTML-отчета
                        elif choice.lower() == 'y':
                            html_report_path = os.path.join(current_dir, 'HTMLreport.html')
                            if os.path.exists(html_report_path):
                                webbrowser.open_new_tab(html_report_path)
                                print("HTML отчёт открыт в браузере.")
                            else:
                                print("HTML отчёт не найден. Проверьте, был ли он успешно создан.")
                            continue

                        # Выход из программы
                        elif choice.lower() == 'q':
                            print("Выход из программы...")
                            sys.exit(0)

                        # Некорректный ввод
                        else:
                            print("Некорректный ввод. Пожалуйста, введите 'y', 'q' или путь к логам.")

                except Exception as e:
                    print(f"Ошибка при анализе логов: {str(e)}")
            else:
                print(f"Файл не найден: {log_path}\nПроверьте путь и попробуйте снова.")

    except FileNotFoundError as e:
        print(f"Ошибка: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Произошла неожиданная ошибка: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
