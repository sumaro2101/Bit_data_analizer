"""
Скрипт для сравнения фактических игровых логов с ожидаемыми событиями из чек-листа.
Анализирует последовательность событий, их параметры и выявляет несоответствия.

Основные функции:
- Валидация чек-листа
- Парсинг логов
- Сравнение ожидаемых и фактических событий
- Генерация отчета о несоответствиях
"""

from typing import Dict, List, Optional, Tuple, Any
import re
from datetime import datetime
import webbrowser
import numpy as np
import pandas as pd
import logging
import os
import sys
from tabulate import tabulate
from table_config import (COLUMN_WIDTHS, HEADERS, TABLE_FORMAT, GREEN, RED, BLUE, YELLOW,
                          WHITE, RESET, BOLD, colorize_table_borders)
import time

from dto import StepTimeInfo, ExpectedEvent, ActualEvent
from enums import DiscrepancyType
from checkers import ChecklistValidator


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
        self.steps_info = {}  # Словарь для хранения информации о шагах
        self.timeline: Dict[int, StepTimeInfo] = {}
        self.step_events = {}  # Добавляем инициализацию step_events

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

    def _analyze_event_timeline(self) -> Dict[int, StepTimeInfo]:
        """
        Анализирует временную последовательность событий и создает временную карту шагов.
        """
        timeline = {}
        actual_events_map = {}  # datetime -> List[ActualEvent]

        # Сначала группируем все фактические события по времени
        for event in self.actual_events:
            if event.timestamp not in actual_events_map:
                actual_events_map[event.timestamp] = []
            actual_events_map[event.timestamp].append(event)

        # Сортируем временные метки
        sorted_timestamps = sorted(actual_events_map.keys())

        # Инициализируем информацию для каждого шага
        for step in self.step_events.keys():
            timeline[step] = StepTimeInfo(step_number=step)

        # Проходим по событиям в хронологическом порядке
        current_step = None
        for timestamp in sorted_timestamps:
            events = actual_events_map[timestamp]
            for event in events:
                # Ищем шаг, которому может принадлежать это событие
                matching_step = None
                for step, step_events in self.step_events.items():
                    if timeline[step].first_event_time is not None:
                        continue  # Пропускаем шаги, для которых уже нашли события

                    for expected_event in step_events:
                        if (event.name.lower() == expected_event.name.lower() or
                           event.name.lower().replace('_', '') == expected_event.name.lower().replace('_', '')):
                            matching_step = step
                            break
                    if matching_step is not None:
                        break

                if matching_step is not None:
                    timeline[matching_step].update_times(timestamp)
                    timeline[matching_step].processed_events.append(event.name)

        return timeline

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
                            clean_cell = '<span class="text-success">✓</span>'
                        elif clean_cell == "✗":
                            clean_cell = '<span class="text-danger">✗</span>'
                        elif clean_cell == "Ивент не найден":
                            clean_cell = '<span class="text-danger">Ивент не найден</span>'
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

        Args:
            expected: Ожидаемое событие из чек-листа
            actual: Фактическое событие из логов

        Returns:
            Tuple[bool, List[str]]: (событие найдено, список несоответствий в параметрах)
        """
        # Проверяем совпадение имен событий
        names_match = (actual.name.lower() == expected.name.lower() or
                       actual.name.lower().replace('_', '') == expected.name.lower().replace('_', ''))

        if not names_match:
            return False, []

        mismatches = []
        actual_params_lower = {k.lower(): k for k in actual.parameters}

        # Проверяем критические параметры, которые определяют идентичность события
        # Это параметры, без которых событие считается другим экземпляром
        critical_params = {k.lower(): v for k, v in expected.parameters.items()
                           if v is not None}  # Параметры с конкретными значениями считаем критическими

        # Проверяем критические параметры
        for param_name, expected_value in critical_params.items():
            if param_name not in actual_params_lower:
                return False, [f"Отсутствует критический параметр: {param_name}"]

            actual_param_name = actual_params_lower[param_name]
            if actual.parameters[actual_param_name] != expected_value:
                return False, [
                    f"Несоответствие критического параметра {param_name}: "
                    f"ожидалось '{expected_value}', получено '{actual.parameters[actual_param_name]}'"
                ]

        # Если критические параметры совпали, проверяем остальные параметры
        # Проверяем наличие ожидаемых параметров
        for param_name, expected_value in expected.parameters.items():
            param_name_lower = param_name.lower()
            if expected_value is None:  # Некритические параметры
                if param_name_lower not in actual_params_lower:
                    mismatches.append(f"Отсутствует параметр: {param_name}")

        # Проверяем наличие лишних параметров
        for actual_param in actual.parameters:
            actual_param_lower = actual_param.lower()
            if not any(exp_param.lower() == actual_param_lower
                       for exp_param in expected.parameters):
                mismatches.append(f"Лишний параметр: {actual_param}")

        # Возвращаем True, так как критические параметры совпали
        return True, mismatches

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

    def _find_best_event_match(self, expected: ExpectedEvent, current_step: int) -> Tuple[
        Optional[ActualEvent], bool, List[str]]:
        """
        Ищет наилучшее соответствие для ожидаемого события с учетом временной карты.
        """
        best_match = None
        best_mismatches = []
        is_truly_missing = True

        # Получаем временную информацию для текущего шага
        step_info = self.timeline.get(current_step)
        if not step_info or not step_info.first_event_time:
            return None, True, []

        # Находим все события с таким же именем
        matching_events = []
        for actual in self.actual_events:
            if (actual.name.lower() == expected.name.lower() or
                    actual.name.lower().replace('_', '') == expected.name.lower().replace('_', '')):
                # Проверяем, попадает ли событие во временной интервал шага
                if step_info.first_event_time <= actual.timestamp <= step_info.last_event_time:
                    matching_events.append(actual)

        if not matching_events:
            return None, True, []

        # Ищем наилучшее соответствие среди событий в правильном временном интервале
        min_mismatches = float('inf')
        best_timestamp = None

        for actual in matching_events:
            matches, mismatches = self._check_event_match(expected, actual)
            if matches:
                if len(mismatches) < min_mismatches or (
                        len(mismatches) == min_mismatches and
                        best_timestamp and actual.timestamp < best_timestamp
                ):
                    min_mismatches = len(mismatches)
                    best_match = actual
                    best_mismatches = mismatches
                    best_timestamp = actual.timestamp
                    is_truly_missing = False

        return best_match, is_truly_missing, best_mismatches

    def _get_step_time_bounds(self, current_step: int) -> Optional[Tuple[datetime, datetime]]:
        """
        Определяет временные границы для шага на основе первого события в текущем
        и следующем шаге.
        """
        sorted_steps = sorted(list(self.step_events.keys()))

        try:
            current_idx = sorted_steps.index(current_step)
        except ValueError:
            return None

        # Находим первое событие текущего шага
        current_step_time = None
        current_events = []
        for event in self.step_events[current_step]:
            matches = [e for e in self.actual_events
                       if (e.name.lower() == event.name.lower() or
                           e.name.lower().replace('_', '') == event.name.lower().replace('_', ''))]
            current_events.extend(matches)

        if current_events:
            current_step_time = min(e.timestamp for e in current_events)

        # Находим время первого события следующего шага
        next_step_time = None
        if current_idx < len(sorted_steps) - 1:
            next_step = sorted_steps[current_idx + 1]
            next_events = []
            for event in self.step_events[next_step]:
                matches = [e for e in self.actual_events
                           if (e.name.lower() == event.name.lower() or
                               e.name.lower().replace('_', '') == event.name.lower().replace('_', ''))]
                next_events.extend(matches)

            if next_events:
                next_step_time = min(e.timestamp for e in next_events)

        # Если не нашли времени для текущего шага, считаем шаг пропущенным
        if not current_step_time:
            return None

        # Определяем границы
        step_start_time = current_step_time
        step_end_time = next_step_time if next_step_time else datetime.max

        return step_start_time, step_end_time

    @staticmethod
    def _check_parameters(expected: ExpectedEvent, actual: ActualEvent) -> List[str]:
        """
        Проверяет соответствие параметров между ожидаемым и фактическим событием.
        """
        mismatches = []
        actual_params_lower = {k.lower(): k for k in actual.parameters}

        # Проверяем каждый ожидаемый параметр
        for param_name, expected_value in expected.parameters.items():
            param_name_lower = param_name.lower()
            param_found = param_name_lower in actual_params_lower

            if not param_found:
                # Параметр отсутствует
                mismatches.append(f"Отсутствует параметр: {param_name}")
            else:
                # Параметр найден, проверяем значение если оно задано
                actual_param_name = actual_params_lower[param_name_lower]
                actual_value = actual.parameters[actual_param_name]

                if expected_value is not None and actual_value.strip() != expected_value.strip():
                    mismatches.append(
                        f"Несоответствие параметра {param_name}: "
                        f"ожидалось '{expected_value}', получено '{actual_value}'"
                    )

        # Проверяем наличие лишних параметров
        for actual_param in actual.parameters:
            actual_param_lower = actual_param.lower()
            if not any(exp_param.lower() == actual_param_lower
                       for exp_param in expected.parameters):
                mismatches.append(f"Лишний параметр: {actual_param}")

        return mismatches

    @property
    def _do_compare_events(self) -> Tuple[List[Dict], List[List[str]], List[List[str]]]:
        """
        Сравнивает фактические события с ожидаемыми на основе заданных критериев.

        Метод выполняет следующие операции:
        - Инициализация структур данных для отслеживания соответствий и расхождений.
        - Обход DataFrame чек-листа для сбора информации по шагам.
        - Сбор всех ExpectedEvent в словарь для быстрого доступа.
        - Получение первых временных меток для каждого шага.
        - Основной цикл сравнения для сопоставления ожидаемых и фактических событий.
          - Вызов `_find_best_event_match` для поиска наилучшего соответствия в логах.
          - Обновление списка использованных индексов для исключения повторного сопоставления.
          - Сохранение всех результатов сравнения (с расхождениями и без них) в таблицы для отчета.
        - Обработка неожиданных событий, которые не указаны в чек-листе.
        - Формирование таблиц отчетов с форматированием для консоли и HTML.

        Returns:
            Tuple[List[Dict], List[List[str]], List[List[str]]]:
                - List[Dict]: Список словарей, описывающих расхождения между ожидаемыми и фактическими событиями.
                - List[List[str]]: Список строк для консольной таблицы с форматированием.
                - List[List[str]]: Список строк для HTML таблицы с форматированием.

        Raises:
            Exception: При возникновении ошибок в процессе сравнения событий.
        """
        discrepancies = []
        console_table_data: List[List[str]] = []
        html_table_data: List[List[str]] = []
        used_event_indices = set()
        missed_events = set()
        try:
            df_check = pd.read_excel(self.checklist_path)
            df_check['Номер шага'] = pd.to_numeric(df_check['Номер шага'], errors='coerce')
            steps_info = {}
            current_step = None

            # Сначала собираем базовую информацию о шагах
            for _, row in df_check.iterrows():
                step_value = row['Номер шага']

                def safe_notna(val):
                    return pd.notna(val) and (not hasattr(val, 'item') or bool(val.item()))

                if safe_notna(step_value):
                    current_step = int(step_value)
                    action_value = row['Действие']
                    current_action = str(action_value) if safe_notna(action_value) else ""
                    if current_step not in steps_info:
                        steps_info[current_step] = {'action': current_action, 'has_events': False}
                if current_step is not None:
                    event_value = row['Проверяем наличие ивента(ов)']
                    event_str = str(event_value) if safe_notna(event_value) else ""
                    if event_str and event_str != '-' and event_str != 'nan':
                        steps_info[current_step]['has_events'] = True

            # Создаем словарь событий по шагам для быстрого доступа
            self.step_events = {}
            for event in self.expected_events:
                if event.step_number not in self.step_events:
                    self.step_events[event.step_number] = []
                self.step_events[event.step_number].append(event)

            # Добавляем информацию о временных метках первых событий для каждого шага
            for step_number, events in self.step_events.items():
                if events and step_number in steps_info:
                    step_events = [e for e in self.actual_events if e.name in [ev.name for ev in events]]
                    if step_events:
                        steps_info[step_number]['first_event_time'] = min(e.timestamp for e in step_events)

            # Основной цикл сравнения
            for step_number in sorted(steps_info.keys()):
                step_info = steps_info[step_number]
                formatted_action = '\n'.join(
                    f"{WHITE}{line.strip()}{RESET}" for line in step_info['action'].split('\n'))
                expected_events_for_step = self.step_events.get(step_number, [])
                first_in_step = True

                if not expected_events_for_step and step_info['has_events']:
                    console_row = self._create_no_expected_events_row(step_number, formatted_action)
                    html_row = self._create_no_expected_events_html_row(step_number, step_info['action'])
                    console_table_data.append(console_row)
                    html_table_data.append(html_row)
                    continue
                elif not step_info['has_events']:
                    console_row = self._create_no_events_expected_row(step_number, formatted_action)
                    html_row = self._create_no_events_expected_html_row(step_number, step_info['action'])
                    console_table_data.append(console_row)
                    html_table_data.append(html_row)
                    continue

                step_completed = False  # Флаг, показывающий, был ли шаг выполнен
                last_event_time = None
                for expected in expected_events_for_step:

                    # Ищем подходящее событие среди неиспользованных
                    best_match = None
                    best_match_idx = None
                    best_mismatches = []
                    min_mismatches = float('inf')
                    best_timestamp = None

                    # Проверяем, есть ли такое же событие в пропущенных шагах
                    event_in_missed_steps = expected.name.lower() in missed_events
                    if not event_in_missed_steps:
                        # Итерируемся по всем неиспользованным событиям
                        for idx, actual in enumerate(self.actual_events):
                            if idx in used_event_indices:
                                continue

                            if (actual.name.lower() == expected.name.lower() or
                                    actual.name.lower().replace('_', '') == expected.name.lower().replace('_', '')):
                                mismatches = self._check_parameters(expected, actual)
                                # Проверяем временную метку
                                if steps_info[step_number].get('first_event_time', None):
                                    if (not last_event_time or actual.timestamp >= last_event_time) and (
                                            actual.timestamp >= steps_info[step_number]['first_event_time']):
                                        if len(mismatches) < min_mismatches or (
                                                len(mismatches) == min_mismatches and
                                                best_timestamp and actual.timestamp < best_timestamp
                                        ):
                                            min_mismatches = len(mismatches)
                                            best_match = actual
                                            best_match_idx = idx
                                            best_mismatches = mismatches
                                            best_timestamp = actual.timestamp
                                else:
                                    if (not last_event_time or actual.timestamp >= last_event_time) and (
                                            len(mismatches) < min_mismatches or (
                                            len(mismatches) == min_mismatches and
                                            best_timestamp and actual.timestamp < best_timestamp
                                    )):
                                        min_mismatches = len(mismatches)
                                        best_match = actual
                                        best_match_idx = idx
                                        best_mismatches = mismatches
                                        best_timestamp = actual.timestamp

                    # Если нашли событие и оно не из пропущенных шагов
                    if best_match is not None and not event_in_missed_steps:
                        used_event_indices.add(best_match_idx)
                        step_completed = True
                        last_event_time = best_match.timestamp  # запоминаем время последнего найденного события для проверки порядка
                        console_row, html_row = self._add_event_to_table(
                            step_number, formatted_action, expected, first_in_step,
                            True, best_match, best_mismatches, is_flexible=expected.flexible_order,
                            timestamp=best_match.timestamp
                        )
                        self._add_discrepancy(
                            discrepancies,
                            step_number,
                            step_info['action'],
                            expected,
                            True,
                            best_match,
                            best_mismatches
                        )
                    else:
                        # Если событие не найдено или находится в пропущенных шагах
                        console_row, html_row = self._add_event_to_table(
                            step_number, formatted_action, expected, first_in_step,
                            False, None, ["Ивент не найден"], is_flexible=expected.flexible_order, timestamp=None
                        )
                        self._add_discrepancy(
                            discrepancies,
                            step_number,
                            step_info['action'],
                            expected,
                            False,
                            None,
                            ["Ивент не найден"]
                        )
                        # Добавляем событие в множество пропущенных
                        missed_events.add(expected.name.lower())

                    console_table_data.append(console_row)
                    html_table_data.append(html_row)
                    first_in_step = False

                # Если шаг не был выполнен, добавляем все его события в пропущенные
                if not step_completed:
                    for expected in expected_events_for_step:
                        missed_events.add(expected.name.lower())

            # Обработка неожиданных событий
            steps = sorted(steps_info.keys())
            for idx, actual in enumerate(self.actual_events):
                if idx not in used_event_indices and actual.name not in self.IGNORED_EVENTS:
                    # Ищем ближайший предыдущий шаг
                    closest_step_number = None
                    for step_idx, step in enumerate(steps):
                        # Для последнего шага или если это последнее событие
                        if step_idx == len(steps) - 1:
                            closest_step_number = step
                            break

                        current_step = step
                        next_step = steps[step_idx + 1]
                        # Проверяем, попадает ли временная метка события между текущим и следующим шагом
                        current_time = steps_info[current_step].get('first_event_time', None)
                        next_time = steps_info[next_step].get('first_event_time', None)

                        if current_time and next_time:
                            if current_time <= actual.timestamp < next_time:
                                closest_step_number = current_step
                                break
                        elif current_time:
                            closest_step_number = current_step
                            break

                    if closest_step_number is None and steps:
                        closest_step_number = steps[-1]

                    closest_step_info = steps_info.get(closest_step_number)
                    formatted_action_unexpected = '\n'.join(
                        f"{WHITE}{line.strip()}{RESET}" for line in
                        closest_step_info['action'].split('\n')) if closest_step_info else ""

                    console_row, html_row = self._add_event_to_table(
                        closest_step_number if closest_step_number is not None else "-",
                        formatted_action_unexpected,
                        ExpectedEvent(step_number="-", name=actual.name, parameters={}, action=""),
                        True, True, actual, [], is_unexpected=True, timestamp=actual.timestamp
                    )
                    console_table_data.append(console_row)
                    html_table_data.append(html_row)
                    discrepancies.append({
                        'step': closest_step_number,
                        'action': closest_step_info['action'] if closest_step_info else "",
                        'event': actual.name,
                        'type': 'unexpected_event',
                        'details': [f"Неожиданный ивент с параметрами: {actual.parameters}"],
                        'timestamp': actual.timestamp
                    })

            if self.user_id:
                self.logger.info(f"\n{BOLD}UserID: {WHITE}{self.user_id}{RESET}\n")

            headers = [f"{BOLD}{header}{RESET}" for header in HEADERS]
            table_output = tabulate(console_table_data, headers, tablefmt=TABLE_FORMAT)
            colored_table = colorize_table_borders(table_output)
            self.logger.info(colored_table)

            return discrepancies, console_table_data, html_table_data
        except Exception as e:
            self.logger.error(f"Ошибка в процессе сравнения: {str(e)}")
            raise

    def _create_no_expected_events_row(self, step_number, formatted_action):
        return [
            f"{WHITE}{str(step_number)}{RESET}",
            formatted_action,
            f"{RED}Ожидаемые ивенты не найдены в чеклисте, но они должны быть!{RESET}",
            f"{RED}Ожидаемые параметры не найдены в чеклисте, но они должны быть!{RESET}",
            f"{RED}✗{RESET}",
            f"{WHITE}Ошибка в чеклисте{RESET}"
        ]

    def _create_no_expected_events_html_row(self, step_number, action):
        return [
            str(step_number),
            action,
            "Ожидаемые ивенты не найдены в чеклисте, но они должны быть!",
            "Ожидаемые параметры не найдены в чеклисте, но они должны быть!",
            "✗",
            "Ошибка в чеклисте"
        ]

    def _create_no_events_expected_row(self, step_number, formatted_action):
        return [
            f"{WHITE}{str(step_number)}{RESET}",
            formatted_action,
            f"{GREEN}Нет ожидаемых ивентов{RESET}",
            f"{GREEN}Нет ожидаемых параметров{RESET}",
            f"{GREEN}✓{RESET}",
            f"{WHITE}{RESET}"
        ]

    def _create_no_events_expected_html_row(self, step_number, action):
        return [
            str(step_number),
            action,
            "Нет ожидаемых ивентов",
            "Нет ожидаемых параметров",
            "✓",
            ""
        ]

    @staticmethod
    def _add_event_to_table(step_number: int, formatted_action: str,
                            expected: ExpectedEvent, first_in_step: bool, found_match: bool,
                            best_match: Optional[ActualEvent], best_mismatches: List[str],
                            is_unexpected: bool = False, is_flexible: bool = False,
                            timestamp: Optional[datetime] = None) -> Tuple[List[str], List[str]]:
        """
        Формирует строки данных для консольной и HTML таблиц на основе сравнения событий.

        Метод определяет, как будет отображаться информация о каждом ожидаемом событии
        в таблицах отчетов. Учитывает, было ли найдено соответствие в логах,
        есть ли расхождения в параметрах, является ли событие опциональным или
        относится ли к группе с гибким порядком.

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
            timestamp (Optional[datetime]): Время события.

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
                f"{RED}Неожиданный ивент{RESET}",
                f"{WHITE}{timestamp}{RESET}" if timestamp else ""
            ]

            html_row = [
                str(step_number) if first_in_step else "",
                formatted_action.replace(WHITE, "").replace(RESET, "") if first_in_step else "",
                f'<span class="text-danger">{best_match.name}</span>',
                f'<span class="text-danger">{params_str}</span>',
                "✗",
                "Неожиданный ивент",
                str(timestamp) if timestamp else ""
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
        else:
            status_color = RED

        status_text = '✓' if (expected.is_optional and found_match and not best_mismatches) or (
                    not expected.is_optional and found_match and not best_mismatches) else \
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
            f"{RED}{chr(10).join(differences)[:COLUMN_WIDTHS['differences']]}{RESET}" if differences else "",
            f"{WHITE}{timestamp}{RESET}" if timestamp else ""
        ]

        # Создаем строки для HTML
        html_row = [
            str(step_number) if first_in_step else "",
            formatted_action.replace(WHITE, "").replace(RESET, "") if first_in_step else "",
            f"{event_name}{html_flexible_marker}{html_optional_marker}",
            "<br>".join(html_params),
            f'<span class="{"text-warning" if expected.is_optional and not found_match else "text-success" if status_color == GREEN else "text-danger"}">{status_text}</span>',
            "<br>".join([
                f'<span class="text-warning">{d.replace(YELLOW, "").replace(RED, "").replace(RESET, "")}</span>'
                if "Опциональный ивент не найден" in d
                else f'<span class="text-danger">{d.replace(RED, "").replace(RESET, "")}</span>'
                for d in differences
            ]) if differences else "",
            str(timestamp) if timestamp else ""
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
            if best_mismatches:
                 discrepancy_type = DiscrepancyType.PARAMETER_MISMATCH.value
                 details = best_mismatches
                 discrepancies.append({
                     'step': step_number,
                     'action': action,
                     'event': expected.name,
                     'type': discrepancy_type,
                     'details': details,
                     'timestamp': best_match.timestamp if best_match else None,
                 })

    def analyze(self) -> None:
        """Выполняет полный анализ логов и выводит результаты."""
        try:
            self.logger.info(f"{WHITE}Начинаем анализ...{RESET}")

            # Парсим чек-лист и логи
            self.parse_checklist()
            self.parse_logs()

            # Создаём словарь для хранения событий по шагам
            self.step_events = {}
            for event in self.expected_events:
                if event.step_number not in self.step_events:
                    self.step_events[event.step_number] = []
                self.step_events[event.step_number].append(event)

            # Первый проход - анализ временной последовательности
            self.timeline = self._analyze_event_timeline()

            # Проверяем временную последовательность на нарушения
            sequence_violations = self._check_timeline_sequence()

            if sequence_violations:
                self.logger.warning("\nОбнаружены нарушения временной последовательности:")
                for violation in sequence_violations:
                    if violation['type'] == 'wrong_order':
                        self.logger.warning(
                            f"  - Шаг {violation['earlier_step']} ({violation['earlier_event']}) "
                            f"должен быть раньше шага {violation['later_step']} ({violation['later_event']})\n"
                            f"    Время шага {violation['earlier_step']}: {violation['earlier_time']}\n"
                            f"    Время шага {violation['later_step']}: {violation['later_time']}"
                        )
                    elif violation['type'] == 'wrong_sequence':
                        self.logger.warning(
                            f"  - В шаге {violation['step']} события идут в неправильном порядке:\n"
                            f"    {violation['later_event']} (время: {violation['later_time']}) "
                            f"должно быть перед {violation['earlier_event']} (время: {violation['earlier_time']})"
                        )

                # Корректируем сопоставление событий
                self._correct_event_matching(sequence_violations)

                # Повторный анализ с учётом исправленной последовательности
                self.timeline = self._analyze_event_timeline()
            # Сравниваем события и получаем несоответствия и данные таблиц
            discrepancies, console_table_data, html_table_data = self._do_compare_events

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
                if missing_events:
                    self.logger.warning(f"  - Отсутствующих событий: {len(missing_events)}")
                if param_mismatches:
                    self.logger.warning(f"  - Количество шагов с несоответствием параметров: {len(param_mismatches)}")
                if unexpected_events:
                    self.logger.warning(f"  - Неожиданных ивентов: {len(unexpected_events)}")

                # Выводим информацию о каждом несоответствии параметров
                for disc in param_mismatches:
                    self.logger.warning(f"\nШаг {disc['step']} - {disc['event']}")
                    self.logger.warning(f"Действие: {disc['action']}")
                    self.logger.warning(f"Тип: parameter_mismatch")
                    if disc.get('timestamp'):
                        self.logger.warning(f"Временная метка: {disc['timestamp']}")
                    self.logger.warning("Детали:")
                    for detail in disc['details']:
                        self.logger.warning(f"  - {detail}")

                # Выводим информацию о неожиданных событиях
                if unexpected_events:
                    self.logger.warning("\nНеожиданные события:")
                    for disc in unexpected_events:
                        self.logger.warning(f"\nШаг {disc['step']} - {disc['event']}")
                        if disc.get('timestamp'):
                            self.logger.warning(f"Временная метка: {disc['timestamp']}")
                        for detail in disc['details']:
                            self.logger.warning(f"  {detail}")

                # В конце выводим список шагов с отсутствующими событиями
                if missing_events:
                    missing_steps = sorted(list(set(d['step'] for d in missing_events)))
                    self.logger.warning(f"\nНе найдены ивенты в шагах: {', '.join(map(str, missing_steps))}")

        except Exception as e:
            self.logger.error(f"Ошибка при анализе: {str(e)}")
            raise

    def _check_timeline_sequence(self) -> List[Dict[str, Any]]:
        """
        Проверяет корректность временной последовательности событий.
        """
        violations = []

        # Получаем все события с их временными метками и шагами
        event_timeline = []
        current_timestamp = None
        step_first_events = {}  # словарь для хранения первого события каждого шага

        # Сначала создаем временную линию всех событий
        for step, events in self.step_events.items():
            try:
                if isinstance(step, str) and 'ALT' in step:
                    continue
                step_num = int(str(step).split()[0]) if isinstance(step, str) else int(step)

                step_events = []
                for expected in events:
                    matches = [
                        actual for actual in self.actual_events
                        if (actual.name.lower() == expected.name.lower() or
                            actual.name.lower().replace('_', '') == expected.name.lower().replace('_', ''))
                    ]
                    for match in matches:
                        event_info = {
                            'step': step_num,
                            'event_name': expected.name,
                            'timestamp': match.timestamp,
                            'actual_event': match
                        }
                        step_events.append(event_info)
                        event_timeline.append(event_info)

                        # Сохраняем первое событие шага
                        if (step_num not in step_first_events
                            or match.timestamp < step_first_events[step_num]['timestamp']):
                            step_first_events[step_num] = event_info

            except (ValueError, TypeError):
                continue

        # Сортируем события по времени
        event_timeline.sort(key=lambda x: x['timestamp'])

        # Проверяем последовательность
        for i in range(len(event_timeline) - 1):
            current_event = event_timeline[i]
            next_event = event_timeline[i + 1]

            # Если следующее событие принадлежит более раннему шагу
            if next_event['step'] < current_event['step']:
                # Проверяем, является ли это первым событием своего шага
                if (step_first_events[next_event['step']]['timestamp'] == next_event['timestamp'] and
                        step_first_events[current_event['step']]['timestamp'] == current_event['timestamp']):
                    violations.append({
                        'type': 'wrong_order',
                        'earlier_step': next_event['step'],
                        'earlier_time': next_event['timestamp'],
                        'later_step': current_event['step'],
                        'later_time': current_event['timestamp'],
                        'earlier_event': next_event['event_name'],
                        'later_event': current_event['event_name']
                    })
            # Если событие того же шага идет в неправильном порядке
            elif next_event['step'] == current_event['step'] and next_event['timestamp'] < current_event['timestamp']:
                violations.append({
                    'type': 'wrong_sequence',
                    'step': current_event['step'],
                    'earlier_event': next_event['event_name'],
                    'earlier_time': next_event['timestamp'],
                    'later_event': current_event['event_name'],
                    'later_time': current_event['timestamp']
                })

        return violations

    def _correct_event_matching(self, violations: List[Dict[str, Any]]) -> None:
        """
        Корректирует сопоставление событий с учетом временных меток.
        """
        # Собираем все события со временем
        all_events = []
        for actual_event in self.actual_events:
            # Находим все возможные соответствия в ожидаемых событиях
            matched_expected = []
            for step, events in self.step_events.items():
                for expected in events:
                    if (actual_event.name.lower() == expected.name.lower() or
                            actual_event.name.lower().replace('_', '') == expected.name.lower().replace('_', '')):
                        matched_expected.append((step, expected))

            if matched_expected:
                all_events.append({
                    'actual': actual_event,
                    'matches': matched_expected,
                    'timestamp': actual_event.timestamp
                })

        # Сортируем события по времени
        all_events.sort(key=lambda x: x['timestamp'])

        # Создаем новую временную карту с учетом временной последовательности
        new_timeline = {}
        used_steps = set()

        for event in all_events:
            # Находим подходящий шаг для события
            best_step = None
            best_expected = None

            for step, expected in event['matches']:
                if isinstance(step, str) and 'ALT' in step:
                    continue
                try:
                    step_num = int(str(step).split()[0]) if isinstance(step, str) else int(step)
                    # Проверяем, что этот шаг идет после всех использованных
                    if not used_steps or step_num > max(used_steps):
                        best_step = step_num
                        best_expected = expected
                        break
                except (ValueError, TypeError):
                    continue

            if best_step is not None:
                if best_step not in new_timeline:
                    new_timeline[best_step] = StepTimeInfo(step_number=best_step)

                new_timeline[best_step].update_times(event['timestamp'])
                new_timeline[best_step].processed_events.append(event['actual'].name)
                used_steps.add(best_step)

        # Обновляем основную временную карту
        self.timeline = new_timeline


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
