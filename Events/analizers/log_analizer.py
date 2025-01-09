from collections.abc import Callable
import logging
import pathlib
import re
import pandas as pd
import numpy as np
from datetime import datetime

from .exceptions import FunctionNotProvideError
from dto import ActualEvent, ExpectedEvent, ResultBackend, Step
from enums import DiscrepancyType
from table_config import (
    GREEN, HEADERS,
    NAME_CHECK_LIST, RED, RESET, WHITE,
    IGNORED_EVENTS, HTML_TEMPLATE
    )
from backends import DefaultBackend


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
    IGNORED_EVENTS = IGNORED_EVENTS

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
        main_dir = pathlib.Path(__file__).resolve().parent.parent
        self.backend = DefaultBackend[Step]
        self.checklist_path = main_dir.joinpath('checklist', NAME_CHECK_LIST)
        self.log_path = pathlib.Path(log_path)
        self.expected_events: list[ExpectedEvent] = []
        self.actual_events: list[ActualEvent] = []
        self.user_id: str | None = None

        # Проверяем существование файла логов
        if not self.log_path.exists():
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

    def generate_html_report(self,
                             table_data: list[list[str]],
                             headers: list[str],
                             ) -> None:
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

            html_template = HTML_TEMPLATE

            table_html = df.to_html(
                classes='table table-striped table-bordered',
                escape=False,
                index=False
            )

            table_html = table_html.replace('<td>',
                                            '<td class="parameter-cell">',
                                            )
            main_dir = pathlib.Path(__file__).resolve().parent.parent
            report_path = main_dir.joinpath('HTMLreport.html')

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_template.format(
                    table_html=table_html,
                    user_id=self.user_id if self.user_id else "Не найден"
                ))

        except Exception as e:
            self.logger.error(f"Ошибка при создании HTML отчета: {str(e)}")
            raise

    @staticmethod
    def is_not_na(value) -> bool:
        """
        Безопасно преобразует различные типы в
        булево значение, указывающее на отсутствие NA (Not Available).

        Args:
            value: Входное значение для проверки

        Returns:
            bool: True, если значение не является NA, иначе False
        """
        if isinstance(value, (pd.Series, np.ndarray)):
            return bool(value.any()) if value is not None else False
        return bool(pd.notna(value))

    def parse_checklist(self) -> list[Step]:
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
                Внутренняя функция для безопасного
                преобразования номеров шагов.
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

            df_check_filtered['Номер шага'] = (df_check['Номер шага']
                                               .apply(parse_step_safely))

            # Сохраняем текущий номер шага для строк, где он пустой
            last_valid_step = None
            last_valid_action = None

            # Заполняем пропущенные номера шагов и действия
            # предыдущими значениями
            for idx, row in df_check_filtered.iterrows():
                # Безопасное получение значения шага
                step_value = row['Номер шага']

                # Проверка на NA с использованием улучшенного метода
                if self.is_not_na(step_value):
                    if isinstance(step_value, (pd.Series, np.ndarray)):
                        step_value = step_value.item()

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
                return (str(val).strip() not in ['ПРЕДУСЛОВИЯ'])

            df_check_filtered = df_check_filtered[
                (df_check_filtered['Проверяем наличие ивента(ов)']
                 .apply(is_valid_event))
            ]

            # Заменяем оригинальный DataFrame на отфильтрованный
            df_check = df_check_filtered

            current_step = 1
            current_action = None
            current_order_group = 0
            flexible_group_active = False  # Флаг, показывающий, активна ли текущая группа гибкого порядка
            list_of_expected_event_step = []
            list_out = []
            flexible_inside = False
            has_events_inside = True
            processed_empty_actions_step = False
            first_in_step = True

            # Проходим по всем строкам
            for index, row in df_check.iterrows():
                event_value = row['Проверяем наличие ивента(ов)']

                # Безопасное получение значений
                def safe_str_convert(val) -> str:
                    """
                    Безопасно преобразует значение в строку,
                    обрабатывая пустые значения.
                    """
                    return str(val).strip() if pd.notna(val) else ""

                step_value = row['Номер шага']
                try:
                    step_value = int(step_value)
                except ValueError:
                    pass

                event_value = safe_str_convert(row['Проверяем наличие ивента(ов)'])
                action_value = safe_str_convert(row['Действие'])
                params_value = safe_str_convert(row['Параметры через запятую'])

                if (step_value != current_step
                   and list_of_expected_event_step):
                    if processed_empty_actions_step:
                        has_events_inside = False
                    step = Step(step_number=current_step,
                                action=current_action,
                                events=list_of_expected_event_step,
                                has_events_inside=has_events_inside,
                                flexible_inside=flexible_inside,
                                )
                    list_out.append(step)
                    has_events_inside = True
                    flexible_inside = False
                    list_of_expected_event_step = []
                    current_step = step_value
                    first_in_step = True
                    processed_empty_actions_step = False

                # Обновляем текущий шаг и действие, если есть номер шага
                if self.is_not_na(step_value):
                    if isinstance(step_value, (pd.Series, np.ndarray)):
                        step_value = step_value.item()

                    if isinstance(step_value, str) and 'ALT' in step_value:
                        current_action = f"{action_value} (Альтернативный путь)"
                    else:
                        current_action = action_value
                if step_value == current_step and processed_empty_actions_step:
                    continue
                if event_value in ['nan', '-', ''] and first_in_step:
                    processed_empty_actions_step = True

                # Определяем тип гибкого порядка
                is_optional = False
                if '[?]' in event_value:
                    is_optional = True
                    event_value = event_value.replace('[?]', '').strip()

                is_flexible = False
                current_event_str_raw_lines = event_value.split('\n')
                is_has_events = True
                if (current_event_str_raw_lines
                   and '-' == current_event_str_raw_lines[0]):
                    is_has_events = False
                    has_events_inside = False
                for current_event_str_raw in current_event_str_raw_lines:
                    current_event_str_trimmed = current_event_str_raw.strip()
                    if (not current_event_str_trimmed
                       and not processed_empty_actions_step):
                        continue

                    current_event_str = current_event_str_trimmed

                    if '[↓]' in current_event_str:
                        is_flexible = True
                        flexible_inside = True
                        flexible_group_active = True  # Активируем группу гибкого порядка
                        current_event_str = current_event_str.replace('[↓]', '').strip()
                        current_order_group += 1
                    elif '[↑]' in current_event_str:
                        is_flexible = True
                        flexible_inside = True
                        current_event_str = (current_event_str
                                             .replace('[↑]', '').strip())
                        if '[↓]' not in event_value:
                            if not flexible_group_active:
                                current_order_group += 1
                            flexible_group_active = False

                    # Создаем событие
                    event = ExpectedEvent(
                        step_number=current_step,
                        name=current_event_str,
                        parameters=self._extract_parameters(params_value),
                        action=current_action or "",
                        flexible_order=is_flexible,
                        order_group=current_order_group if is_flexible else 0,
                        is_optional=is_optional,
                        has_events=is_has_events,
                    )

                    # Добавляем событие в список
                    list_of_expected_event_step.append(event)
                    first_in_step = False
            if list_of_expected_event_step:
                step = Step(step_number=current_step,
                            action=current_action,
                            events=list_of_expected_event_step,
                            has_events_inside=has_events_inside,
                            flexible_inside=flexible_inside,
                            )
                list_out.append(step)
            self.logger.info(f'{WHITE}Успешно обработано '
                             f'{len(self.expected_events)} '
                             f'ожидаемых событий{RESET}')
            return list_out

        except Exception as e:
            self.logger.error(f"Ошибка при чтении чек-листа: {str(e)}")
            raise e

    def parse_logs(self,
                   expected_steps: list[Step],
                   ) -> list[Step]:
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
                actual_events = []

                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # Если строка начинается с даты, это новое событие
                    if re.match(r'\d{2}\.\d{2}\.\d{4}', line) or re.match(r'\d{1}/\d{1}/\d{4}', line):
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
                                actual_events.append(event)
                                self.logger.debug(f"Добавлено событие: {current_event}")

                        # Парсим новое событие
                        parts = line.split('|')
                        time = parts[0].strip()
                        if len(parts) >= 2:
                            if time.split(' ')[-1] in ['AM', 'PM']:
                                current_time = datetime.strptime(parts[0].strip(),
                                                                 '%d/%m/%Y %I:%M:%S %p')
                            else:
                                current_time = datetime.strptime(parts[0].strip(),
                                                                '%d.%m.%Y %H:%M:%S')
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
                    actual_events.append(event)

            self.logger.info(f"{WHITE}Успешно прочитано {len(self.actual_events)} фактических событий{RESET}")
            actual_steps = self._parse_steps(
                actual_events=actual_events,
                expected_steps=expected_steps,
            )
            return actual_steps

        except Exception as e:
            self.logger.error(f"Ошибка при чтении лог-файла: {str(e)}")
            raise e

    def _parse_steps(self,
                     actual_events: list[ActualEvent],
                     expected_steps: list[Step],
                     ) -> list[Step]:
        """
        Алгоритм который исходя из ожидаемых и актуальных событий
        генерирует актуальные шаги для дальнейшей обработки.

        Args:
            actual_events (list[ActualEvent]): Актуальные события.
            expected_steps (list[Step]): Ожидаемые шаги.

        Returns:
            list[Step]: Сгенерированные актуальные шаги.
        """

        list_of_steps = []
        start = 0
        leng_actual = len(actual_events)
        for step_index, expected_step in enumerate(expected_steps):
            if start >= leng_actual:
                return list_of_steps
            leng_expected = len(expected_step)
            if leng_expected == 1 and expected_step[0].name in ['-',
                                                                'nan',
                                                                '']:
                continue
            if expected_step.step_number == 13:
                pass
            curr_expected_names = [event.name for event in expected_step]
            next_expected_names = self._get_next_steps_names(
                index=step_index,
                expected_steps=expected_steps,
            )
            actual_next_names = [event.name
                                 for event
                                 in actual_events[start +
                                                  leng_expected: start +
                                                  leng_expected +
                                                  len(next_expected_names) + 1]]
            closest_index = self._get_index(
                names=next_expected_names,
                actual_next=actual_next_names,
            )
            if closest_index is None:
                actual_step = actual_events[start:start + leng_expected]
            else:
                actual_step = actual_events[start:start + leng_expected + closest_index]
            actual_names = [event.name for event in actual_step]
            last_actual_index = self._get_index(
                names=curr_expected_names,
                actual_next=actual_names,
                func=max,
            )
            if last_actual_index is None:
                if closest_index is None:
                    new_index = self._get_index(
                        names=next_expected_names,
                        actual_next=actual_names,
                    )
                    if new_index is None:
                        new_interval_step = []
                    else:
                        new_interval_step = actual_events[start: start + new_index]
                else:
                    actual_next_names = [event.name
                                         for event
                                         in actual_events[start: start +
                                                          leng_expected +
                                                          len(next_expected_names) + 1]]
                    closest_index = self._get_index(
                        names=next_expected_names,
                        actual_next=actual_next_names,
                    )
                    if closest_index is not None:
                        new_interval_step = actual_events[start: start + closest_index]
                    else:
                        new_interval_step = actual_events[start: start + min(closest_index, leng_expected)]
            else:
                if last_actual_index == 0:
                    new_interval_step = actual_step
                else:
                    new_interval_step = actual_events[start: start + last_actual_index + 1]
                last_expected_event = expected_step[-1]
                last_index = self._get_last_index(
                    name=last_expected_event.name,
                    curr_interval=actual_names,
                )
                if last_index != 0 and last_index != len(new_interval_step):
                    first_next_event = next_expected_names[0]
                    last_actual_event = curr_expected_names[-1]
                    curr_interval_names = [event.name for event in new_interval_step]
                    next_entrypoints = actual_next_names.count(first_next_event)
                    curr_entrypoints = curr_interval_names.count(first_next_event)
                    full_interval = actual_events[start + len(new_interval_step):]
                    full_interval_names = [event.name for event in full_interval]
                    first_next_index = self._get_index(
                        names=[first_next_event],
                        actual_next=full_interval_names,
                    )
                    if first_next_index is None:
                        actual_step = actual_events[start:start + leng_expected]
                    elif first_next_event == last_actual_event and next_entrypoints > curr_entrypoints:
                        new_interval_step = actual_events[start: start + len(new_interval_step) + 1]
                    else:
                        new_interval_step = actual_events[start: start + len(new_interval_step) + first_next_index]
            try:
                last_pass = self._check_parameters(
                    actual_event=new_interval_step[-1],
                    expected_curr=expected_step[-1],
                )
                next_event_step = actual_events[start + len(new_interval_step)]
            except IndexError:
                next_event_step = None
            if (next_event_step and
                not last_pass
               and next_expected_names[0] == expected_step[-1].name):
                equal_next = self._check_parameters(
                    actual_event=next_event_step,
                    expected_curr=expected_step[-1],
                )
                if equal_next and last_pass:
                    pass
                elif not equal_next and last_pass:
                    pass
                elif equal_next and not last_pass:
                    new_interval_step = actual_events[start: start + len(new_interval_step) + 1]

            actual_step = Step(
                step_number=expected_step.step_number,
                action=expected_step.action,
                events=new_interval_step,
            )
            list_of_steps.append(actual_step)
            start += len(new_interval_step)
        return list_of_steps

    def _get_next_steps_names(self,
                              index: int,
                              expected_steps: list[Step],
                              ) -> list[str]:
        """
        Получение группы имен из следующего шага.
        Учитывается то что следующий шаг может быть пустым,
        в этом случае алгоритм углубляется дальше.

        Args:
            index (int): Актуальный индекс.
            expected_steps (list[Step]): Ожидаемые шаги.

        Returns:
            list[str]: Список имен из найденного следующего шага.
        """

        index += 1
        try:
            next_expected_step = expected_steps[index]
        except IndexError:
            index -= 1
            next_expected_step = expected_steps[index]
            next_expected_names = [event.name for event in next_expected_step]
            return next_expected_names
        while next_expected_step[0].name in ['',
                                             '-',
                                             'nan',
                                             ]:
            index += 1
            try:
                next_expected_step = expected_steps[index]
            except IndexError:
                break
        next_expected_names = [event.name for event in next_expected_step]
        return next_expected_names

    def _get_index(self,
                   names: list[str],
                   actual_next: list[str],
                   func: Callable | None = None,
                   ) -> int:
        """
        Получение индекса близжайшего или самого удаленного
        имени из предложенного массива имен.

        Если функция не предоставляется - то возвращает первый
        найденный идекс.

        Args:
            names (list[str]): Массив имен.
            actual_next (list[str]): Имена следующего предположительного
            шага.
            func (Callable | None, optional): Функция для фильтрации
            близжайшего или дальнего индекса.

        Returns:
            int: Индекс.
        """

        if func:
            func_names = ['max', 'min']
            func_name = func.__name__
            if func_name not in func_names:
                raise FunctionNotProvideError(f'{func_name} не поддерживается для '
                                              'поиска индекса, возможно только '
                                              'использовать функции `min`, `max`')
        list_indexes = []
        actual_next_copy = actual_next.copy()
        for name in names:
            try:
                index = actual_next_copy.index(name)
                actual_next_copy[index] = 'finded'
                if not func:
                    return index
                list_indexes.append(index)
            except ValueError:
                continue
        if list_indexes:
            result = func(list_indexes)
            return result

    def _get_last_index(self,
                        name: str,
                        curr_interval: list[str],
                        ) -> int | None:
        """
        Получение самого последнего индекса
        по имени из массива имен.

        Args:
            name (str): Имя для поиска
            curr_interval (list[str]): Текущий интервал именн
            актуального предположительного шага.

        Returns:
            int | None: Индекс, елси не найден - None.
        """

        index = 0
        try:
            finded = curr_interval[::-1].index(name)
            index = len(curr_interval) - finded
        except ValueError:
            return None
        return index

    def _check_parameters(self,
                          actual_event: ActualEvent,
                          expected_curr: ExpectedEvent,
                          ) -> bool:
        """
        Проверяет сходство параметров двух событий.

        Args:
            actual_event (ActualEvent): Актуальное событие.
            expected_curr (ExpectedEvent): Ожидаемое событие

        Returns:
            bool: Результат проверки равенства.
        """

        actual_param = {key: value
                        for key, value
                        in actual_event.parameters.items()
                        if key != 'PreciseTime'}
        expected_curr = {key: value
                         for key, value
                         in expected_curr.parameters.items()
                         if key != 'PreciseTime'}
        return actual_param == expected_curr

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

    def _extract_parameters(self, params_str: str) -> dict[str, str | None]:
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

    def _analize(self) -> ResultBackend:
        """
        Главный entrypoint для парсинга и анализа данных.

        Returns:
            ResultBackend: Сущность результата после анализа.
        """

        expected_steps = self.parse_checklist()
        actual_steps = self.parse_logs(expected_steps)
        result = self.backend(
            actial_events=actual_steps,
            expected_events=expected_steps,
        )
        return result.backend_data

    def analyze(self) -> None:
        """Выполняет полный анализ логов и выводит результаты."""
        try:
            self.logger.info(f"{WHITE}Начинаем анализ...{RESET}")
            result = self._analize()
            discrepancies = result.discrepancies
            table_data = result.html_data
            try:
                self.generate_html_report(table_data, HEADERS)
                self.logger.info(
                    f"\n{GREEN}HTML отчет сохранен в файл: {pathlib.Path(__file__).resolve().parent.parent.joinpath('HTMLreport.html')}{RESET}")
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
                    self.logger.warning("Тип: parameter_mismatch")
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
                    missing_steps = list(set(d['step'] for d in missing_events))
                    self.logger.warning(f"\nНе найдены ивенты в шагах: {', '.join(map(str, missing_steps))}")

        except Exception as e:
            self.logger.error(f"Ошибка при анализе: {str(e)}")
            raise Exception
