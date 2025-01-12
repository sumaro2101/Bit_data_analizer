import pandas as pd
from pandas import DataFrame
import numpy as np
from typing import Any, List, Optional, cast

from table_config import GREEN, RESET


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
        current_xlsx_columns = set(df.columns)
        missing_columns = set(self.REQUIRED_COLUMNS.keys()) - current_xlsx_columns
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

    def _validate_column(self, df: pd.DataFrame, column: str, rules: dict) -> None:
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

    def _validate_step_numbers(self, df: pd.DataFrame) -> None:
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
                        return int(val_str.split(' ')[0].strip())
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
                sorted_steps = sorted(duplicate_steps)
                self.errors.append(
                    f"Обнаружены повторяющиеся номера шагов (не ALT): {sorted_steps}"
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
