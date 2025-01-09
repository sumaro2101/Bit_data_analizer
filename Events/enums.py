from enum import Enum


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
