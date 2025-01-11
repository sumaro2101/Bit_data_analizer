from dataclasses import dataclass, field
from datetime import datetime


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
    step_number: int | str
    name: str
    parameters: dict[str, str | None]  # None означает, что параметр должен существовать без конкретного значения
    action: str
    flexible_order: bool = False  # Флаг для событий с произвольным порядком
    order_group: int = 0  # Группа событий с произвольным порядком
    is_optional: bool = False  # Флаг, является ли событие опциональным
    has_events: bool = True

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
    parameters: dict[str, str]
    timestamp: datetime
    pass_count: int = 0


@dataclass
class ResultBackend:
    table_data: list[list[str, str, str, str, str, str]]
    html_data: list[list[str, str, str, str, str, str]]
    discrepancies: list[list[str, str, str, str, str, str]]
    unexpected_events_count: int


@dataclass
class EventResult:
    event: ActualEvent | None = None
    mismatch: list[str] = field(default_factory=list)
    best_match: bool = False
    checked: bool = False
    expected: ExpectedEvent | None = None
    flexible: bool = False
    found_match: bool = False
    is_unexpected: bool = True
    optional: bool = False
    timestamp: datetime | None = None
    wrong_time: bool = False


@dataclass
class Step:
    step_number: int | str
    action: str
    events: list[ExpectedEvent | ActualEvent]
    has_events_inside: bool = True
    flexible_inside: bool = False

    def __getitem__(self, index):
        return self.events[index]

    def __iter__(self):
        return iter(self.events)

    def __len__(self):
        return len(self.events)

    def __eq__(self, value):
        if not isinstance(value, type(self)):
            return
        if len(self) != len(value):
            return
        self_events = [event.name for event in self.events]
        other_events = [event.name for event in value.events]
        return self_events == other_events
