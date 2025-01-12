from datetime import datetime
from typing import TypeVar, TypeAlias

from dto import (ActualEvent,
                 ExpectedEvent,
                 Step,
                 EventResult,
                 )


HT = TypeVar('HT', bound=list[list[str, str, str, str, str, str]])
AE = TypeVar('AE', bound=ActualEvent)
EX = TypeVar('EX', bound=ExpectedEvent)
DE = TypeVar('SE', bound=dict[str, ExpectedEvent])
SI = TypeVar('SI', bound=dict[str, str])
ER = TypeVar('ER', bound=EventResult)
S = TypeVar('S', bound=Step)
Discrepancies: TypeAlias = list[dict[str, str | int | datetime | None]]
