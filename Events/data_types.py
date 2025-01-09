from typing import TypeVar

from dto import (ActualEvent,
                 ExpectedEvent,
                 Step,
                #  FlexibleGroup,
                 EventResult,
                 )


HT = TypeVar('HT', bound=list[list[str, str, str, str, str, str]])
AE = TypeVar('AE', bound=ActualEvent)
EX = TypeVar('EX', bound=ExpectedEvent)
DE = TypeVar('SE', bound=dict[str, ExpectedEvent])
SI = TypeVar('SI', bound=dict[str, str])
# FX = TypeVar('FX', bound=FlexibleGroup)
ER = TypeVar('ER', bound=EventResult)
S = TypeVar('S', bound=Step)
