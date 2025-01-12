from datetime import datetime
import re
from typing import Generic, ClassVar
from functools import partial

from enums import DiscrepancyType
from data_types import ER, HT, S, AE, EX
from collections import defaultdict
from dto import ResultBackend, EventResult
from utils import do_formatted_action
from table_config import (BLUE, COLUMN_WIDTHS,
                          RED, WHITE, RESET, GREEN,
                          YELLOW,
                          )


class DefaultBackend(Generic[S]):
    """
    Дэфолтный бэкенд для анализа шагов.
    """

    unexpected_events_count: ClassVar[int] = 0

    def __init__(self,
                 actial_events: list[S],
                 expected_events: list[S],
                 ) -> None:
        self.actial_events = list(actial_events)
        self.expected_events = list(expected_events)
        self._table_data: HT = []
        self._html_data: HT = []
        self._discrepancies = []

    @property
    def table_data(self):
        return self._table_data

    @property
    def html_data(self):
        return self._html_data

    @property
    def discrepancies(self):
        return self._discrepancies

    @property
    def backend_data(self):
        self._reset_backend_data()
        self._change_relative_events(
            self.actial_events,
            self.expected_events,
        )
        return ResultBackend(
            table_data=self._table_data,
            html_data=self._html_data,
            discrepancies=self._discrepancies,
            unexpected_events_count=self.unexpected_events_count,
        )

    def _reset_backend_data(self):
        """
        Обновление всех данных.
        """
        self._table_data = []
        self._html_data = []
        self._discrepancies = []
        self.unexpected_events_count = 0

    def _change_relative_events(
        self,
        actual_events: list[S],
        expected_events: list[S],
    ) -> None:
        """
        Основной движок для сравнения шагов.
        """
        actual_event_index = 0
        for expected_step in expected_events:
            if expected_step.step_number == 67:
                pass
            if not expected_step.has_events_inside:
                self._register_no_expect_action(
                    expected_step=expected_step,
                )
                continue
            try:
                curr_actual_step = actual_events[actual_event_index]
            except IndexError:
                self._register_undefind_step(
                    expected_step=expected_step,
                )
                actual_event_index += 1
                continue
            if curr_actual_step != expected_step:
                self._register_not_equal_steps_new(
                    expected_step=expected_step,
                    actual_step=curr_actual_step,
                )
                actual_event_index += 1
            else:
                self._register_equal_steps(
                    expected_step=expected_step,
                    actual_step=curr_actual_step,
                )
                actual_event_index += 1

    def _register_undefind_step(self,
                                expected_step: S,
                                ):
        """
        Регистрация не найденного события события.

        Args:
            expected_step (S): Ожидаемый шаг.
        """
        formatted_action = do_formatted_action(
            action=expected_step.action,
            )
        first = True
        for expected in expected_step:
            event_to_table = self._patrial_add_event(
                expected_event=expected,
                first=first,
                formatted_action=formatted_action,
            )
            discrepancies = self._patrial_add_discrepancies(
                expected_event=expected,
            )
            event_to_table(
                best_mismatches=[],
                found_match=False,
                is_unexpected=True,
            )
            discrepancies(
                found_match=False,
                best_match=None,
                best_mismatches=[],
            )
            first = False

    def _register_equal_steps(self,
                              expected_step: S,
                              actual_step: S,
                              ):
        """
        Регистрация равных между собой шагов.

        Args:
            expected_step (S): Ожидаемый шаг.
            actual_step (S): Актуальный шаг.
        """
        formatted_action = do_formatted_action(
            action=expected_step.action,
            )
        first = True
        for expected, actual in zip(expected_step, actual_step):
            event_to_table = self._patrial_add_event(
                expected_event=expected,
                first=first,
                formatted_action=formatted_action,
                flexible=expected.flexible_order,
            )
            discrepancies = self._patrial_add_discrepancies(
                expected_event=expected,
            )
            is_match, matches = self._check_event_match(
                expected=expected,
                actual=actual,
            )
            is_unexpected = all((not is_match,
                                 not expected.is_optional))
            event_to_table(
                best_mismatches=matches,
                found_match=is_match,
                best_match=(actual
                            if not is_unexpected
                            else is_unexpected),
                is_unexpected=is_unexpected,
                timestamp=actual.timestamp,
            )
            discrepancies(
                found_match=is_match,
                best_match=(actual
                            if not is_unexpected
                            else is_unexpected),
                best_mismatches=matches,
            )
            first = False

    def _register_not_equal_steps_new(self,
                                      expected_step: S,
                                      actual_step: S,
                                      ) -> None:
        """
        Регистрация не равных шагов между собой.

        Args:
            expected_step (S): Ожидаемый шаг.
            actual_step (S): Актуальный шаг.
        """

        step = self._initialization_steps(
            actual_step=actual_step,
        )
        list_usality = []
        unexpected_events = defaultdict(list)
        if not step:
            for expected_event in expected_step:
                event = EventResult(
                    mismatch=[],
                    found_match=False,
                    is_unexpected=True,
                    best_match=None,
                    expected=expected_event,
                    optional=expected_event.is_optional,
                    flexible=expected_event.flexible_order,
                    pass_expected=expected_step.pass_expected,
                )
                step.append(event)
                self.unexpected_events_count += 1
            self._register_events(
                step=step,
                expected_event=expected_step[0],
            )
            return

        for expected_event in expected_step:
            for index, event in enumerate(step):
                if (not event.event.name == expected_event.name or
                   event.checked):
                    continue
                is_match, mismatch = self._check_event_match(
                    expected=expected_event,
                    actual=event.event,
                    )
                if mismatch:
                    queue_event = EventResult(
                        mismatch=mismatch,
                        event=event,
                        best_match=event.event,
                        checked=True,
                        expected=expected_event,
                        flexible=event.flexible,
                        found_match=not mismatch,
                        is_unexpected=bool(mismatch),
                        optional=event.optional,
                        timestamp=event.event.timestamp,
                    )
                    unexpected_events[index].append(queue_event)
                else:
                    self._add_correct_event(
                        event=event,
                        expected_event=expected_event,
                        list_usality=list_usality,
                    )
                    unexpected_events = self._remove_from_queue(
                        index=index,
                        unexpected_events=unexpected_events,
                        expected_event=expected_event
                    )
                    break

        if expected_step.step_number == 9:
            pass

        matching_step = self._find_best_match_from_queues(
            unexpected_events=unexpected_events,
            list_usality=list_usality,
            step=step,
        )

        step_with_unexpected = self._add_unexpected_events(
            step=matching_step,
            expected_event=expected_step[0],
        )

        step_with_undefind = self._add_undefind_events(
            expected_step=expected_step,
            step=step_with_unexpected,
            list_usality=list_usality,
        )

        out_list_step = self._add_timeline_mark(
            correct_step=step_with_undefind,
            fill_step=matching_step,
        )
        if expected_event.step_number == 19:
                pass
        self._register_events(
            step=out_list_step,
            expected_event=expected_step[0],
        )

    def _initialization_steps(self,
                              actual_step: S,
                              ) -> list[ER]:
        """
        Инициализация шага для обработки.

        Args:
            actual_step (S): Актуальный шаг.

        Returns:
            list[ER]: Новый шаг для обработки.
        """
        step = [EventResult(event=event,
                            timestamp=event.timestamp)
                for event
                in actual_step]
        return step

    def _add_timeline_mark(self,
                           correct_step: list[ER],
                           fill_step: list[ER],
                           ) -> list[ER]:
        """
        Добавление метки корректности к каждому временному
        интервалу в событии.

        Args:
            correct_step (list[ER]): Сгенерированный шаг с полными данными.

            fill_step (list[ER]): Сгенерированный шаг с заполненной временной
            последовательностью.

        Returns:
            list[ER]: Сгенерированный шаг.
        """

        out_list_step = []
        rigth_time_event = []
        wrong_time_event = []
        for wr_event in correct_step:
            if wr_event not in fill_step:
                out_list_step.append(wr_event)
            if not wr_event.found_match:
                continue
            if wr_event.expected in wrong_time_event:
                continue
            for st_event in fill_step:
                if not st_event.found_match:
                    if st_event not in out_list_step:
                        out_list_step.append(st_event)
                    continue
                if st_event.wrong_time or st_event.expected in rigth_time_event:
                    continue
                if not st_event.expected == wr_event.expected and not st_event.expected.flexible_order:
                    st_event.wrong_time = True
                    out_list_step.append(st_event)
                    wrong_time_event.append(st_event.expected)
                else:
                    rigth_time_event.append(st_event.expected)
                    out_list_step.append(st_event)
                    break

        del rigth_time_event
        del wrong_time_event
        return out_list_step

    def _add_undefind_events(self,
                             expected_step: S,
                             step: list[ER],
                             list_usality: list[ER],
                             ) -> list[ER]:
        """
        Добавление не найденных событий которые ожидались.

        Args:
            expected_step (S): Ожидаемый шаг.
            step (list[ER]): Сгенерированный шаг.
            list_usality (list[ER]): Список найденных событий.

        Returns:
            list[ER]: Сгенерированный шаг.
        """

        to_write_events = []
        pass_events = []
        for ex_event in expected_step:
            if ex_event not in list_usality:
                undefind_event = EventResult(
                    mismatch=[],
                    found_match=False,
                    is_unexpected=True,
                    best_match=None,
                    expected=ex_event,
                    optional=ex_event.is_optional,
                    flexible=ex_event.flexible_order,
                )
                to_write_events.append(undefind_event)
                continue
            for ac_event in step:
                if not ac_event.found_match:
                    if ac_event not in pass_events:
                        pass_events.append(ac_event)
                        to_write_events.append(ac_event)
                    continue
                if ac_event.expected == ex_event:
                    to_write_events.append(ac_event)
        return to_write_events

    def _add_unexpected_events(self,
                               step: list[ER],
                               expected_event: EX,
                               ) -> list[ER]:
        """
        Добавление неожидаемых событий.

        Args:
            step (list[ER]): Сгенерированный шаг.
            expected_event (EX): Ожидаемое событие для инициализации.

        Returns:
            list[ER]: Сгенерированный шаг.
        """

        for event in step:
            if not event.checked:
                event.best_match = event.event
                event.checked = True
                event.expected = expected_event
                event.found_match = False
                event.is_unexpected = True
                event.optional = event.optional
                event.flexible = event.flexible
        return step

    def _find_best_match_from_queues(self,
                                     unexpected_events: defaultdict[int,
                                                                    list[ER]],
                                     list_usality: list[ER],
                                     step: list[ER],
                                     ) -> list[ER]:
        """
        Находит самое подходящие события из каждой очереди
        претендентов на не найденное ожидаемое событие.

        Args:
            unexpected_events (defaultdict[int, list[ER]]): Словарь
            с очередями претендентов.
            list_usality (list[ER]): Список найденных событий.
            step (list[ER]): Список с основными событиями.

        Returns:
            list[ER]: Список с основными событиями.
        """

        if unexpected_events:
            unexpected_events = {key: value
                                 for key, value
                                 in unexpected_events.items() if value}
            unex_priority = dict()
            for unex_key, unex_event in unexpected_events.items():
                if not unex_event:
                    pass
                priority = min(unex_event, key=lambda a: len(a.mismatch))
                unex_priority[unex_key] = priority
            for unex_event in unex_priority.values():
                best_match_expected = {unex_key: unex_value
                                       for unex_key, unex_value
                                       in unex_priority.items()
                                       if (unex_value.expected.parameters ==
                                           unex_event.expected.parameters)
                                       and unex_value.expected not
                                       in list_usality}
                if best_match_expected:
                    best_match = min(best_match_expected.items(),
                                     key=lambda a: len(a[-1].mismatch))
                    curr_match: EventResult = best_match[-1]
                    curr_match.is_unexpected = False
                    curr_match.found_match = True
                    step[best_match[0]] = curr_match
                    list_usality.append(curr_match.expected)
        return step

    def _remove_from_queue(self,
                           index: int,
                           unexpected_events: defaultdict[int, list[ER]],
                           expected_event: EX,
                           ) -> defaultdict[int, list[ER]]:
        """
        Удаление действия из очереди конкуриющих событий.

        Args:
            index (int): Текущий индекс действия.

            unexpected_event (defaultdict[int, list[ER]]): Словарь с
            очередями конкурирующих действий.

            expected_event (EX): Ожидаемое событие.
        """

        for unex_event_del in unexpected_events.values():
            find_event = [event
                          for event
                          in unex_event_del
                          if (event.expected.parameters ==
                              expected_event.parameters)]
            if find_event:
                index_event = unex_event_del.index(find_event[0])
                unex_event_del.pop(index_event)
        if unexpected_events.get(index, None):
            unexpected_events.pop(index)
        return unexpected_events

    def _add_correct_event(self,
                           event: ER,
                           expected_event: EX,
                           list_usality: list[ER],
                           ) -> None:
        """
        Добавление корректного действия в список найденых.

        Args:
            event (ER): Корректное действие.
            expected_event (EX): Ожидаемое действие.
            list_usality (list[ER]): Список найденых.
        """

        event.best_match = event.event
        event.checked = True
        event.expected = expected_event
        event.flexible = event.flexible
        event.found_match = True
        event.is_unexpected = False
        event.optional = event.optional
        event.timestamp = event.event.timestamp
        list_usality.append(expected_event)

    def _register_events(self,
                         step: list[ER],
                         expected_event: EX,
                         ) -> None:
        """
        Регистрация сгенерированного шага.

        Args:
            step (list[ER]): Сгенерированный шаг.
            expected_event (EX): Ожидаемое действие для иницализации.
        """

        formatted_action = do_formatted_action(
            action=expected_event.action,
            )
        event_to_table = self._patrial_add_event(
            formatted_action=formatted_action,
            first=True,
            expected_event=expected_event,
        )
        discrepancies = self._patrial_add_discrepancies(
            expected_event=expected_event,
            )
        first = True
        for event in step:
            event_to_table(
                step_number=expected_event.step_number,
                best_mismatches=event.mismatch,
                found_match=event.found_match,
                is_unexpected=event.is_unexpected,
                first_in_step=first,
                best_match=event.best_match,
                expected=event.expected,
                is_flexible=event.expected.flexible_order,
                timestamp=event.timestamp,
                wrong_time=event.wrong_time,
                pass_expected=event.pass_expected,
            )
            discrepancies(
                step_number=expected_event.step_number,
                action=expected_event.action,
                expected=event.expected,
                found_match=event.found_match,
                best_match=event.best_match,
                best_mismatches=event.mismatch,
                pass_expected=event.pass_expected,
            )
            first = False

    def _patrial_add_event(self,
                           expected_event: EX,
                           first: bool,
                           formatted_action: str,
                           flexible: bool = False,
                           ) -> partial:
        event_to_table = partial(
            self._add_event_to_table,
            step_number=expected_event.step_number,
            formatted_action=(formatted_action
                              if first
                              else ""),
            first_in_step=first,
            expected=expected_event,
            is_flexible=flexible,
        )
        return event_to_table

    def _patrial_add_discrepancies(self,
                                   expected_event: EX,
                                   ) -> partial:
        discrepancies = partial(
            self._add_discrepancy,
            discrepancies=self.discrepancies,
            step_number=expected_event.step_number,
            action=expected_event.action,
            expected=expected_event,
        )
        return discrepancies

    def _register_no_expect_action(
        self,
        expected_step: S,
    ) -> None:
        """
        Регистрация шага который не ожидает действий.

        Args:
            expected_step (S): Ожидаемый шаг.
        """

        curr_step = expected_step.step_number
        try:
            curr_expect_step = expected_step[0]
        except IndexError:
            curr_action = 'undefind_action'
        curr_action = curr_expect_step.action
        formatted_action = do_formatted_action(
            action=curr_action,
        )
        self._add_no_expect_events(
                index=curr_step,
                table_data=self.table_data,
                html_data=self.html_data,
                action_detail=curr_action,
                formatted_action=formatted_action,
            )

    def _add_event_to_table(
        self,
        step_number: int,
        formatted_action: str,
        first_in_step: bool,
        found_match: bool,
        best_mismatches: list[str],
        expected: EX | None,
        best_match: AE | None = None,
        is_unexpected: bool = False,
        is_flexible: bool = False,
        timestamp: datetime | None = None,
        wrong_time: bool = False,
        pass_expected: bool = False,
    ) -> tuple[list[str], list[str]]:
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
            self.html_data.append(html_row)
            self.table_data.append(console_row)
            return

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
                if pass_expected:
                    html_params.append(f"{param_name}={value_str}")
                else:
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
        elif found_match and not wrong_time:
            status_color = GREEN if not best_mismatches else RED
        elif pass_expected:
            status_color = YELLOW
        else:
            status_color = RED

        status_text = '✓' if (expected.is_optional and found_match and not best_mismatches) or (
                    not expected.is_optional and found_match and not best_mismatches) else \
            '✓' if (expected.is_optional and not found_match) else '✗' if not found_match else '✗'
        if wrong_time:
            status_text = '✗'
        if pass_expected:
            status_text = '-'

        differences = list(best_mismatches)

        # Обработка опциональных событий
        if expected.is_optional and not found_match:
            differences = [f"{YELLOW}Опциональный ивент не найден, но это нормально{RESET}"]
        elif not found_match and not pass_expected:
            status_text = 'Ивент не найден'
        elif pass_expected:
            differences = [f'{YELLOW} Шаг намерено пропущен пользователем{RESET}']
        if wrong_time:
            wrong_text = f'{RED}Нарушена последовательность шагов!{RESET}'

        timestamp_text = f"{WHITE}{timestamp}{RESET}" if timestamp else ""
        # Создаем строки для консоли
        console_row = [
            f"{WHITE}{str(step_number)}{RESET}" if first_in_step else "",
            formatted_action if first_in_step else "",
            f"{WHITE}{event_name}{flexible_marker}{optional_marker}{RESET}",
            "✗",
            f"{WHITE}\n{RESET}".join(console_params),
            f"{status_color}{status_text}{RESET}",
            f"{RED}{chr(10).join(differences)[:COLUMN_WIDTHS['differences']]}{RESET}" if differences else "",
            timestamp_text
        ]

        # Создаем строки для HTML
        html_row = [
            str(step_number) if first_in_step else "",
            formatted_action.replace(WHITE, "").replace(RESET, "") if first_in_step else "",
            f"{event_name}{html_flexible_marker}{html_optional_marker}",
            "<br>".join(html_params),
            f'<span class="{"text-warning" if (expected.is_optional or pass_expected) and not found_match else "text-success" if status_color == GREEN else "text-danger"}">{status_text}</span>',
            "<br>".join([
                f'<span class="text-warning">{d.replace(YELLOW, "").replace(RED, "").replace(RESET, "")}</span>'
                if "Опциональный ивент не найден" in d or 'Шаг намерено пропущен пользователем' in d
                else f'<span class="text-danger">{d.replace(RED, "").replace(RESET, "")}</span>'
                for d in differences
            ]) if differences else "" if not wrong_time else f'<span class="text-danger">{wrong_text}</span>',
            str(timestamp) if timestamp else ""
        ]

        self.html_data.append(html_row)
        self.table_data.append(console_row)
        return

    def _add_discrepancy(self,
                         discrepancies: list[dict],
                         step_number: int,
                         action: str,
                         expected: EX,
                         found_match: bool,
                         best_match: AE | None,
                         best_mismatches: list[str],
                         pass_expected: bool = False,
                         ) -> None:
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
        if not found_match and not best_match and not pass_expected:
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
        elif pass_expected:
            discrepancy_type = DiscrepancyType.PASS_EVENT.value
            details = ['Шаг намерено пропущен пользователем']
            discrepancies.append({
                'step': step_number,
                'action': action,
                'event': expected.name,
                'type': discrepancy_type,
                'details': details,
                'timestamp': best_match.timestamp if best_match else None
            })
        elif best_match and not found_match:
            discrepancy_type = DiscrepancyType.UNEXPECTED_EVENT.value
            details = ['Неожиданное событие']
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

    def _check_event_match(self,
                           expected: EX,
                           actual: AE,
                           ) -> tuple[bool, list[str]]:
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

    def _add_no_expect_events(self,
                              index: int,
                              table_data: list[list[str]],
                              html_data: HT,
                              action_detail: str,
                              formatted_action: str,
                              ) -> None:
        console_row = self._create_no_events_expected_row(index,
                                                          formatted_action,
                                                          )
        html_row = self._create_no_events_expected_html_row(index,
                                                            action_detail,
                                                            )
        table_data.append(console_row)
        html_data.append(html_row)

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
