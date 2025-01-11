import os
import re
from typing import List, Optional, Tuple, Dict, Set

# Определение цветов как констант
BLUE = "\033[94m"  # Синий
GREEN = "\033[92m"  # Зеленый
YELLOW = "\033[93m"  # Желтый
RESET = "\033[0m"  # Сброс цвета
BRIGHT_RED = "\033[91m"  # Ярко-красный для "Потрачено"
BRIGHT_GREEN = "\033[32m"  # Ярко-зеленый для "Получено"
CYAN = "\033[96m"  # Бирюзовый для Product
GOLD = "\033[93m"  # Золотой для особых значений
RED = "\033[91m"  # Красный для Failed событий


def find_bank_rules_before(lines: List[str], current_index: int, currency_name: str) -> Optional[Tuple[str, str, str]]:
    """
    Ищет последний BankRules лог перед указанной строкой для конкретной валюты.

    :param lines: Все строки лога
    :param current_index: Текущий индекс для CurrencyChanged события
    :param currency_name: Название валюты для поиска
    :return: Кортеж (before_value, after_value, item_type) или None
    """
    # Получаем текущее время события
    current_line = lines[current_index]
    current_time_match = re.match(r'\[(.*?)]', current_line)
    if not current_time_match:
        return None

    for i in range(current_index - 1, -1, -1):
        line = lines[i]

        # Проверяем, что это лог BankRules для нужной валюты
        match = re.match(
            r'\[(.*?)] Log : BankRules\.\s*(Take|Give)\.\s*Item:\s*(\w+),\s*Count:\s*(\d+),\s*Before:\s*(\d+),\s*After:\s*(\d+)',
            line
        )

        if match and match.group(3) == currency_name:
            # Нашли нужную валюту
            return match.group(5), match.group(6), match.group(3)

    return None


def clean_string(line: str, substrings_to_remove: List[str]) -> str:
    """
    Удаляет указанные подстроки и всё, что находится справа от них на той же строке.
    Также заменяет '[window | None]' на '[window | Закрыто]'.

    :param line: Исходная строка
    :param substrings_to_remove: Список подстрок для удаления
    :return: Очищенная строка
    """
    for substring in substrings_to_remove:
        if substring in line:
            line = line.split(substring)[0]
    line = line.strip()
    return line.replace('[window | None]', '[window | Закрыто]')


def add_colors(line: str, substrings_to_remove: List[str]) -> str:
    """
    Добавляет цвета к различным частям строки лога для вывода в терминал.
    """
    # Очищаем строку
    line = clean_string(line, substrings_to_remove)

    # Добавляем цвет к дате и времени
    time_match = re.match(r'(\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2}:\d{2})', line)
    if time_match:
        date_time = time_match.group(1)
        rest_of_line = line[len(date_time):].strip()
        colored_date_time = f"{BLUE}{date_time}{RESET}"
    else:
        return line

    # Стандартная обработка для событий
    event_name_match = re.search(r'EventName:\s*(\w+)', rest_of_line)
    if event_name_match:
        event_name = event_name_match.group(1)
        # Определяем цвет в зависимости от типа события
        if event_name in ["StoreInitializationFailed", "PurchaseCanceled", "PurchaseFailed"]:
            color = BRIGHT_RED
        else:
            color = GREEN
        # Применяем цвет к имени события
        replacement = f"EventName: {color}{event_name}{RESET}"
        rest_of_line = re.sub(r'EventName:\s*\S+', replacement, rest_of_line)

    # Специальная обработка для CurrencyChanged событий
    if "CurrencyChanged" in line:
        rest_of_line = re.sub(
            r'\[eventType \| sink]',
            f'[eventType | {RED}Списание{RESET}]',
            rest_of_line
        )
        rest_of_line = re.sub(
            r'\[eventType \| source]',
            f'[eventType | {BRIGHT_GREEN}Начисление{RESET}]',
            rest_of_line
        )
        rest_of_line = re.sub(
            r'\[currencyName \| ([^]]+)]',
            f'[currencyName | {GREEN}\\1{RESET}]',
            rest_of_line
        )

    # Добавляем цвет к остальным параметрам в квадратных скобках
    rest_of_line = re.sub(r'(?<!\033)\[([^]]+)]', f'{YELLOW}[\\1]{RESET}', rest_of_line)

    return f"{colored_date_time} {rest_of_line}"


def check_duplicate_parameters(params_list: List[Tuple[str, str]]) -> Set[str]:
    """
    Проверяет наличие дублирующихся параметров в списке параметров.

    :param params_list: Список кортежей параметров (ключ, значение)
    :return: Множество дублирующихся параметров
    """
    param_count: Dict[str, int] = {}
    duplicates: Set[str] = set()

    for key, _ in params_list:
        param_count[key] = param_count.get(key, 0) + 1
        if param_count[key] > 1:
            duplicates.add(key)

    return duplicates


def clean_log_line(lines: List[str], index: int, pass_events: int, filtered_lines) -> Optional[Tuple[str, List[Tuple[str, str]]]]:
    """
    Очищает строки лога, оставляя только дату, время, тип события и параметры.
    """
    line = lines[index]

    # Извлекаем дату и время из квадратных скобок
    match = re.match(r'\[(.*?)]', line)
    if not match:
        return None, pass_events

    date_time = match.group(1)
    rest_of_line = line[match.end():].strip()

    # Проверяем различные форматы событий
    telegram_send_params = ['Send.', 'Start send file to telegram']
    event_info = None
    is_currency_changed = False
    is_keys_changed = False
    is_store_init = False  # Общий флаг для StoreInitialization событий
    telegram_send = False

    # # Обработка IAPPurchaseStarted
    # if "BusinessEvents. IAPPurchaseStarted" in rest_of_line:
    #     event_info = "IAPPurchaseStarted"
    # # Пропускаем аналитику покупок, так как это дублирует бизнес события
    # elif "AnalyticsManager. CustomEvents. IAPPurchaseStarted" in line:
    #     return None

    # Обработка KeysChanged
    if "CustomEvents. KeysChanged" in rest_of_line:
        event_info = "EventName: KeysChanged"
        is_keys_changed = True
        telegram_send = True
    # Обработка других событий
    elif "EventName:" in rest_of_line:
        event_index = rest_of_line.find("EventName:")
        event_info = rest_of_line[event_index:].replace("AnalyticsManager. CustomEvents. CustomEvent. ", "")
        if "CurrencyChanged" in event_info:
            is_currency_changed = True
        telegram_send = True
        # if "StoreInitializationCompleted" in event_info or "StoreInitializationFailed" in event_info:
        #     is_store_init = True
    elif 'TelegramChatLogBehaviour' in rest_of_line:
        if (telegram_send_params[0] in rest_of_line and
           telegram_send_params[-1] in rest_of_line):
            pass_events += 1
            return None, pass_events

    elif "CustomEvents. CurrencyChanged" in rest_of_line:
        event_info = "EventName: CurrencyChanged"
        is_currency_changed = True
        telegram_send = True
    # elif "CustomEvents. IAPP" in rest_of_line:
    #     event_info = "IAPPurchaseStarted"
    # elif "BusinessEvents. StoreInitializationCompleted" in rest_of_line:
    #     event_info = "EventName: StoreInitializationCompleted"
    #     is_store_init = True
    # elif "BusinessEvents. StoreInitializationFailed" in rest_of_line:
    #     event_info = "EventName: StoreInitializationFailed"
    #     is_store_init = True

    if not event_info:
        return None, pass_events

    # Если это StoreInitialization события, возвращаем сразу без параметров
    if telegram_send and pass_events > 2:
        formatted_line = '26.10.1969 00:00:00 | EventName: Ожидаемое количество пропущенных шагов чек листа'
        params_list = dict(countPassed=pass_events - 2)
        formatted_line += f" | Params:{format_event_dict(params_list)}"
        filtered_lines.append(formatted_line + "\n")
        filtered_lines.append("\n")
        pass_events = 0
    else:
        pass_events = 0
    if is_store_init:
        return f"{date_time} | {event_info}", []

    # Объединяем все строки до ParamsDict: в одну строку
    full_event_info = event_info
    for i in range(index + 1, len(lines)):
        if "ParamsDict:" in lines[i]:
            break
        if not any(tech in lines[i] for tech in ['Gammister.', 'System.', 'Zenject.', 'UnityEngine.']):
            full_event_info += " " + lines[i].strip()

    # Извлекаем имя события
    event_name_match = None
    if "EventName:" in full_event_info:
        event_name_match = re.search(r'EventName:\s*(\w+)', full_event_info)
    if event_name_match:
        event_name = event_name_match.group(1)
        event_info = f"EventName: {event_name}"
    else:
        # Для CurrencyChanged, KeysChanged и IAPP событий используем уже подготовленный event_info
        pass

    params_list = []
    params_started = False
    bank_rules_info = None

    # Ищем ParamsDict в текущей и следующих строках
    for i in range(index, len(lines)):
        current_line = lines[i].strip()

        if "ParamsDict:" in current_line:
            params_started = True
            params_line = current_line.split("ParamsDict:", 1)[1].strip()
        elif params_started and not any(
                tech in current_line for tech in ['Gammister.', 'System.', 'Zenject.', 'UnityEngine.']):
            if current_line.startswith("["):
                params_line = current_line
            else:
                break
        else:
            if params_started:
                break
            continue

        if params_started:
            # Если это CurrencyChanged, найдем соответствующий BankRules лог
            if is_currency_changed and bank_rules_info is None:
                currency_match = re.search(r'\[currencyName \| ([^]]+)]', params_line)
                if currency_match:
                    currency_name = currency_match.group(1)
                    bank_rules_info = find_bank_rules_before(lines, index, currency_name)
            # Если это KeysChanged, найдем соответствующий BankRules лог
            elif is_keys_changed and bank_rules_info is None:
                key_match = re.search(r'\[keyName \| ([^]]+)]', params_line)
                if key_match:
                    key_name = key_match.group(1)
                    bank_rules_info = find_bank_rules_before(lines, index, key_name)

            params = re.findall(r'\[([^]]+)]', params_line)
            for param in params:
                parts = param.split('|', 1)
                if len(parts) == 2:
                    key, value = parts
                    key = key.strip()
                    value = value.strip()

                    # Специальная обработка для CurrencyChanged и KeysChanged
                    if is_currency_changed or is_keys_changed:
                        if key == 'eventType':
                            if value == 'sink':
                                value = 'Списание'
                            elif value == 'source':
                                value = 'Начисление'
                        elif key == 'balance' and bank_rules_info:
                            params_list.append((key, value))
                            params_list.append(('before', bank_rules_info[0]))
                            continue

                    params_list.append((key, value))

    return (f"{date_time} | {event_info}", params_list), pass_events


def format_event_dict(params_dict: Dict[str, str], indent: int = 58, with_colors: bool = False) -> str:
    """
    Форматирует параметры события, разделяя их запятыми и переносами строк.

    :param params_dict: Словарь с параметрами события
    :param indent: Количество пробелов для отступа (по умолчанию 58)
    :param with_colors: Флаг для добавления цветов в вывод
    :return: Отформатированная строка с параметрами события
    """
    if not params_dict:
        return ""

    if with_colors:
        formatted_params = "\n" + ",\n".join(
            f"{'':{indent}}{YELLOW}[{key} | {value}]{RESET}"
            for key, value in params_dict.items()
        )
    else:
        formatted_params = "\n" + ",\n".join(
            f"{'':{indent}}[{key} | {value}]"
            for key, value in params_dict.items()
        )
    return formatted_params


def parse_user_id(line: str) -> str:
    """
    Извлекает UserID из строки лога.

    :param line: Строка лога, содержащая информацию о пользователе
    :return: Извлеченный UserID или "Unknown User", если ID не найден
    """
    match = re.search(r'Nickname":"(USER_[^"]+)"', line)
    if match:
        return match.group(1)
    return "Unknown User"


def process_different_results(line: str, lines: List[str], index: int) -> Optional[Tuple[str, str]]:
    """
    Обрабатывает логи с ошибками рассинхронизации между клиентом и сервером.
    """
    error_marker = "The client's result is different from the server's result"
    if error_marker not in line:
        return None

    # Извлекаем дату и время
    date_match = re.match(r'\[(.*?)]', line)
    if not date_match:
        return None
    date_time = date_match.group(1)

    # Находим операцию перед маркером ошибки
    clean_line = line[line.find('] ') + 2:].strip()
    if clean_line.startswith('Error : '):
        clean_line = clean_line[8:].strip()

    before_error = clean_line[:clean_line.find(error_marker)].strip()
    # Берём всё до маркера ошибки, убираем точку в конце если она есть
    operation = before_error.rstrip('.')

    # Собираем данные клиента и сервера
    client_data = []
    server_data = []
    current_section = None

    i = index + 1
    while i < len(lines):
        current_line = lines[i].strip()

        if any(tech in current_line for tech in ['Gammister.', 'Firebase.', '<>', 'System.']):
            i += 1
            continue

        if current_line.startswith('Client:'):
            current_section = client_data
        elif current_line.startswith('Server:'):
            current_section = server_data
        elif current_section is not None and current_line:
            current_section.append(current_line)

        if client_data and server_data and not current_line:
            break

        i += 1

    if not client_data or not server_data:
        return None

    # Форматируем вывод для файла
    file_output = (
        f"{date_time} {operation}. {error_marker}\n"
        f"Client:\n{''.join(client_data)}\n"
        f"Server:\n{''.join(server_data)}"
    )

    # Форматируем вывод для консоли с цветами
    console_output = (
        f"{BLUE}{date_time}{RESET} {BRIGHT_RED}{operation}. {error_marker}{RESET}\n"
        f"{YELLOW}Client:{RESET}\n{''.join(client_data)}\n"
        f"{YELLOW}Server:{RESET}\n{''.join(server_data)}"
    )

    return file_output, console_output


def sanitize_path(path: str) -> str:
    """
    Очищает путь от кавычек и лишних пробелов, обрабатывает экранированные символы.

    :param path: Исходный путь
    :return: Очищенный путь
    """
    # Удаляем начальные и конечные пробелы
    path = path.strip()

    # Если путь в кавычках, удаляем их
    if (path.startswith('"') and path.endswith('"')) or (path.startswith("'") and path.endswith("'")):
        path = path[1:-1]

    # Заменяем двойные обратные слеши на одинарные
    path = path.replace('\\\\', '\\')

    # Обрабатываем экранированные кавычки внутри пути
    path = path.replace('\\"', '"').replace("\\'", "'")

    # Удаляем любые оставшиеся кавычки
    path = path.replace('"', '').replace("'", '')

    return path


def process_side_quest_info(line: str) -> Optional[Tuple[str, str, bool]]:
    """
    Обрабатывает строку лога, связанную с прогрессом сайд-квеста.

    :param line: Строка лога для обработки.
    :return: Кортеж из двух строк (для файла и для консоли) с информацией о прогрессе сайд-квеста
             и флага, указывающего, ненулевой ли прогресс, или None, если информация не найдена.
    """
    quest_progress_match = re.search(r'"QuestProgressResult":\s*{([^}]+)}', line)
    if quest_progress_match:
        quest_progress_str = quest_progress_match.group(1)

        progress_match = re.search(r'"Progress":(\d+)', quest_progress_str)
        side_quest_type_match = re.search(r'"SideQuestType":(\d+)', quest_progress_str)
        have_reward_match = re.search(r'"HaveReward":(true|false)', quest_progress_str)

        progress = int(progress_match.group(1)) if progress_match else 0
        side_quest_type_num = int(side_quest_type_match.group(1)) if side_quest_type_match else -1
        have_reward = have_reward_match and have_reward_match.group(1) == "true"

        side_quest_types = {
            0: "Тузы",
            1: "Стриты",
            2: "Флеши"
        }
        side_quest_type = side_quest_types.get(side_quest_type_num, "Неизвестный тип")

        reward_info = ""
        if have_reward:
            rewards_match = re.search(r'"RewardsResult":\s*{([^}]+)}', line)
            if rewards_match:
                rewards_str = rewards_match.group(1)
                game_rewards_match = re.search(r'"GameRewards":\s*\[([^]]*)]', rewards_str)
                if game_rewards_match and game_rewards_match.group(1).strip():
                    reward_info = f", Награда: {game_rewards_match.group(1)}"

        result_for_file = f"Получено очков сайд-квеста: {progress}, Тип сайд квеста: {side_quest_type}{reward_info}"
        result_for_console = f"{BRIGHT_GREEN}Получено{RESET} очков сайд-квеста: {progress}, Тип сайд квеста: {side_quest_type}{reward_info}"
        return result_for_file, result_for_console, progress > 0
    return None


# def process_iap_error(line: str) -> Optional[Tuple[str, str]]:
#     """
#     Обрабатывает логи с ошибками IAP покупок.
#     """
#     is_iap_failed = "IAPPurchaseFailed" in line
#     is_shop_failed = "Error : ShopController. OnPurchaseFailed" in line
#
#     if not (is_iap_failed or is_shop_failed):
#         return None
#
#     date_match = re.match(r'\[(.*?)]', line)
#     if not date_match:
#         return None
#     date_time = date_match.group(1)
#
#     try:
#         params: List[Dict[str, str]] = []
#         event_name = "IAP Purchase Failed" if is_iap_failed else "Shop Purchase Failed"
#
#         if is_shop_failed:
#             mode_match = re.search(r'PurchaseMode:\s*([^,]+)', line)
#             if mode_match:
#                 params.append({"key": "mode", "value": mode_match.group(1).strip()})
#
#         # Общие параметры для обоих типов
#         product_match = re.search(r'Product(?:Id)?:\s*\[?([^,\]]+)]?', line)
#         reason_match = re.search(r'(?:Purchase failure )?[Rr]eason:\s*([^,]+)', line)
#         message_match = re.search(r'(?:Purchase failure )?[Mm]essage:\s*([^,\n]+)', line)
#
#         if product_match:
#             params.append({"key": "product_id", "value": product_match.group(1).strip()})
#         if reason_match:
#             params.append({"key": "reason", "value": reason_match.group(1).strip()})
#         if message_match and message_match.group(1).strip():
#             params.append({"key": "message", "value": message_match.group(1).strip()})
#
#         # Преобразуем список словарей в словарь для format_event_dict
#         params_dict = {param["key"]: param["value"] for param in params}
#
#         # Форматируем вывод для файла
#         file_output = f"{date_time} | {event_name}"
#         if params_dict:
#             file_output += f" | Params:{format_event_dict(params_dict)}"
#
#         # Форматируем вывод для консоли с цветами
#         console_output = f"{BLUE}{date_time}{RESET} | {BRIGHT_RED}{event_name}{RESET}"
#         if params_dict:
#             console_output += f" | Params:{format_event_dict(params_dict, with_colors=True)}"
#
#         return file_output, console_output
#
#     except Exception:
#         return None


def process_seasons_rules(line: str) -> Optional[Tuple[str, str]]:
    """
    Обрабатывает строку лога, связанную с правилами сезонов.

    :param line: Строка лога для обработки
    :return: Кортеж из строк (для файла, для консоли) или None
    """
    match = re.match(r'\[(.*?)] Log : SeasonsRules\. AddExp\. value:\s*(\d+)', line)
    if not match:
        return None

    date_time = match.group(1)
    value = match.group(2)

    # Версия для файла
    file_output = f"{date_time} | Seasons AddExp | Params:\n"
    file_output += f"                                                          [value | {value}]"

    # Версия для консоли с цветами
    console_output = f"{BLUE}{date_time}{RESET} | {GREEN}Seasons AddExp{RESET} | Params:\n"
    console_output += f"                                                          {GREEN}[value | {value}]{RESET}"

    return file_output, console_output


def process_deferred_purchase(line: str) -> Optional[Tuple[str, str]]:
    """
    Обрабатывает строку лога, связанную с отложенными платежами.

    :param line: Строка лога для обработки
    :return: Кортеж из строк (для файла, для консоли) или None
    """
    # Проверяем начало отложенного платежа
    deferred_start = "IAPService. OnDeferredPurchase" in line and "Product purchasing is deferred" in line
    # Проверяем успешное завершение отложенного платежа
    deferred_success = "IAPService. ProcessValidatedPurchaseAsync. Successfully validated Product:" in line

    if not (deferred_start or deferred_success):
        return None

    date_match = re.match(r'\[(.*?)]', line)
    if not date_match:
        return None
    date_time = date_match.group(1)

    if deferred_start:
        # Извлекаем информацию о продукте и режиме
        product_match = re.search(r'Product: \[(.*?)]', line)
        mode_match = re.search(r'PurchaseMode: (\w+)', line)
        if not (product_match and mode_match):
            return None

        params_dict = {
            "product": product_match.group(1),
            "mode": mode_match.group(1)
        }

        # Версия для файла
        file_output = f"{date_time} | Purchase Deferred | Params:{format_event_dict(params_dict)}"

        # Версия для консоли с цветами
        console_output = f"{BLUE}{date_time}{RESET} | {YELLOW}Purchase Deferred{RESET} | Params:{format_event_dict(params_dict, with_colors=True)}"

    else:  # deferred_success
        # Извлекаем информацию о продукте
        product_match = re.search(r'Product: \[(.*?)]', line)
        if not product_match:
            return None

        params_dict = {
            "product": product_match.group(1),
            "status": "Success"
        }

        # Версия для файла
        file_output = f"{date_time} | Purchase Complete | Params:{format_event_dict(params_dict)}"

        # Версия для консоли с цветами
        console_output = f"{BLUE}{date_time}{RESET} | {GREEN}Purchase Complete{RESET} | Params:{format_event_dict(params_dict, with_colors=True)}"

    return file_output, console_output


def process_logs(file_path: str) -> Tuple[str, str, str, str, str]:
    """
    Обрабатывает файл логов, фильтрует и форматирует логи, сохраняет результат в новый файл.

    :param file_path: Путь к файлу с исходными логами
    :return: Кортеж с UserID, версией игры, вариантом AB-тестирования, версией конфигурации AB-тестирования и путем к выходному файлу
    """
    # Дополнительная проверка и очистка пути
    file_path = sanitize_path(file_path)

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines: List[str] = file.readlines()
    except FileNotFoundError:
        print(f"Файл не найден: {file_path}")
        print("Проверьте путь и попробуйте снова.")
        return "Unknown User", "Unknown Version", "Unknown Variant", "Unknown Config Version", ""
    except PermissionError:
        print(f"Нет доступа к файлу: {file_path}")
        print("Проверьте права доступа и попробуйте снова.")
        return "Unknown User", "Unknown Version", "Unknown Variant", "Unknown Config Version", ""
    except Exception as e:
        print(f"Произошла ошибка при открытии файла: {e}")
        print(f"Путь к файлу: {file_path}")
        return "Unknown User", "Unknown Version", "Unknown Variant", "Unknown Config Version", ""

    filtered_lines: List[str] = []
    processed_user_id = "Unknown User"
    processed_game_version = "Unknown Version"
    processed_ab_testing_variant = "Unknown Variant"
    processed_ab_testing_game_config_version = "Unknown Config Version"
    pass_events = 0
    substrings_to_remove = [
        "____", "GameAnalyticsSDK", 'Firebase', 'ParamsDict', '<Animate',
        "No boobs", ' Log : ', 'Gammister.FunLand'
    ]
    events_with_duplicates: Dict[str, Set[str]] = {}

    for i, line in enumerate(lines):
        # Проверяем на ошибки рассинхронизации
        different_results = process_different_results(line, lines, i)
        if different_results:
            file_output, console_output = different_results
            filtered_lines.append(file_output + "\n")
            filtered_lines.append("\n")
            print(console_output + "\n")
            continue

        # Проверяем на SeasonsRules
        seasons_rules = process_seasons_rules(line)
        if seasons_rules:
            file_output, console_output = seasons_rules
            filtered_lines.append(file_output + "\n")
            filtered_lines.append("\n")
            print(console_output + "\n")
            continue

        # Добавляем обработку информации о сайд-квестах
        side_quest_info = process_side_quest_info(line)
        if side_quest_info:
            info_for_file, info_for_console, non_zero_progress = side_quest_info
            if non_zero_progress:
                filtered_lines.append(f"{info_for_file}\n")
                filtered_lines.append("\n")
                print(f"{info_for_console}\n")
            continue

        # Проверяем на отложенные платежи
        deferred_purchase = process_deferred_purchase(line)
        if deferred_purchase:
            file_output, console_output = deferred_purchase
            filtered_lines.append(file_output + "\n")
            filtered_lines.append("\n")
            print(console_output + "\n")
            continue

        if "Nickname" in line and "USER_" in line:
            processed_user_id = parse_user_id(line.strip())
        if '"GameState":{"Version":"' in line:
            match = re.search(r'"GameState":\{"Version":"([^"]+)"', line)
            if match:
                processed_game_version = match.group(1)
        if '"ABTestingVariant":"' in line:
            match = re.search(r'"ABTestingVariant":"([^"]+)"', line)
            if match:
                processed_ab_testing_variant = match.group(1)
        if '"ABTestingGameConfigVersion":"' in line:
            match = re.search(r'"ABTestingGameConfigVersion":"([^"]+)"', line)
            if match:
                processed_ab_testing_game_config_version = match.group(1)

        # Проверяем на ошибки IAP
        # iap_error = process_iap_error(line)
        # if iap_error:
        #     file_output, console_output = iap_error
        #     filtered_lines.append(file_output + "\n")
        #     filtered_lines.append("\n")
        #     print(console_output + "\n")
        #     continue

        if not any(exclude in line.lower() for exclude in
                   ['unityengine', 'gammister.funland', 'system', 'spine.', ' syste',
                    '--------- beginning of main', 'zenject', 'Firebase']):
            cleaned_line, pass_events = clean_log_line(lines, i, pass_events, filtered_lines)
            if cleaned_line:
                event_info, params_list = cleaned_line
                duplicates = check_duplicate_parameters(params_list)
                if duplicates:
                    event_name = event_info.split("|")[1].strip().split()[1]  # Извлекаем имя события
                    if event_name not in events_with_duplicates:
                        events_with_duplicates[event_name] = set()
                    events_with_duplicates[event_name].update(duplicates)
                formatted_line = event_info
                if params_list:
                    params_dict = dict(params_list)  # Преобразуем список в словарь для форматирования
                    formatted_line += f" | Params:{format_event_dict(params_dict)}"
                filtered_lines.append(formatted_line + "\n")
                filtered_lines.append("\n")
                print(add_colors(formatted_line, substrings_to_remove) + "\n")

    processed_output_file: str = os.path.splitext(file_path)[0] + '_cleaned.txt'

    cleaned_lines = [clean_string(line, substrings_to_remove).replace('[window | None]', '[window | Закрыто]') + '\n'
                     for line in filtered_lines]

    # Добавляем предупреждение о дублировании параметров в начало файла
    if events_with_duplicates:
        warning_lines = ["Внимание! Обнаружено дублирование параметров. Обратите внимание на ивенты:"]
        for event, dupes in events_with_duplicates.items():
            warning_lines.append(f"  {event}: {', '.join(sorted(dupes))}")
        warning_for_file = "\n".join(warning_lines) + "\n\n"
        warning_for_console = f"{BRIGHT_RED}{warning_for_file}{RESET}"
        cleaned_lines.insert(0, warning_for_file)
        print(warning_for_console)

    # Добавляем информацию в начало файла
    cleaned_lines.insert(0, f"UserID: {processed_user_id}\n")
    cleaned_lines.insert(1, f"Version: {processed_game_version}\n")
    cleaned_lines.insert(2, f"ABTestingVariant: {processed_ab_testing_variant}\n")
    cleaned_lines.insert(3, f"ABTestingGameConfigVersion: {processed_ab_testing_game_config_version}\n\n")

    with open(processed_output_file, 'w', encoding='utf-8') as file:
        file.writelines(cleaned_lines)

    return (processed_user_id, processed_game_version, processed_ab_testing_variant,
            processed_ab_testing_game_config_version, processed_output_file)


if __name__ == "__main__":
    while True:
        # Получаем путь к входному файлу от пользователя
        input_file_path: str = input("Введите путь к файлу с логами (или 'q' для выхода): ")

        if input_file_path.lower() == 'q':
            print("Программа завершена.")
            break

        # Очищаем путь от кавычек и лишних пробелов
        input_file_path = sanitize_path(input_file_path)

        # Проверяем, существует ли файл
        if os.path.isfile(input_file_path):
            # Обрабатываем логи
            user_id, game_version, ab_testing_variant, ab_testing_game_config_version, output_file = process_logs(
                input_file_path)

            # Выводим информацию один раз после обработки
            print(f"\nUserID: {user_id}")
            print(f"Version: {game_version}")
            print(f"ABTestingVariant: {ab_testing_variant}")
            print(f"ABTestingGameConfigVersion: {ab_testing_game_config_version}")
            if output_file:
                print(f"Обработанные логи сохранены в: {output_file}")
        else:
            print(f"Файл не найден: {input_file_path}")
            print("Проверьте путь и попробуйте снова.")

        print()  # Добавляем пустую строку для лучшей читаемости вывода
