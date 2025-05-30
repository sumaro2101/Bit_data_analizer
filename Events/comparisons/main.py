import logging
import subprocess
import sys
import os
import time
import pathlib
import pandas as pd
import webbrowser

from analizers.log_analizer import LogAnalyzer
from checkers.list_checker import ChecklistValidator


def find_checklist_file(checklist_dir):
    """
    Находит единственный .xlsx файл в папке checklist, игнорируя временные файлы Excel.

    Args:
        checklist_dir (pathlib.Path): Путь к папке checklist.

    Returns:
        pathlib.Path or None: Путь к единственному .xlsx файлу или None, если файл не найден
        или их больше одного.
    """
    xlsx_files = [f for f in checklist_dir.glob('*.xlsx') if not f.name.startswith('~$')]
    if len(xlsx_files) == 0:
        print("Ошибка: Чек-лист не найден в папке checklist.")
        return None
    elif len(xlsx_files) > 1:
        print("Ошибка: Обнаружено более одного .xlsx файла в папке checklist. Оставьте только один чек-лист.")
        return None
    return xlsx_files[0]


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
        current_dir = pathlib.Path(__file__).resolve().parent
        checklist_dir = current_dir.joinpath('checklist')

        # Проверка существования папки checklist
        if not checklist_dir.is_dir():
            print("Ошибка: Папка checklist не найдена.")
            sys.exit(1)

        # Поиск чек-листа
        checklist_path = find_checklist_file(checklist_dir)
        if checklist_path is None:
            print("Программа завершена из-за ошибки с чек-листом.")
            sys.exit(1)
        print(f"Используется чек-лист: {checklist_path.name}")

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
            raise e

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
                    analyzer = LogAnalyzer(log_path, checklist_path)
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
                                analyzer = LogAnalyzer(log_path, checklist_path)
                                analyzer.analyze()
                                continue  # Продолжаем внутренний цикл с меню выбора действий
                            except Exception as e:
                                print(f"Ошибка при анализе логов: {str(e)}")
                                raise Exception
                            continue  # В случае ошибки тоже остаемся во внутреннем цикле

                        # Просмотр HTML-отчета
                        elif choice.lower() == 'y':
                            html_report_path = current_dir.joinpath('HTMLreport.html')
                            if html_report_path.exists():
                                print(f"Попытка открыть файл: {html_report_path.as_posix()}")
                                try:
                                    if os.name == 'posix':  # macOS и Linux
                                        subprocess.run(['open', html_report_path.as_posix()], check=True)
                                    else:  # Windows и другие
                                        webbrowser.open_new_tab(html_report_path.as_posix())
                                    print("HTML отчёт открыт в браузере.")
                                except Exception as e:
                                    print(f"Ошибка при открытии отчета: {e}")
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
                    raise Exception
            else:
                print(f"Файл не найден: {log_path}\nПроверьте путь и попробуйте снова.")

    except FileNotFoundError as e:
        print(f"Ошибка: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Произошла неожиданная ошибка: {str(e)}")
        raise Exception


if __name__ == "__main__":
    main()
