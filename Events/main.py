import logging
import sys
import os
import time
import pathlib
import pandas as pd
import webbrowser

from table_config import NAME_CHECK_LIST
from analizers import LogAnalyzer
from checkers import ChecklistValidator


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
        checklist_path = current_dir.joinpath('checklist', NAME_CHECK_LIST)

        if not checklist_path.exists():
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
                                raise Exception
                            continue  # В случае ошибки тоже остаемся во внутреннем цикле

                        # Просмотр HTML-отчета
                        elif choice.lower() == 'y':
                            html_report_path = current_dir.joinpath('HTMLreport.html')
                            if html_report_path.exists():
                                webbrowser.open_new_tab(html_report_path.as_posix())
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
