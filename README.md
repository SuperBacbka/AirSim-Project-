AirSim Power Line Inspection Platform
Описание проекта

Разработка платформы управления БПЛА с использованием искусственного интеллекта для обнаружения и анализа объектов инфраструктуры линий электропередачи.

Проект разработан в рамках выпускной квалификационной работы по специальности 09.02.07 «Информационные системы и программирование».

Платформа реализует:

автономную навигацию БПЛА;
обход препятствий;
инспекцию объектов ЛЭП;
обнаружение изоляторов с использованием YOLO;
сопровождение объектов с использованием Deep SORT;
обучение поведения на основе Reinforcement Learning (PPO);
моделирование среды в Unreal Engine и AirSim.
_______________________________________________________________________________________________________
Архитектура системы
PPO Navigation Agent

Навигационный агент отвечает за:

следование маршруту;
достижение контрольных точек;
обход препятствий;
поддержание устойчивого полёта.
PPO Inspection Agent

Инспекционный агент отвечает за:

позиционирование относительно объекта;
удержание дистанции инспекции;
центрирование объекта в кадре;
выполнение сценария осмотра.
Computer Vision

Используемые технологии:

YOLO — обнаружение объектов;
YOLO Segmentation — сегментация объектов инфраструктуры;
Deep SORT — сопровождение целей между кадрами.
Simulation Environment

Среда моделирования построена на:

Unreal Engine 4.27<img width="1540" height="690" alt="EpicIcon" src="https://github.com/user-attachments/assets/82d2c4e8-2cc0-4403-b698-3af53a0e740b" />

Microsoft AirSim.
-------------------------------------------------------------------------------------------------------
Используемые технологии
Python
PyTorch
Stable-Baselines3
PPO
YOLO
Deep SORT
OpenCV
Unreal Engine 4.27
AirSim
-------------------------------------------------------------------------------------------------------
Логи обучения распологаются на google disk https://drive.google.com/drive/folders/1W8ik6xVuMvRt8Uut-oA1sqpvjz2fNqfW?usp=drive_link
-----------------------------------------------------------------------------------------------------------------
Проект Unreal Engine https://drive.google.com/drive/folders/1W8ik6xVuMvRt8Uut-oA1sqpvjz2fNqfW?usp=drive_link
-----------------------------------------------------------------------------------------------------------------

Инструкция: сборка Unreal Engine 4.27 (Source Build) под AirSim, создание проекта и подключение управления из Python
1. Цель и итоговая конфигурация
    Цель: получить стабильную связку UE 4.27 + AirSim + Python API без конфликтов компиляторов и ошибок линковки.
    Итоговая конфигурация:
    •	Unreal Engine: 4.27, собранный из исходников (Source Build)
    •	Компилятор: Visual Studio 2022 (MSVC v143)
    •	Windows SDK: 10.0.19041.0
    •	AirSim: плагин внутри проекта Plugins/AirSim
    •	Управление: Python (PyCharm) через airsim API
    •	Конфиг AirSim: Documents\AirSim\settings.json (в нашем случае — в OneDrive)

2. нужен UE 4.27 Source Build, а не Epic Launcher
    При использовании UE 4.27 из Epic Launcher часто возникает конфликт toolchain:
    •	AirSim (AirLib) собран VS2022 → в линковке появляется IL версии P1=20240319
    •	Проект/движок собирается VS2019/v142 → IL P2=20210202
    Итог: fatal error C1900 и LNK1257.
    Решение: собрать UE 4.27 из исходников в VS2022, чтобы и UE, и AirSim использовали один MSVC

3. Подготовка окружения (Windows)
    3.1 Visual Studio 2022
    В Visual Studio Installer включить:
    •	Workload: Desktop development with C++
    •	Компоненты:
    o	MSVC v143 x64/x86 build tools
    o	Windows 10 SDK 10.0.19041.0 (важно: UE4.27 стабильнее с 19041, чем с 26100)
    o	.NET Framework 4.6.2 Developer Pack (Targeting Pack)
    3.2 Git + доступ к EpicGames/UnrealEngine
    •	Git установлен
    •	GitHub привязан к Epic аккаунту
    •	Доступ к репозиторию EpicGames/UnrealEngine получен
________________________________________
4. Скачивание UE 4.27 из исходников
    Открыть Developer Command Prompt for VS 2022.
    Пример установки в D:\UE\:
    cd /d D:\
    mkdir UE
    cd /d D:\UE
    git clone -b 4.27 --single-branch https://github.com/EpicGames/UnrealEngine.git UE4.27-src
________________________________________
5. Setup и генерация проекта под VS2022
    cd /d D:\UE\UE4.27-src
    Setup.bat
    GenerateProjectFiles.bat -2022
    Типовая проблема: MSB3644 (.NET 4.6.2)
    Ошибка вида:
    MSB3644: не найдены ссылочные сборки для .NETFramework,Version=v4.6.2
    Решение: установить .NET Framework 4.6.2 Developer Pack, затем повторить GenerateProjectFiles.bat -2022.
________________________________________
6. Сборка UE4Editor
    6.1 Базовая команда сборки
    cd /d D:\UE\UE4.27-src
    Engine\Build\BatchFiles\Build.bat UE4Editor Win64 Development -WaitMutex -NoHotReload
    После успешной сборки появляется:
    D:\UE\UE4.27-src\Engine\Binaries\Win64\UE4Editor.exe
________________________________________
7. Ошибки компиляции UE4.27 в VS2022 и их устранение
    7.1 Ошибка C4756 в RenderGraphPrivate.cpp (INFINITY)
    Симптом:
    RenderGraphPrivate.cpp(173) : error C4756: переполнение при расчете констант
    Причина: использование INFINITY как compile-time константы.
    Исправление (патч):
    В файле:
    Engine\Source\Runtime\RenderCore\Private\RenderGraphPrivate.cpp
    Заменить INFINITY на:
    #include <limits>
    // ...
    case 3:
    {
        const float Inf = std::numeric_limits<float>::infinity();
        return FLinearColor(Inf, Inf, Inf, Inf);
    }
    (При необходимости аналогично для NAN: std::numeric_limits<float>::quiet_NaN().)
-----------------------------------------------------------------------------------------------------------------------------------------------
    7.2 Ошибки в тестовых/VR-плагинах (NetcodeUnitTest, SteamVR и т.п.)
    Некоторые плагины в UE4.27 дают ошибки на VS2022 (C4756/C4834 и др.). Для проекта AirSim они не обязательны.
    Практика ускорения сборки: временно отключать/перемещать такие плагины из Engine\Plugins (например в D:\UE\_DisabledPlugins\...), если они ломают сборку.
-----------------------------------------------------------------------------------------------------------------------------------------------
    7.3 Проблемы из-за Windows SDK 10.0.26100
    Если сборка использует 10.0.26100.0, возможны конфликты в winrt\wrl\event.h (C4668).
    Решение: принудительно использовать SDK 19041.
    Создать/отредактировать:
    Engine\Saved\UnrealBuildTool\BuildConfiguration.xml
    <?xml version="1.0" encoding="utf-8"?>
    <Configuration xmlns="https://www.unrealengine.com/BuildConfiguration">
      <WindowsPlatform>
        <WindowsSdkVersion>10.0.19041.0</WindowsSdkVersion>
      </WindowsPlatform>
    </Configuration>
-----------------------------------------------------------------------------------------------------------------------------------------------
8. Запуск собранного UE 4.27
    Запускать не через Epic Launcher, а напрямую:
    D:\UE\UE4.27-src\Engine\Binaries\Win64\UE4Editor.exe
-----------------------------------------------------------------------------------------------------------------------------------------------
9. Создание проекта AirSimLab (UE 4.27)
    9.1 Тип проекта
    Рекомендуется создавать проект как C++ (не Blueprint-only), чтобы плагины корректно компилировались.
    Параметры:
    •	Games → Blank
    •	C++
    •	Desktop/Console
    •	Raytracing Off
________________________________________
10. Подключение AirSim как плагина в проект
    1.	В проекте создать папку:
    <ProjectRoot>\Plugins\
    2.	Скопировать плагин AirSim:
    D:\Airsim\AirSim\Unreal\Plugins\AirSim
    →
    <ProjectRoot>\Plugins\AirSim
    3.	После копирования очистить артефакты проекта (по необходимости):
    •	<ProjectRoot>\Intermediate
    •	<ProjectRoot>\Binaries
    •	<ProjectRoot>\Plugins\AirSim\Intermediate
    •	<ProjectRoot>\Plugins\AirSim\Binaries
________________________________________
11. Сборка проекта с AirSim через Build.bat
    Важный нюанс: сборка из редактора иногда запускается с -NoEngineChanges и может блокировать обновление .modules.
    Надёжный способ — собрать проект вручную:
    cd /d D:\UE\UE4.27-src
    Engine\Build\BatchFiles\Build.bat AirSimLabEditor Win64 Development -Project="C:\...\AirSimLab.uproject" -WaitMutex -NoHotReload
    Если сборка прошла, затем проект открывается в редакторе без ошибок.
________________________________________
12. Конфигурация AirSim: settings.json (важно!)
    AirSim читает settings.json не из проекта, а из Documents\AirSim текущего пользователя.
    12.1 Если “Документы” в OneDrive
    Путь обычно:
    •	C:\Users\<User>\OneDrive\Documents\AirSim\settings.json
    (или ...\OneDrive\Документы\AirSim\settings.json)
    12.2 Конфиг для дрона (Multirotor)
    Чтобы при Play появлялся дрон, а не машина:
     "   
    {
      "SettingsVersion": 1.2,
      "SimMode": "Multirotor",
      "Vehicles": {
        "Drone1": {
          "VehicleType": "SimpleFlight"
        }
      }
    }
    "
    Если появляется автомобиль — значит SimMode не Multirotor или читается другой settings.json.
________________________________________
13. Запуск симуляции
    1.	Открыть проект в UE4Editor.exe (source build)
    2.	Открыть карту (например Minimal_Default)
    3.	Нажать Play
    4.	В Output Log искать AirSim / SimMode для подтверждения старта.
________________________________________
14. Подключение из PyCharm (Python API)
    14.1 Установка
    В виртуальном окружении:
    pip install airsim
    14.2 Минимальный тест взлёта
    import airsim
    
    client = airsim.MultirotorClient()
    client.confirmConnection()
    
    client.enableApiControl(True, "Drone1")
    client.armDisarm(True, "Drone1")
    
    client.takeoffAsync(vehicle_name="Drone1").join()
    client.landAsync(vehicle_name="Drone1").join()
    
    client.armDisarm(False, "Drone1")
    client.enableApiControl(False, "Drone1")
    print("OK")
    Правило: сначала Play в UE, потом запуск Python-скрипта.
________________________________________
15. Контрольные точки успешной установки
    1.	UE4Editor.exe запускается и открывает проект
    2.	В Output Log: Mounting plugin AirSim и StartupModule: AirSim plugin
    3.	При Play появляется дрон (не автомобиль)
    4.	Python confirmConnection() проходит, команды takeoff/land выполняются
________________________________________
16. Краткий список типовых проблем и решений
    •	C1900 / LNK1257: конфликт VS2019 vs VS2022 → решается UE Source Build под VS2022
    •	MSB3644 (.NET 4.6.2): поставить Developer Pack 4.6.2
    •	C4756 (INFINITY): заменить на std::numeric_limits<float>::infinity()
    •	WinSDK 26100 ошибки: принудительно использовать 19041
    •	-NoEngineChanges / “would modify .modules”: собирать проект вручную через Build.bat ... -Project=...
    •	Появляется авто вместо дрона: поправить settings.json (OneDrive Documents), SimMode=Multirotor
    
--------------------------------------------------------------------------------------------------------------------------------------------------
Методика развертывания стенда AirSim на Unreal Engine 4.27 (Source Build) с управлением из Python
Методика описывает развертывание программно-аппаратного стенда моделирования БПЛА на базе Microsoft AirSim и Unreal Engine 4.27, собранного из исходников, с целью обеспечения:
    •	устойчивой сборки плагина AirSim без конфликтов компиляторов;
    •	запуска симуляции Multirotor (дрон) в UE;
    •	управления дроном из Python (PyCharm) через AirSim API.
________________________________________
А.2 Обоснование выбора конфигурации (единый toolchain)
    При использовании Unreal Engine 4.27 из Epic Launcher и одновременной сборке AirSim (AirLib) в Visual Studio 2022 возможно возникновение конфликта промежуточного кода компилятора (IL) в линковке:
    •	AirLib собран MSVC (VS2022) → P1;
    •	проект/движок собирается MSVC (VS2019/v142) → P2;
    что приводит к ошибкам C1900 и LNK1257.
    Для устранения конфликтов применяется принцип “единого toolchain”:
    движок UE 4.27 + плагин AirSim + проект собираются одним компилятором (MSVC v143 из VS2022).
    Поэтому выбран UE 4.27 Source Build, собранный в VS2022.
________________________________________
А.3 Требования к окружению
    А.3.1 ПО
    •	Windows 10/11.
    •	Git.
    •	Visual Studio 2022.
    •	Python 3.x + PyCharm (или любой IDE).
    •	Репозитории:
    o	EpicGames/UnrealEngine (ветка 4.27)
    o	Microsoft/AirSim (актуальная ветка)
А.3.2 Компоненты Visual Studio 2022 (обязательные)
    В Visual Studio Installer включить:
    •	Workload: Desktop development with C++
    •	Components:
    o	MSVC v143 x64/x86 build tools
    o	Windows 10 SDK 10.0.19041.0 (рекомендуется именно этот SDK)
    o	.NET Framework 4.6.2 Developer Pack (Targeting Pack)
________________________________________
А.4 Сборка Unreal Engine 4.27 из исходников (Source Build)
    А.4.1 Клонирование исходников
    В Developer Command Prompt for VS 2022:
    cd /d D:\
    mkdir UE
    cd /d D:\UE
    git clone -b 4.27 --single-branch https://github.com/EpicGames/UnrealEngine.git UE4.27-src
А.4.2 Загрузка зависимостей и генерация проекта
    cd /d D:\UE\UE4.27-src
    Setup.bat
    GenerateProjectFiles.bat -2022
    Типовая проблема: MSB3644 (.NET Framework 4.6.2)
    Ошибка вида:
    MSB3644: не найдены ссылочные сборки для .NETFramework,Version=v4.6.2
    Решение: установить .NET Framework 4.6.2 Developer Pack, затем повторить GenerateProjectFiles.bat -2022.
А.4.3 Сборка UE4Editor
    cd /d D:\UE\UE4.27-src
    Engine\Build\BatchFiles\Build.bat UE4Editor Win64 Development -WaitMutex -NoHotReload
    Контроль: наличие файла
    D:\UE\UE4.27-src\Engine\Binaries\Win64\UE4Editor.exe.
________________________________________
А.5 Устранение критичных ошибок сборки UE4.27 в VS2022
    А.5.1 Ошибка C4756 (переполнение констант) — INFINITY/NAN
    Пример:
    RenderGraphPrivate.cpp(...) : error C4756: переполнение при расчете констант
    Причина: использование INFINITY/NAN как compile-time констант в выражениях.
    Решение: заменить на значения через std::numeric_limits:
    #include <limits>
    // ...
    const float Inf = std::numeric_limits<float>::infinity();
    const float NaN = std::numeric_limits<float>::quiet_NaN();
    (Применяется точечно в местах, где MSVC трактует макросы как переполнение.)
А.5.2 Проблемы Windows SDK 10.0.26100
    При использовании WinSDK 26100 возможны ошибки в winrt/wrl.
    Решение: принудительно использовать WinSDK 19041.
    Создать/изменить файл:
    Engine\Saved\UnrealBuildTool\BuildConfiguration.xml
    <?xml version="1.0" encoding="utf-8"?>
    <Configuration xmlns="https://www.unrealengine.com/BuildConfiguration">
      <WindowsPlatform>
        <WindowsSdkVersion>10.0.19041.0</WindowsSdkVersion>
      </WindowsPlatform>
    </Configuration>
________________________________________
А.6 Создание проекта UE 4.27 под AirSim
    Рекомендуемый формат проекта: C++, так как AirSim является C++ плагином.
    Параметры:
    •	Games → Blank
    •	C++
    •	Desktop/Console
    •	Raytracing: Off
________________________________________
А.7 Сборка AirSim и подключение плагина
    А.7.1 Сборка AirSim (VS2022)
    В Developer Command Prompt for VS 2022:
    cd /d D:\Airsim\AirSim
    build.cmd
    А.7.2 Подключение плагина к проекту
    1.	Создать папку:
    <ProjectRoot>\Plugins\
    2.	Скопировать:
    D:\Airsim\AirSim\Unreal\Plugins\AirSim
    →
    <ProjectRoot>\Plugins\AirSim
________________________________________
А.8 Сборка проекта с AirSim
    А.8.1 Рекомендованный способ (через Build.bat)
    Сборку выполнять вручную, чтобы избежать блокировки -NoEngineChanges и сообщений вида
    Building would modify UE4Editor.modules.
    cd /d D:\UE\UE4.27-src
    Engine\Build\BatchFiles\Build.bat AirSimLabEditor Win64 Development -Project="C:\...\AirSimLab.uproject" -WaitMutex -NoHotReload
    Контроль: отсутствие ERROR: и завершение со временем выполнения (Total execution time…).
________________________________________
А.9 Конфигурация AirSim (settings.json)
    А.9.1 Принцип размещения
    AirSim читает настройки не из проекта UE, а из папки Documents\AirSim текущего пользователя Windows.
    При переносе “Документы” в OneDrive путь обычно:
    •	C:\Users\<User>\OneDrive\Documents\AirSim\settings.json
    (или ...\OneDrive\Документы\AirSim\settings.json)
    А.9.2 Настройки для Multirotor (дрон)
    {
      "SettingsVersion": 1.2,
      "SimMode": "Multirotor",
      "Vehicles": {
        "Drone1": { "VehicleType": "SimpleFlight" }
      }
    }
    Контроль: при запуске Play в UE должен появиться дрон (а не автомобиль).
    Если появляется автомобиль — AirSim читает другой файл settings.json или SimMode не Multirotor.
________________________________________
А.10 Запуск симуляции и контроль через Output Log
    1.	Запустить редактор:
    D:\UE\UE4.27-src\Engine\Binaries\Win64\UE4Editor.exe
    2.	Открыть проект .uproject
    3.	Нажать Play
    4.	Открыть лог: Window → Developer Tools → Output Log
    5.	Фильтр по словам: AirSim, SimMode, Vehicle.
________________________________________
А.11 Подключение из Python (PyCharm)
    А.11.1 Установка пакета
    pip install airsim
    А.11.2 Минимальный тест
    import airsim
    
    client = airsim.MultirotorClient()
    client.confirmConnection()
    
    client.enableApiControl(True, "Drone1")
    client.armDisarm(True, "Drone1")
    
    client.takeoffAsync(vehicle_name="Drone1").join()
    client.landAsync(vehicle_name="Drone1").join()
    
    client.armDisarm(False, "Drone1")
    client.enableApiControl(False, "Drone1")
    print("OK")
    Правило: сначала Play в UE, затем запуск Python-скрипта.
________________________________________
А.12 Типовые проблемы и решения (выжимка)
    1.	C1900/LNK1257 → конфликт компиляторов → решается UE4.27 Source Build под VS2022.
    2.	MSB3644 → нет .NET 4.6.2 targeting pack → установить Developer Pack.
    3.	C4756 INFINITY/NAN → заменить на std::numeric_limits.
    4.	Ошибки WinSDK 26100 → принудить WinSDK 19041.
    5.	“would modify UE4Editor.modules / -NoEngineChanges” → собирать проект командой Build.bat ... -Project=... (без -NoEngineChanges).
    6.	Появляется авто вместо дрона → поправить settings.json (SimMode Multirotor) в OneDrive Documents.
________________________________________
Приложение Б
    Схема программного взаимодействия стенда (пайплайн)
    Unreal Engine 4.27 (Source Build)
    → загрузка плагина AirSim (C++ module)
    → чтение Documents\AirSim\settings.json (SimMode, Vehicles, sensors)
    → запуск симуляции (Play) и инициализация vehicle (Drone1)
    → поднятие RPC/API AirSim
    → Python клиент (airsim.MultirotorClient)
    → команды управления (arm, takeoff, move, land)
    → обратные данные (state, images, IMU, barometer и т.д.)
