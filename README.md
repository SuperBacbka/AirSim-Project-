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

Unreal Engine 4.27;
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
    
