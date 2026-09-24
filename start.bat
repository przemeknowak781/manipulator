@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1
title LeRobot 101 x MediaPipe - sterowanie gestami

rem ===================================================================
rem  Uruchamiacz dla Windows.
rem
rem  Klikniecie dwukrotne otwiera menu. Mozna tez wywolac z argumentami,
rem  ktore ida prosto do aplikacji, np.:
rem      start.bat --port COM5 --mode ik
rem
rem  Przy pierwszym uruchomieniu tworzy srodowisko .venv (Python 3.12,
rem  jesli jest) i instaluje zaleznosci w wersjach z constraints.txt.
rem  Kolejne starty sa juz natychmiastowe. Panel blizniaka (opcja 7)
rem  doinstalowuje swoje pakiety [twin,feetech] przy pierwszym uzyciu.
rem ===================================================================

cd /d "%~dp0"

rem --- kolory ANSI [Windows 10+]; na starszych po prostu znikaja ------
for /F %%a in ('echo prompt $E ^| cmd') do set "E=%%a"
set "C_TITLE=%E%[1;33m"
set "C_OK=%E%[1;32m"
set "C_ERR=%E%[1;31m"
set "C_DIM=%E%[90m"
set "C_OFF=%E%[0m"

set "VENV=%~dp0.venv"
set "VENV_PY=%VENV%\Scripts\python.exe"

call :banner
call :ensure_python || goto :fail
call :ensure_venv   || goto :fail

rem --- argumenty podane wprost - pomijamy menu -----------------------
if not "%~1"=="" (
    call :run %*
    goto :end
)

:menu
echo.
echo   %C_TITLE%Co uruchomic?%C_OFF%
echo.
echo     %C_OK%1%C_OFF%  Symulator            %C_DIM%- bez robota, do nauki gestow%C_OFF%
echo     %C_OK%2%C_OFF%  Prawdziwe ramie      %C_DIM%- SO-101 podlaczone przez USB%C_OFF%
echo     %C_OK%3%C_OFF%  Symulator + tryb IK  %C_DIM%- sterowanie kartezjanskie%C_OFF%
echo     %C_OK%4%C_OFF%  Demo z pliku wideo   %C_DIM%- bez kamery%C_OFF%
echo     %C_OK%5%C_OFF%  Diagnostyka          %C_DIM%- testy i wypis konfiguracji%C_OFF%
echo     %C_OK%6%C_OFF%  Napraw srodowisko    %C_DIM%- przeinstaluj zaleznosci%C_OFF%
echo     %C_OK%7%C_OFF%  Panel blizniaka      %C_DIM%- symulacja MuJoCo w przegladarce (lerobot-twin ui)%C_OFF%
echo     %C_OK%0%C_OFF%  Wyjscie
echo.
set "PICK="
set /p "PICK=  Wybor: "

rem Kazda opcja skacze do wlasnej etykiety. Zapis `if ... cmd & goto`
rem wygladalby krocej, ale `&` NIE nalezy do `if` - skok wykonalby sie
rem zawsze i opcja "0" nigdy by nie zadzialala.
if "%PICK%"=="0" goto :end
if "%PICK%"=="1" goto :m_sim
if "%PICK%"=="2" goto :m_real
if "%PICK%"=="3" goto :m_ik
if "%PICK%"=="4" goto :m_video
if "%PICK%"=="5" goto :m_diag
if "%PICK%"=="6" goto :m_fix
if "%PICK%"=="7" goto :m_twin
echo   %C_ERR%Nie znam opcji "%PICK%".%C_OFF%
goto :menu

:m_sim
call :run
goto :menu
:m_real
call :run_real
goto :menu
:m_ik
call :run --mode ik
goto :menu
:m_video
call :run_video
goto :menu
:m_diag
call :diagnose
goto :menu
:m_fix
call :install force
goto :menu
:m_twin
call :run_twin
goto :menu

rem ===================================================================
:banner
echo.
echo   %C_TITLE%LeRobot 101 x MediaPipe%C_OFF%
echo   %C_DIM%sterowanie ramieniem SO-101 gestami dloni%C_OFF%
echo.
echo   %C_DIM%SPACJA wlacza sterowanie, zwiniete palce pauzuja,%C_OFF%
echo   %C_DIM%szczypniecie kciuk-wskazujacy steruje chwytakiem.%C_OFF%
exit /b 0

rem --- szukanie Pythona ----------------------------------------------
:ensure_python
rem Najpierw 3.12 - z nim sprawdzone sa wszystkie wersje w constraints.txt
rem (numpy 2.2.6 nie ma kol dla 3.14, LeRobot 0.6 wymaga 3.12).
set "LAUNCHER="
py -3.12 --version >nul 2>&1 && set "LAUNCHER=py -3.12"
if not defined LAUNCHER (
    py -3 --version >nul 2>&1 && set "LAUNCHER=py -3"
)
if not defined LAUNCHER (
    python --version >nul 2>&1 && set "LAUNCHER=python"
)
if not defined LAUNCHER (
    echo.
    echo   %C_ERR%Nie znalazlem Pythona.%C_OFF%
    echo   Zainstaluj Pythona 3.12 z https://www.python.org/downloads/
    echo   %C_DIM%W instalatorze zaznacz "Add python.exe to PATH".%C_OFF%
    exit /b 1
)
if not "%LAUNCHER%"=="py -3.12" (
    echo   %C_DIM%Uwaga: nie ma Pythona 3.12 - uzywam: %LAUNCHER%. Sprawdzony jest 3.12.%C_OFF%
)
exit /b 0

rem --- srodowisko wirtualne ------------------------------------------
:ensure_venv
if exist "%VENV_PY%" exit /b 0
echo.
echo   %C_DIM%Pierwsze uruchomienie - tworze srodowisko .venv ...%C_OFF%
%LAUNCHER% -m venv "%VENV%"
if not exist "%VENV_PY%" (
    echo   %C_ERR%Nie udalo sie utworzyc srodowiska .venv%C_OFF%
    exit /b 1
)
call :install
exit /b %ERRORLEVEL%

:install
echo.
echo   %C_DIM%Instaluje zaleznosci - to potrwa kilka minut ...%C_OFF%
"%VENV_PY%" -m pip install --upgrade pip --quiet
rem [feetech] to sam pyserial - bez niego opcja 2 (prawdziwe ramie) nie ruszy.
if "%~1"=="force" (
    "%VENV_PY%" -m pip install --force-reinstall --no-cache-dir -e ".[feetech]" -c constraints.txt
) else (
    "%VENV_PY%" -m pip install -e ".[feetech]" -c constraints.txt
)
if errorlevel 1 (
    echo   %C_ERR%Instalacja nie powiodla sie.%C_OFF%
    echo   %C_DIM%Sprobuj opcji 6 albo sprawdz polaczenie z internetem.%C_OFF%
    exit /b 1
)
echo   %C_OK%Gotowe.%C_OFF%
exit /b 0

rem --- uruchomienia ---------------------------------------------------
:run
echo.
echo   %C_DIM%Uruchamiam: lerobot-mp %*%C_OFF%
echo   %C_DIM%Zamkniecie: klawisz X w oknie podgladu. Stop awaryjny: ESC.%C_OFF%
echo.
"%VENV_PY%" -m lerobot_mp %*
if errorlevel 1 echo   %C_ERR%Aplikacja zakonczyla sie bledem.%C_OFF%
exit /b 0

:run_real
echo.
echo   %C_DIM%Porty szeregowe widoczne w systemie:%C_OFF%
rem Pusta lista to najczestszy powod "aplikacja nie widzi ramienia" - lepiej
rem powiedziec to wprost niz pokazac nic i poprosic o nazwe portu.
set "FOUND="
for /f "usebackq delims=" %%p in (`powershell -NoProfile -Command "[System.IO.Ports.SerialPort]::GetPortNames()" 2^>nul`) do (
    echo      %%p
    set "FOUND=1"
)
if not defined FOUND (
    echo      %C_ERR%brak - system nie widzi zadnego portu COM%C_OFF%
    echo.
    echo   %C_DIM%Ramie zglasza sie jako przejsciowka USB-serial. Kiedy portu nie ma:%C_OFF%
    echo   %C_DIM%  - kabel USB musi byc do danych, nie sam do ladowania,%C_OFF%
    echo   %C_DIM%  - moze brakowac sterownika przejsciowki (CH340, CP210x, FTDI),%C_OFF%
    echo   %C_DIM%  - na maszynie wirtualnej albo zdalnej trzeba przepuscic to%C_OFF%
    echo   %C_DIM%    urzadzenie do systemu goszczonego - inaczej nigdy nie dotrze.%C_OFF%
    echo.
    exit /b 0
)
echo.
set "PORT="
set /p "PORT=  Port ramienia [np. COM5]: "
if "%PORT%"=="" (
    echo   %C_ERR%Bez portu nie da sie polaczyc z ramieniem.%C_OFF%
    exit /b 0
)
echo.
echo   %C_ERR%UWAGA:%C_OFF% ramie zaraz sie poruszy. Zrob wokol niego miejsce.
echo   %C_DIM%Sterowanie rusza dopiero po nacisnieciu SPACJI, ESC to stop awaryjny,%C_OFF%
echo   %C_DIM%a X zamyka aplikacje (ramie wraca wtedy do pozycji domowej).%C_OFF%
pause
call :run --port %PORT%
exit /b 0

:run_video
echo.
set "VID="
set /p "VID=  Sciezka do pliku wideo: "
if "%VID%"=="" exit /b 0
if not exist "%VID%" (
    echo   %C_ERR%Nie ma takiego pliku: %VID%%C_OFF%
    exit /b 0
)
call :run --source "%VID%" --clutch always
exit /b 0

:diagnose
echo.
echo   %C_TITLE%Konfiguracja%C_OFF%
"%VENV_PY%" -m lerobot_mp --print-config
echo.
echo   %C_TITLE%Testy%C_OFF%
"%VENV_PY%" -m pip install -e ".[feetech,dev]" -c constraints.txt --quiet
rem Testy blizniaka pomijaja sie same, jesli panel (opcja 7) nie byl jeszcze instalowany.
"%VENV_PY%" -m pytest -q
exit /b 0

rem --- panel blizniaka ------------------------------------------------
:run_twin
"%VENV_PY%" -c "import mujoco, viser, gymnasium, torch" >nul 2>&1
if errorlevel 1 (
    call :install_twin || exit /b 0
)
if not exist "%~dp0workspace\twin.json" (
    echo.
    echo   %C_DIM%Nie ma jeszcze stanowiska ^(workspace\twin.json^). Przyklad w symulacji:%C_OFF%
    echo   %C_DIM%ramie sim i dwie skalibrowane kamery symulowane ^(examples\twin.sim.json^).%C_OFF%
    set "DEMO="
    set /p "DEMO=  Wczytac przyklad? [T/n]: "
    if /i not "!DEMO!"=="n" "%VENV_PY%" -m lerobot_mp.twin.cli demo
)
echo.
echo   %C_OK%Panel: http://localhost:8080%C_OFF%  %C_DIM%- otworz w przegladarce; karta musi byc widoczna.%C_OFF%
echo   %C_DIM%Zamkniecie: Ctrl+C w tym oknie.%C_OFF%
echo.
"%VENV_PY%" -m lerobot_mp.twin.cli ui
exit /b 0

:install_twin
echo.
echo   %C_DIM%Pierwsze uruchomienie panelu - instaluje MuJoCo, viser i torcha ...%C_OFF%
rem torch NAJPIERW z indeksu PyTorcha: z PyPI na Windows jest bez CUDA.
rem Z karta NVIDIA - kolo z CUDA (ok. 2,6 GB), bez niej - CPU (panel dziala i tak).
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
) else (
    set "TORCH_INDEX=https://download.pytorch.org/whl/cu126"
)
"%VENV_PY%" -m pip install torch==2.11.0 --index-url !TORCH_INDEX!
if errorlevel 1 goto :install_twin_fail
"%VENV_PY%" -m pip install -e ".[twin,feetech]" -c constraints.txt
if errorlevel 1 goto :install_twin_fail
echo   %C_OK%Gotowe.%C_OFF% %C_DIM%Trening na GPU: pip install -e ".[train]" -c constraints.txt%C_OFF%
exit /b 0
:install_twin_fail
echo   %C_ERR%Instalacja panelu nie powiodla sie.%C_OFF%
echo   %C_DIM%Sprawdz polaczenie z internetem i wolne miejsce na dysku.%C_OFF%
exit /b 1

rem ===================================================================
:fail
echo.
echo   %C_ERR%Nie udalo sie przygotowac srodowiska.%C_OFF%
:end
echo.
pause
endlocal
