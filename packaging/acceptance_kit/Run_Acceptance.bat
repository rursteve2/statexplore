@echo off
setlocal EnableDelayedExpansion
rem StatExplore 1.3.6 consolidated acceptance run (tests 55-61 + owner R diagnostic). No installation, no network,
rem no upload. It works on copies of the fixtures in a temporary folder and writes one evidence ZIP into .\results.
rem Usage:  Run_Acceptance.bat                       (installed StatExplore)
rem         Run_Acceptance.bat "D:\path\StatExplore.exe"   (another installed / portable copy)
rem         Run_Acceptance.bat "" --quick            (shorter bootstrap)
set "EXE=%~1"
if "%EXE%"=="" set "EXE=%ProgramFiles%\StatExplore\StatExplore.exe"
if not exist "%EXE%" set "EXE=%LOCALAPPDATA%\Programs\StatExplore\StatExplore.exe"
if not exist "%EXE%" (
  echo StatExplore.exe was not found. Give its path:  Run_Acceptance.bat "C:\...\StatExplore.exe"
  pause
  exit /b 2
)
set "OUT=%~dp0results"
if not exist "%OUT%" mkdir "%OUT%"
echo Using "%EXE%"
echo Fixtures: "%~dp0fixtures"
echo Results:  "%OUT%"
echo.
echo Running about 10 minutes (quick: about 5). Progress is written to results\acceptance_live_*.log
echo (open it in Notepad and press F5 to refresh). Do not close this window.
echo.
start "" /wait "%EXE%" --acceptance --fixtures "%~dp0fixtures" --out "%OUT%" %2 %3
set RC=%ERRORLEVEL%
echo.
echo Finished with exit code %RC% (0 = every check passed).
for /f "delims=" %%f in ('dir /b /o:d "%OUT%\Acceptance_Summary_*.md" 2^>nul') do set "LAST=%%f"
if defined LAST type "%OUT%\!LAST!"
echo.
echo Upload the newest StatExplore_v1.3.6_Acceptance_Results_*.zip from "%OUT%".
pause
exit /b %RC%
