@echo off

REM Get the current directory of the batch file
SET DIR=%~dp0

REM Change to the directory where the .bat file is located
cd /d "%DIR%"

REM Upgrade pip to the latest version
E:\ComfyUI\ComfyUI_windows_portable\python_embeded\Scripts\python.exe -m pip install --upgrade pip

REM Install packages from requirements.txt in the current folder
E:\ComfyUI\ComfyUI_windows_portable\python_embeded\Scripts\pip.exe install -r requirements.txt

REM Let the user know the process is done
echo All required packages have been installed/updated from requirements.txt.
pause
