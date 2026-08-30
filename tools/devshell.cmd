@echo off
rem Activate the MSVC host compiler and user-installed CMake/Ninja for this shell.
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo ERROR: Visual Studio Build Tools were not found.
  exit /b 1
)
for /f "usebackq delims=" %%I in (`"%VSWHERE%" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS_INSTALL=%%I"
if not defined VS_INSTALL (
  echo ERROR: Install the MSVC C++ build tools workload first.
  exit /b 1
)
call "%VS_INSTALL%\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
set "PATH=%APPDATA%\Python\Python312\Scripts;%PATH%"
where cmake >nul 2>nul || (echo ERROR: CMake not found. Run: python -m pip install --user cmake ninja & exit /b 1)
where ninja >nul 2>nul || (echo ERROR: Ninja not found. Run: python -m pip install --user cmake ninja & exit /b 1)
echo CUDA operator build environment ready.

