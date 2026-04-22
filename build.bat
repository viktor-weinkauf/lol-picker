@echo off
REM Build LoLPicker.exe with PyInstaller.
REM Bundles templates/, static/, and data/ so the app works before any GitHub sync.

pyinstaller ^
  --name LoLPicker ^
  --onefile ^
  --add-data "templates;templates" ^
  --add-data "static;static" ^
  --add-data "data;data" ^
  --collect-all flask ^
  app.py

echo.
echo Done. Executable is at dist\LoLPicker.exe
