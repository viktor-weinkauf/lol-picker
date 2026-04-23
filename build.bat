@echo off
REM Build LoLPicker.exe with PyInstaller.
REM Bundles templates/, static/, and data/ so the app works before any GitHub sync.

python -m PyInstaller ^
  --name LoLPicker ^
  --onefile ^
  --noconsole ^
  --noconfirm ^
  --icon "assets/icon.ico" ^
  --add-data "templates;templates" ^
  --add-data "static;static" ^
  --add-data "data;data" ^
  --add-data "assets;assets" ^
  --collect-all flask ^
  --collect-all pystray ^
  --collect-all PIL ^
  app.py

echo.
echo Done. Executable is at dist\LoLPicker.exe
