@echo off
REM Activate the virtual environment for FaceLocking project
echo Activating FaceLocking virtual environment...
call .venv\Scripts\activate.bat
echo.
echo Virtual environment activated!
echo.
echo Available commands:
echo   - python -m src.enroll       : Enroll new faces
echo   - python -m src.recognize    : Run face recognition
echo   - python -m src.evaluate     : Evaluate model performance
echo   - python -m src.rebuild_db   : Rebuild database from crops
echo.
