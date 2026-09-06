@echo off
setlocal

set /p "commit_msg=Commit: "

if "%commit_msg%"=="" (
    echo ERROR: El mensaje no puede estar vacio.
    pause
    exit /b 1
)

git add .
if errorlevel 1 goto error_add

git commit -m "%commit_msg%"
if errorlevel 1 goto error_commit

git push
if errorlevel 1 goto error_push

exit /b 0


:error_add
echo.
echo ERROR: Fallo en "git add ."
pause
exit /b 1

:error_commit
echo.
echo ERROR: Fallo en "git commit".
pause
exit /b 1

:error_push
echo.
echo ERROR: Fallo en "git push".
pause
exit /b 1