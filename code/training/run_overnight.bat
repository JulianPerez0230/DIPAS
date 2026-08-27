@echo off
title DIPAS - CFD Dataset Builder (Overnight Runner)
echo ========================================================
echo   DIPAS - CFD Dataset Overnight Simulation
echo ========================================================
echo.
echo Evitando suspension de Windows...
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /change monitor-timeout-ac 2

echo.
echo Iniciando dataset_cfd_builder.py en prioridad baja (Baja temperatura/ruido)...
start /belownormal /wait python "c:\Users\JULIAN\JunoWorkspace\projects\DIPAS\code\dataset_cfd_builder.py"

echo.
echo ========================================================
echo   Simulaciones finalizadas exitosamente!
echo ========================================================
pause
