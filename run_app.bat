@echo off
title DIPAS - Inferencia y Diseno Inverso Aerodinamico
echo =====================================================================
echo  Iniciando DIPAS // Diseno Inverso para Perfiles con Autoencoder
echo =====================================================================
echo.
cd /d "%~dp0"
python -m streamlit run code/app.py
pause
