# -*- coding: utf-8 -*-
# Diagnostic script using relative transcript to find TUI prompts without crashes
import os

def test_tui():
    project_path = "C:/Users/JULIAN/JunoWorkspace/projects/DIPAS/archive/Simulacion_perfil_prueba.wbpj"
    log_path = "C:/Users/JULIAN/JunoWorkspace/projects/DIPAS/data/tui_test.log"
    
    with open(log_path, "w") as log:
        log.write(">> Iniciando grabacion de ayuda de prompts relativo...\n")
        try:
            Open(FilePath=project_path)
            system = GetSystem(Name="FFF")
            setup = system.GetContainer(ComponentName="Setup")
            
            setup.Edit(Interactive=False)
            
            # Iniciamos la transcripcion relativa pura
            log.write(">> Iniciando transcripcion...\n")
            setup.SendCommand(Command='/file/start-transcript "tui_help.trn"')
            
            # Enviamos el comando de ayuda
            log.write(">> Enviando comando /define/boundary-conditions/velocity-inlet inlet ?...\n")
            setup.SendCommand(Command="/define/boundary-conditions/velocity-inlet inlet ?")
            
            # Detenemos transcripcion
            setup.SendCommand(Command="/file/stop-transcript")
            
            log.write(">> [OK] Consulta finalizada.\n")
            setup.Exit()
        except Exception as e:
            log.write(">> Error: " + str(e) + "\n")

test_tui()
