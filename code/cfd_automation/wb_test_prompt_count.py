# -*- coding: utf-8 -*-
# Diagnostic script to capture the exact TUI prompts of velocity-inlet
import os

def test_tui():
    project_path = "C:/Users/JULIAN/JunoWorkspace/projects/DIPAS/archive/Simulacion_perfil_prueba.wbpj"
    log_path = "C:/Users/JULIAN/JunoWorkspace/projects/DIPAS/data/tui_test.log"
    
    with open(log_path, "w") as log:
        log.write(">> Iniciando grabacion de secuencia de prompts...\n")
        try:
            Open(FilePath=project_path)
            system = GetSystem(Name="FFF")
            setup = system.GetContainer(ComponentName="Setup")
            
            setup.Edit(Interactive=False)
            
            # Iniciamos transcripcion relativa
            setup.SendCommand(Command='/file/start-transcript "tui_prompts.trn"')
            
            # Enviamos 15 newlines para responder todo por defecto y salir
            log.write(">> Enviando comando de inlet con 15 newlines...\n")
            cmd = '(ti-menu-load-string "/define/boundary-conditions/velocity-inlet inlet\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n\\n")'
            setup.SendCommand(Command=cmd)
            
            # Detenemos transcripcion
            setup.SendCommand(Command="/file/stop-transcript")
            
            log.write(">> [OK] Grabacion finalizada.\n")
            setup.Exit()
        except Exception as e:
            log.write(">> Error: " + str(e) + "\n")

test_tui()
