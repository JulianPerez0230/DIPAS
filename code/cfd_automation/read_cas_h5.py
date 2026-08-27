import os
import h5py

def inspect_case_file():
    project_dir = r"C:\Users\JULIAN\JunoWorkspace\projects\DIPAS\archive\Simulacion_perfil_prueba_files"
    cas_file = None
    
    # Buscamos el archivo .cas.h5 en el proyecto
    for root, dirs, files in os.walk(project_dir):
        for f in files:
            if f.endswith(".cas.h5"):
                cas_file = os.path.join(root, f)
                break
        if cas_file:
            break
            
    if not cas_file:
        print("Error: No se encontro ningun archivo .cas.h5 en el proyecto.")
        return
        
    print(f">> Inspeccionando archivo de caso: {cas_file}")
    
    try:
        with h5py.File(cas_file, "r") as f:
            # En Fluent H5, las zonas y sus nombres estan en la ruta /settings/boundary-conditions
            # o en la definicion de las zonas
            if "settings" in f:
                settings = f["settings"]
                print("\nEstructura de settings encontrada:")
                for k in settings.keys():
                    print(f" - {k}")
                    
            # Tambien listamos los datasets principales para ver la organizacion
            print("\nGrupos principales en el archivo H5:")
            for k in f.keys():
                print(f" - {k}")
                
            # Busqueda recursiva de nombres de zonas
            print("\nBuscando nombres de zonas de frontera...")
            def find_zones(name, obj):
                if isinstance(obj, h5py.Group):
                    # En algunos formatos h5 de Fluent, los nombres se guardan en atributos o subgrupos
                    if "zone" in name.lower() or "boundary" in name.lower():
                        print(f" Grupo: {name}")
                        for attr_name, attr_value in obj.attrs.items():
                            print(f"   Atributo {attr_name}: {attr_value}")
            f.visititems(find_zones)
            
    except Exception as e:
        print(f"Ocurrio un error al leer el archivo H5: {e}")

if __name__ == "__main__":
    inspect_case_file()
