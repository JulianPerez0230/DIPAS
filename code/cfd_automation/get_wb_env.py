import os

def dump_env():
    # Buscamos la ruta absoluta para guardar el archivo txt en la carpeta data
    output_path = r"c:\Users\JULIAN\JunoWorkspace\projects\DIPAS\data\wb_env.txt"
    with open(output_path, "w") as f:
        for k, v in sorted(os.environ.items()):
            f.write(f"{k} = {v}\n")
    print(f"Environment variables dumped successfully to {output_path}")

if __name__ == "__main__":
    dump_env()
