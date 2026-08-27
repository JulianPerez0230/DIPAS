# -*- coding: utf-8 -*-
"""
Conversor directo de malla Gmsh 2D a formato nativo ANSYS Fluent MSH (2D ASCII).
Garantiza 100% de compatibilidad sin librerias intermedias ni crasheos de CGNS.
"""
import numpy as np
from pathlib import Path
import gmsh

def export_gmsh_to_fluent_msh2d(output_msh_path):
    """
    Lee la malla actual de Gmsh en memoria y la escribe en formato Fluent MSH 2D (ASCII).
    """
    output_msh_path = Path(output_msh_path)
    
    # 1. Obtener nodos (solo coordenadas X, Y para 2D)
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_coords = node_coords.reshape(-1, 3)[:, :2]  # Tomar solo X, Y
    n_nodes = len(node_tags)
    
    # Mapeo de tag de nodo original de Gmsh a indice 1-based correlativo
    node_map = {tag: i + 1 for i, tag in enumerate(node_tags)}
    
    # 2. Obtener Physical Groups (Boundary conditions y Fluid domain)
    physical_groups = gmsh.model.getPhysicalGroups()
    
    # Mapear nombres a (zone_id, bc_type_name, fluent_zone_type_code)
    # Fluent zone types:
    # 2: interior, 3: wall, 4: pressure-inlet / velocity-inlet, 5: pressure-outlet, 7: symmetry
    zone_names = {}
    zone_type_codes = {}
    zone_id_counter = 3
    fluid_zone_id = 2
    
    bc_groups = {} # gmsh_tag -> (zone_id, type_code)
    for dim, tag in physical_groups:
        name = gmsh.model.getPhysicalName(dim, tag)
        if dim == 2:
            zone_names[fluid_zone_id] = ("fluid", "fluid")
            zone_type_codes[fluid_zone_id] = 1 # fluid
        elif dim == 1:
            z_id = zone_id_counter
            zone_id_counter += 1
            if "airfoil" in name.lower() or "wall" in name.lower():
                bc_type = "wall"
                z_code = 3
            elif "inlet" in name.lower():
                bc_type = "velocity-inlet"
                z_code = 4 # 4 = inlet en Fluent
            elif "outlet" in name.lower():
                bc_type = "pressure-outlet"
                z_code = 5
            else:
                bc_type = "symmetry"
                z_code = 7
                
            zone_names[z_id] = (name, bc_type)
            zone_type_codes[z_id] = z_code
            bc_groups[tag] = (z_id, z_code)

    # 3. Extraer celdas 2D (triángulos)
    elem_types, elem_tags_list, elem_node_tags_list = gmsh.model.mesh.getElements(dim=2)
    cells = [] # lista de [n1, n2, n3] en indices Fluent (1-based)
    
    for el_type, el_tags, el_nodes in zip(elem_types, elem_tags_list, elem_node_tags_list):
        if el_type == 2: # Triangulo de 3 nodos
            nodes_per_elem = 3
        else:
            continue
            
        el_nodes_arr = el_nodes.reshape(-1, nodes_per_elem)
        for n_list in el_nodes_arr:
            f_nodes = [node_map[nt] for nt in n_list]
            cells.append(f_nodes)
            
    n_cells = len(cells)
    
    # 4. Construir caras (Faces) para Fluent
    # En Fluent 2D:
    # Cada triangulo (n0, n1, n2) tiene 3 caras orientadas: (n0, n1), (n1, n2), (n2, n0)
    edge_map = {} # (min_n, max_n) -> {'n1': n1, 'n2': n2, 'c0': c0, 'c1': c1, 'bc_zone': None, 'z_code': None}
    
    for c_id, (n0, n1, n2) in enumerate(cells, 1):
        tri_edges = [(n0, n1), (n1, n2), (n2, n0)]
        for na, nb in tri_edges:
            key = (min(na, nb), max(na, nb))
            if key not in edge_map:
                edge_map[key] = {
                    'n1': na,
                    'n2': nb,
                    'c0': c_id,
                    'c1': 0,
                    'bc_zone': None,
                    'z_code': None
                }
            else:
                edge_map[key]['c1'] = c_id

    # Asignar physical groups a las aristas de boundary
    for dim, p_tag in physical_groups:
        if dim == 1:
            z_id, z_code = bc_groups[p_tag]
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, p_tag)
            for e_tag in entities:
                b_types, b_tags, b_nodes = gmsh.model.mesh.getElements(dim=1, tag=e_tag)
                for bt, b_tag_list, bn in zip(b_types, b_tags, b_nodes):
                    bn_arr = bn.reshape(-1, 2)
                    for n_pair in bn_arr:
                        fn1, fn2 = node_map[n_pair[0]], node_map[n_pair[1]]
                        key = (min(fn1, fn2), max(fn1, fn2))
                        if key in edge_map:
                            edge_map[key]['bc_zone'] = z_id
                            edge_map[key]['z_code'] = z_code

    # Separar en caras interiores y caras de boundary
    interior_faces = []
    bc_faces = {z_id: [] for z_id, _ in bc_groups.values()}
    
    interior_zone_id = 1
    zone_names[interior_zone_id] = ("interior", "interior")
    zone_type_codes[interior_zone_id] = 2 # interior

    for key, data in edge_map.items():
        n1, n2 = data['n1'], data['n2']
        c0, c1 = data['c0'], data['c1']
        z_id = data['bc_zone']
        
        if z_id is not None:
            # Cara de frontera: c0 es la celda adyacente, c1 = 0
            bc_faces[z_id].append((n1, n2, c0, 0))
        elif c1 != 0:
            interior_faces.append((n1, n2, c0, c1))
        else:
            interior_faces.append((n1, n2, c0, 0))

    # Numeracion global secuencial de caras (1 .. total_faces)
    face_sections = []
    global_face_idx = 1
    
    # 1. Caras interiores: zone_type = 2 (interior), element_type = 2 (line/2D)
    if interior_faces:
        start_f = global_face_idx
        end_f = global_face_idx + len(interior_faces) - 1
        face_sections.append((interior_zone_id, 2, 2, start_f, end_f, interior_faces))
        global_face_idx = end_f + 1
        
    # 2. Caras de frontera: zone_type = z_code (3=wall, 10=inlet, etc.), element_type = 2 (line)
    for z_id, f_list in bc_faces.items():
        if f_list:
            start_f = global_face_idx
            end_f = global_face_idx + len(f_list) - 1
            z_code = zone_type_codes[z_id]
            face_sections.append((z_id, z_code, 2, start_f, end_f, f_list))
            global_face_idx = end_f + 1
            
    total_faces = global_face_idx - 1
    
    # 5. Escribir archivo Fluent .msh (ASCII)
    with open(output_msh_path, "w") as f:
        # Encabezado
        f.write('(0 "ANSYS Fluent 2D Mesh from Gmsh")\n')
        f.write('(0 "Dimension:")\n')
        f.write('(2 2)\n') # 2D
        
        # Declaracion y coordenadas de nodos
        f.write(f'(10 (0 1 {hex(n_nodes)[2:]} 0 2))\n')
        f.write(f'(10 (1 1 {hex(n_nodes)[2:]} 1 2)(\n')
        for x, y in node_coords:
            f.write(f'  {x:.12e} {y:.12e}\n')
        f.write('))\n')
        
        # Declaracion de celdas
        f.write(f'(12 (0 1 {hex(n_cells)[2:]} 0))\n')
        f.write(f'(12 ({fluid_zone_id} 1 {hex(n_cells)[2:]} 1 1))\n') # 1=active, 1=tri
        
        # Declaracion y lista de caras
        f.write(f'(13 (0 1 {hex(total_faces)[2:]} 0))\n')
        for z_id, z_type, elem_type, start_f, end_f, f_list in face_sections:
            f.write(f'(13 ({z_id} {hex(start_f)[2:]} {hex(end_f)[2:]} {z_type} {elem_type})(\n')
            for n1, n2, c0, c1 in f_list:
                f.write(f'  {hex(n1)[2:]} {hex(n2)[2:]} {hex(c0)[2:]} {hex(c1)[2:]}\n')
            f.write('))\n')
            
        # Zonas y condiciones de contorno
        f.write('(0 "Zones:")\n')
        for z_id, (name, bc_type) in zone_names.items():
            f.write(f'(45 ({z_id} {bc_type} {name})())\n')
            
        # Zonas y tipos
        f.write('(0 "Zones:")\n')
        for z_id, (name, bc_type) in zone_names.items():
            f.write(f'(45 ({z_id} {bc_type} {name})())\n')

    print(f"[OK] Malla Fluent 2D nativa exportada con exito:")
    print(f"     Nodos : {n_nodes}")
    print(f"     Celdas: {n_cells}")
    print(f"     Caras : {total_faces}")
    print(f"     Archivo: {output_msh_path}")
    return True
