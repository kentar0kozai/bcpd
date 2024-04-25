import glob
import os
import re
import struct

import numpy as np
import pymeshlab

directory_path = "data/Mug"

ply_files = glob.glob(f"{directory_path}/*.ply")
if not os.path.exists(f'{directory_path}/remesh'):
    os.makedirs(f'{directory_path}/remesh')


def extract_number(filename):
    match = re.search(r"\d+", filename)
    return int(match.group()) if match else 0

def save_ply(filename, verts, normals, colors, faces):
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    with open(filename, 'wb') as f:
        # ヘッダー情報を書き込む
        f.write(b'ply\n')
        f.write(b'format binary_little_endian 1.0\n')
        f.write(b'comment VCGLIB generated\n')
        f.write(f'element vertex {num_verts}\n'.encode('ascii'))
        f.write(b'property float x\n')
        f.write(b'property float y\n')
        f.write(b'property float z\n')
        f.write(b'property float nx\n')
        f.write(b'property float ny\n')
        f.write(b'property float nz\n')
        f.write(b'property uchar red\n')
        f.write(b'property uchar green\n')
        f.write(b'property uchar blue\n')
        f.write(b'property uchar alpha\n')
        f.write(f'element face {num_faces}\n'.encode('ascii'))
        f.write(b'property list uchar int vertex_indices\n')
        f.write(b'end_header\n')

        for i in range(num_verts):
            f.write(struct.pack('<fff', verts[i, 0], verts[i, 1], verts[i, 2]))
            f.write(struct.pack('<fff', normals[i, 0], normals[i, 1], normals[i, 2]))
            f.write(struct.pack('<BBBB', colors[i, 0], colors[i, 1], colors[i, 2], colors[i, 3]))

        for i in range(num_faces):
            f.write(struct.pack('<Biii', 3, faces[i, 0], faces[i, 1], faces[i, 2]))

result_ms = pymeshlab.MeshSet()
ms = pymeshlab.MeshSet()
ply_files_sorted = sorted(ply_files, key=extract_number)
if len(ply_files_sorted) == 0:
    raise ValueError(f'No model found : {directory_path}')
for i, file in enumerate(ply_files_sorted):
    print(file)

    ms.load_new_mesh(file)
    ms.set_current_mesh(0)
    ms.compute_matrix_from_scaling_or_normalization(unitflag=True)
    ms.compute_matrix_from_translation(traslmethod=2)
    ms.add_mesh(ms.current_mesh(), set_as_current=False)
    ms.compute_scalar_ambient_occlusion_gpu(conedir=np.array([0.0, 0.0, 0.0]), usegpu=True)
    ms.compute_selection_by_color_per_face(colorspace=1, percentrh=0.0, percentgs=0.0, percentbv=0.0)
    ms.meshing_remove_selected_vertices_and_faces()
    ms.generate_sampling_poisson_disk(samplenum=100000)
    ms.generate_surface_reconstruction_screened_poisson()
    for j in range(5):
        ms.apply_coord_hc_laplacian_smoothing()
    ms.transfer_attributes_per_vertex(sourcemesh=1, targetmesh=3)

    cur_msh = ms.current_mesh()
    verts = cur_msh.vertex_matrix()
    normals = cur_msh.vertex_normal_matrix()
    faces = cur_msh.face_matrix()
    colors = (cur_msh.vertex_color_matrix() * 255).astype(np.int32)
    save_ply(f"{directory_path}/remesh/Mug{i+1}_remesh.ply", verts.astype(np.float32), normals.astype(np.float32), colors, faces)
    result_ms.add_mesh(ms.current_mesh(), set_as_current=False)
    ms.clear()

result_ms.show_polyscope()
result_ms.clear()
