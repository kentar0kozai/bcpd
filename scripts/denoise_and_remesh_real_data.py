import glob
import os
import re

import numpy as np
import pymeshlab

directory_path = "../data/Mug"

ply_files = glob.glob(f"{directory_path}/*.ply")
if not os.path.exists(f'{directory_path}/remesh'):
    os.makedirs(f'{directory_path}/remesh')


def extract_number(filename):
    match = re.search(r"\d+", filename)
    return int(match.group()) if match else 0


result_ms = pymeshlab.MeshSet()
ms = pymeshlab.MeshSet()
ply_files_sorted = sorted(ply_files, key=extract_number)
for i, file in enumerate(ply_files_sorted):
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
    ms.save_current_mesh(f"{directory_path}/remesh/Mug{i+1}_remesh.ply")
    result_ms.add_mesh(ms.current_mesh(), set_as_current=False)
    ms.clear()

result_ms.show_polyscope()
result_ms.clear()
