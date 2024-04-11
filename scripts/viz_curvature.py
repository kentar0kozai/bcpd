import numpy as np
import pyvista
import open3d as o3d
import time
import copy
    
from sklearn.decomposition import PCA

class CurvatureInfo:
    def __init__(self):
        self.PD1 = None
        self.PD2 = None
        self.PV1 = None
        self.PV2 = None
        self.PV3 = None # normal scale
        self.PD3 = None # normal direction
        self.Curv = None

def load_ply_mesh(filename):
    with open(filename, 'r') as file:
        lines = file.read().splitlines()

    vertex_count = 0
    face_count = 0
    header_end_index = 0

    # Read PLY header
    for i, line in enumerate(lines):
        if line.startswith('element vertex'):
            vertex_count = int(line.split()[-1])
        elif line.startswith('element face'):
            face_count = int(line.split()[-1])
        elif line.startswith('end_header'):
            header_end_index = i
            break

    # Read vertex data
    vertex_data = np.genfromtxt(lines[header_end_index + 1:header_end_index + 1 + vertex_count], dtype=float)
    verts = vertex_data[:, :3]
    verts_mean = np.mean(verts, axis=0)
    verts -= verts_mean
    
    curv_info = CurvatureInfo()
    curv_info.PD1 = vertex_data[:, 3:6]
    curv_info.PD2 = vertex_data[:, 6:9]
    curv_info.PV1 = vertex_data[:, 9]
    curv_info.PV2 = vertex_data[:, 10]
    curv_info.Curv = vertex_data[:, 11]

    # Read face data
    face_data = np.genfromtxt(lines[header_end_index + 1 + vertex_count:header_end_index + 1 + vertex_count + face_count], dtype=int)[:, 1:]
    faces = face_data

    return verts, faces, curv_info

def compute_average_edge_length(vertices, faces):
    """
    vertices: numpy array of shape (num_vertices, 3)
    faces: numpy array of shape (num_faces, 3)
    """
    # get coordinate of triangles
    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]
    
    # compute edge length of triangles
    edge_lengths = np.concatenate([
        np.linalg.norm(v1 - v2, axis=1),
        np.linalg.norm(v2 - v3, axis=1),
        np.linalg.norm(v3 - v1, axis=1)
    ])
    
    # compute average edge length
    average_edge_length = np.mean(edge_lengths)
    
    return average_edge_length


def get_pyvista_coord_frame(size=10):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size)
    verts = np.asarray(frame.vertices)
    triangles = np.asarray(frame.triangles)
    faces = np.concatenate((np.full((triangles.shape[0], 1), 3), triangles), axis=1)
    frame_pv = pyvista.PolyData(verts, faces)
    frame_pv.point_data["color"] = np.asarray(frame.vertex_colors)
    return frame_pv

def visualize_principal_curvature(verts, faces, pd1, pv1, pd2, pv2, pd3, pv3, indices=None, scale=None):
    ''' 
    sample_indices = np.random.choice(mesh.n_points, int(mesh.n_points * sample_rate), replace=False)
    visualize_principal_curvature(..., indices=sample_indices)
    '''
    triangles = np.concatenate((np.full((faces.shape[0], 1), 3), faces), axis=1)
    mesh = pyvista.PolyData(verts, triangles)
    pcd = pyvista.PolyData(verts)
    pcd.point_data["pd1"] = pd1
    pcd.point_data["pv1"] = pv1
    pcd.point_data["pd2"] = pd2
    pcd.point_data["pv2"] = pv2
    pcd.point_data["normals"] = pd3
    pcd.point_data["normals_scale"] = pv3

    if indices is None:
        indices = np.arange(mesh.n_points)

    sampled_pcd = pcd.extract_points(indices)

    # Generate vector field of principal curvature
    if scale is None:
        scale = 10
        vf1 = sampled_pcd.glyph(
            orient="pd1", factor=scale, geom=pyvista.Arrow(),
        )
        vf2 = sampled_pcd.glyph(
            orient="pd2", factor=scale, geom=pyvista.Arrow(),
        )
        vf3 = sampled_pcd.glyph(
            orient="normals", factor=scale, geom=pyvista.Arrow(),
        )

    else:
        vf1 = sampled_pcd.glyph(
            orient="pd1", scale="pv1", factor=scale, geom=pyvista.Arrow(),
        )
        vf2 = sampled_pcd.glyph(
            orient="pd2", scale="pv2", factor=scale, geom=pyvista.Arrow(),
        )
        vf3 = sampled_pcd.glyph(
            orient="normals", scale="normals_scale", factor=scale, geom=pyvista.Arrow(),
        )

    return mesh, vf1, vf2, vf3


def show_two_mesh_and_vf(mesh_x, x_vf1, x_vf2, x_vf3, x_vf1_comp,
                         mesh_y, y_vf1, y_vf2, y_vf3, y_vf1_comp,
                         sync_view:bool=True):
    # Create a subplot with 2 rows and 1 column
    plotter = pyvista.Plotter(shape=(1, 2))

    # Add the first mesh and vector fields to the first subplot
    plotter.subplot(0, 0)
    plotter.add_mesh(mesh_x, color='gray', opacity=0.3)
    plotter.add_mesh(x_vf1, line_width=2, color='r') # cmap='jet')
    plotter.add_mesh(x_vf2, line_width=2, color='g')# cmap='jet')
    plotter.add_mesh(x_vf3, line_width=2, color='b')
    plotter.add_title("Mesh X")
    plotter = add_principal_comp(plotter, x_vf1_comp)

    # Add the second mesh and vector fields to the second subplot
    plotter.subplot(0, 1)
    plotter.add_mesh(mesh_y, color='gray', opacity=0.3)
    plotter.add_mesh(y_vf1, line_width=2, color='r') # cmap='jet')
    plotter.add_mesh(y_vf2, line_width=2, color='g')# cmap='jet')
    plotter.add_mesh(y_vf3, line_width=2, color='b')
    plotter.add_title("Mesh Y")
    plotter = add_principal_comp(plotter, y_vf1_comp)

    # Link the camera between subplots
    if sync_view:
        plotter.link_views()

    # Show the plot
    plotter.show()

def principal_analysis(data, weights):
    '''
    Principal analization of Weighted data
    '''
    weights_norm = (weights - weights.min()) / (weights.max() - weights.min())
    weights_column = weights_norm.reshape(-1, 1)
    weighted_data = data * weights_column
    pca = PCA(n_components=3) 
    pca.fit(weighted_data)
    principal_components = pca.components_
    projected_data = pca.transform(weighted_data)
    return projected_data, principal_components

def add_principal_comp(plotter, principal_comp):
    '''
    Add principal components(1,2,3) to plotter
    '''
    origin = np.array([[0, 0, 0]])
    for i, pc in enumerate(principal_comp):
        if i == 0:
            color = 'red'
        elif i == 1:
            color = 'green'
        else:
            color = 'blue'
        
        arrow = pyvista.Arrow(origin, pc, shaft_radius=0.02, tip_radius=0.04, tip_length=0.1, scale=10)
        plotter.add_mesh(arrow, color=color)
    return plotter

if __name__ == "__main__":
    # Store principal curv direction 1, 2 and principal curv value 1,2
    # Use debug/curv/*.ply in main/cpp
    x_path = "win/VisualStudio/BCPD-Win/debug/curv/X_mesh.ply"
    y_path = "win/VisualStudio/BCPD-Win/debug/curv/Y_mesh.ply"
    X_verts, X_faces, X_Curv = load_ply_mesh(x_path)
    Y_verts, Y_faces, Y_Curv = load_ply_mesh(y_path)

    # compute average edge length but unused yet
    avg_x_edge_len = compute_average_edge_length(X_verts, X_faces)
    avg_y_edge_len = compute_average_edge_length(Y_verts, Y_faces)
    avg_x_edge_len = 1
    avg_y_edge_len = 1

    X_Curv.PD3 = np.cross(X_Curv.PD1, X_Curv.PD2)
    X_Curv.PV3 = (X_Curv.PV1 + X_Curv.PV2) / 2
    Y_Curv.PD3 = np.cross(Y_Curv.PD1, Y_Curv.PD2)
    Y_Curv.PV3 = (Y_Curv.PV1 + Y_Curv.PV2) / 2

    X_PV1_norm = (X_Curv.PV1 - X_Curv.PV1.min()) / (X_Curv.PV1.max() - X_Curv.PV1.min())
    X_PV2_norm = (X_Curv.PV2 - X_Curv.PV2.min()) / (X_Curv.PV2.max() - X_Curv.PV2.min())
    X_PV3_norm = (X_Curv.PV3 - X_Curv.PV3.min()) / (X_Curv.PV3.max() - X_Curv.PV3.min())
    Y_PV1_norm = (Y_Curv.PV1 - Y_Curv.PV1.min()) / (Y_Curv.PV1.max() - Y_Curv.PV1.min())
    Y_PV2_norm = (Y_Curv.PV2 - Y_Curv.PV2.min()) / (Y_Curv.PV2.max() - Y_Curv.PV2.min())
    Y_PV3_norm = (Y_Curv.PV3 - Y_Curv.PV3.min()) / (Y_Curv.PV3.max() - Y_Curv.PV3.min())

    projected_X_d1, X_pd1_principal_comp = principal_analysis(X_Curv.PD1, X_PV1_norm)
    projected_Y_d1, Y_pd1_principal_comp = principal_analysis(Y_Curv.PD1, Y_PV1_norm)

    projected_X_d2, X_pd2_principal_comp = principal_analysis(X_Curv.PD2, X_PV2_norm)
    projected_Y_d2, Y_pd2_principal_comp = principal_analysis(Y_Curv.PD2, Y_PV2_norm)

    projected_X_d3, X_pd3_principal_comp = principal_analysis(X_Curv.PD3, X_PV3_norm)
    projected_Y_d3, Y_pd3_principal_comp = principal_analysis(Y_Curv.PD3, Y_PV3_norm)

    x_mesh, x_vf1, x_vf2, x_vf3 = visualize_principal_curvature(X_verts, 
                                                        X_faces, 
                                                        X_Curv.PD1, 
                                                        X_Curv.PV1 * avg_x_edge_len, 
                                                        X_Curv.PD2,
                                                        X_Curv.PV2 * avg_x_edge_len,
                                                        X_Curv.PD3,
                                                        X_Curv.PV3 * avg_x_edge_len,
                                                        scale=5)
    y_mesh, y_vf1, y_vf2, y_vf3 = visualize_principal_curvature(Y_verts, 
                                                        Y_faces, 
                                                        Y_Curv.PD1, 
                                                        Y_Curv.PV1 * avg_y_edge_len, 
                                                        Y_Curv.PD2,
                                                        Y_Curv.PV2 * avg_y_edge_len,
                                                        Y_Curv.PD3,
                                                        Y_Curv.PV3 * avg_y_edge_len,
                                                        scale=5)
    
    show_two_mesh_and_vf(x_mesh, x_vf1, x_vf2, x_vf3, X_pd1_principal_comp,
                         y_mesh, y_vf1, y_vf2, y_vf3, Y_pd1_principal_comp,
                         sync_view=False)
    
    # show_two_mesh_and_vf(x_mesh, x_vf1, x_vf2, x_vf3, X_pd2_principal_comp,
    #                      y_mesh, y_vf1, y_vf2, y_vf3, Y_pd2_principal_comp,
    #                      sync_view=False)

    # show_two_mesh_and_vf(x_mesh, x_vf1, x_vf2, x_vf3, X_pd3_principal_comp,
    #                      y_mesh, y_vf1, y_vf2, y_vf3, Y_pd3_principal_comp,
    #                      sync_view=False)