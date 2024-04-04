import igl
import numpy as np
import pyvista

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

    # 主曲率方向ベクトルを表示するためのラインを作成
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

if __name__ == "__main__":
    # path = '/workspace/Data/Test/mug28_manifold.ply'
    # path = '/workspace/Data/Test/Mug1_remesh.ply'
    # path = '/workspace/Data/Test/sorted_mug100.ply'
    path = '/workspace/Data/Test/sort_mug100_remesh_recnst.ply'
    verts, tris = igl.read_triangle_mesh(path)

    d1, d2, v1, v2 = igl.principal_curvature(verts, 
                                             tris,
                                             radius=5,
                                             use_k_ring=True)
    v3 = 0.5 * (v1 + v2)
    d3 = np.cross(d1, d2)
    sample_rate  = 0.05
    sample_indices = np.random.choice(len(verts), int(len(verts) * sample_rate), replace=False)
    msh, vf1, vf2, vf3 = visualize_principal_curvature(verts, tris, d1, v1, d2, v2, d3, v3,
                                                       indices=sample_indices,
                                                       scale=10
                                                       )
    plotter = pyvista.Plotter()
    plotter.add_mesh(msh, color='gray', opacity=0.3)
    plotter.add_mesh(vf1, line_width=2, color='r')
    plotter.add_mesh(vf2, line_width=2, color='g')
    plotter.add_mesh(vf3, line_width=2, color='b')
    plotter.show()