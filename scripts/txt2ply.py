import struct


def read_point_cloud(filename):
    points = []
    with open(filename, "r") as file:
        for line in file:
            x, y, z = map(float, line.split())
            points.append((x, y, z))
    return points


def read_mesh_indices(filename):
    triangles = []
    with open(filename, "r") as file:
        for line in file:
            indices = map(int, line.split())
            triangles.append(indices)
    return triangles


def write_ply(filename, points, triangles):
    with open(filename, "w") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write(f"element face {len(triangles)}\n")
        file.write("property list uchar int vertex_index\n")
        file.write("end_header\n")
        for x, y, z in points:
            file.write(f"{x} {y} {z}\n")
        for triangle in triangles:
            file.write(f"3 {' '.join(map(str, triangle))}\n")


def write_ply_binary(filename, points, triangles):
    with open(filename, "wb") as file:
        # ヘッダーの書き込み
        file.write(b"ply\n")
        file.write(b"format binary_little_endian 1.0\n")
        file.write(f"element vertex {len(points)}\n".encode())
        file.write(b"property float x\n")
        file.write(b"property float y\n")
        file.write(b"property float z\n")
        file.write(f"element face {len(triangles)}\n".encode())
        file.write(b"property list uchar int vertex_index\n")
        file.write(b"end_header\n")

        # 頂点データの書き込み
        for point in points:
            file.write(struct.pack("<fff", *point))

        # 面データの書き込み
        for triangle in triangles:
            # 面を構成する頂点の数（この場合は3）と頂点インデックスを書き込む
            file.write(struct.pack("<Biii", 3, *triangle))


# ファイルからデータを読み込む
points = read_point_cloud("../data/armadillo-g-x.txt")
triangles = read_mesh_indices("../data/armadillo-g-triangles.txt")

# PLYファイルを作成する
write_ply_binary("../data/armadillo-x-mesh.ply", points, triangles)
