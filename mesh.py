"""Mesh cleanup, scaling and export."""
import numpy as np
import open3d as o3d


def load_mesh(path):
    return o3d.io.read_triangle_mesh(str(path))


def clean_mesh(mesh, target_triangles=200_000, smooth_iters=3):
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    # keep only the largest connected component (drops floating junk)
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels, counts = np.asarray(labels), np.asarray(counts)
    if len(counts):
        mesh.remove_triangles_by_mask(labels != counts.argmax())
        mesh.remove_unreferenced_vertices()
    if smooth_iters:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smooth_iters)
    if len(mesh.triangles) > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_triangles)
    mesh.compute_vertex_normals()
    return mesh


def scale_to_real(mesh, measured_model_units, real_mm):
    """Uniformly scale so a known reference distance matches real millimetres."""
    factor = real_mm / measured_model_units
    mesh.scale(factor, center=(0, 0, 0))
    return mesh, factor


def export_stl(mesh, path):
    mesh.compute_triangle_normals()
    o3d.io.write_triangle_mesh(str(path), mesh)
    return path
