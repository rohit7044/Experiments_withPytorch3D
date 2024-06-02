import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex
)
from plot_image_grid import image_grid

# CONSTANTS
# Rasterization settings for silhouette rendering
sigma = 1e-4


def check_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_mesh(obj_path, device):
    return load_objs_as_meshes([obj_path], device=device)


# Generate camera poses for each view
def setup_cameras(num_views, bb_size, device):
    elev = torch.linspace(0, 360, num_views)
    azim = torch.linspace(-180, 180, num_views)
    R, T = look_at_view_transform(bb_size * 2.0, elev, azim)
    mv_cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    sv_camera = FoVPerspectiveCameras(device=device, R=R[None, 1, ...],
                                      T=T[None, 1, ...])
    return mv_cameras, sv_camera


# Setup lighting
def setup_lights(device):
    return PointLights(device=device, location=[[0.0, 0.0, -3.0]])


# Render silhouette images.  The 3rd channel of the rendering output is
# the alpha/silhouette channel
def render_silhouette_images(mesh, mv_cameras, sv_camera, lights, num_views, image_size):
    raster_settings_silhouette = RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
        faces_per_pixel=50,
    )
    # Silhouette renderer
    renderer_silhouette = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=sv_camera,
            raster_settings=raster_settings_silhouette
        ),
        shader=SoftSilhouetteShader()
    )
    meshes = mesh.extend(num_views)
    silhouette_images = renderer_silhouette(meshes, cameras=mv_cameras, lights=lights)
    target_silhouette = [silhouette_images[i, ..., 3] for i in range(num_views)]

    # Visualize silhouette images
    image_grid(silhouette_images.cpu().numpy(), rows=4, cols=5, rgb=False)
    plt.show()


def compute_bounding_box(mesh):
    # Compute the bounding box of the mesh
    vertices = mesh.verts_packed()
    min_xyz = vertices.min(dim=0).values
    max_xyz = vertices.max(dim=0).values
    center = (min_xyz + max_xyz) / 2
    size = (max_xyz - min_xyz).max()
    return size


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python mesh_prediction.py <obj_path> <image_views> <image_size>")
        sys.exit(1)

    obj_path = sys.argv[1]
    # images_path = int(sys.argv[2])
    images_views = int(sys.argv[2])
    image_size = int(sys.argv[3])
    device = check_device()
    obj = load_mesh(obj_path, device)
    bb_size = compute_bounding_box(obj)
    mv_cameras, sv_camera = setup_cameras(images_views, bb_size, device)
    lights = setup_lights(device)
    render_silhouette_images(obj, mv_cameras, sv_camera, lights, images_views, image_size)
