import sys
import os
import torch
import matplotlib.pyplot as plt
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader
)
from plot_image_grid import image_grid


# Compute the bounding box of a mesh
def compute_bounding_box(mesh):
    vertices = mesh.verts_packed()
    min_xyz = vertices.min(dim=0).values
    max_xyz = vertices.max(dim=0).values
    return min_xyz, max_xyz


# Generate multi-view images of a mesh
def generate_multi_view(obj_path, num_views, image_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the mesh from the OBJ file
    mesh = load_objs_as_meshes([obj_path], device=device)

    # Compute the bounding box of the mesh
    min_xyz, max_xyz = compute_bounding_box(mesh)
    center = (min_xyz + max_xyz) / 2
    size = (max_xyz - min_xyz).max()

    # Generate camera poses for each view
    elev = torch.linspace(0, 360, num_views)
    azim = torch.linspace(-180, 180, num_views)
    R, T = look_at_view_transform(size * 2.0, elev, azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Set up the renderer
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1
    )
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )

    # Render the mesh from each view
    meshes = mesh.extend(num_views)
    target_images = renderer(meshes, cameras=cameras, lights=lights)
    target_rgb = [target_images[i, ..., :3] for i in range(num_views)]

    return target_rgb

    # Display the rendered images in a grid
    # image_grid(target_images.cpu().numpy(), rows=4, cols=5, rgb=True)
    # plt.show()


# Save the rendered images to a directory
def save_images(images, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i, image in enumerate(images):
        plt.imsave(os.path.join(directory, f"view_{i}.png"), image.cpu().numpy())


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python multi_view.py <obj_path> <num_views> <image_size>")
        sys.exit(1)

    obj_path = sys.argv[1]
    num_views = int(sys.argv[2])
    image_size = int(sys.argv[3])
    images = generate_multi_view(obj_path, num_views, image_size)
    save_images(images, "rendered_images")
