import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pytorch3d.utils import ico_sphere
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.loss import mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    SoftSilhouetteShader,
)

# Constants for rasterization settings and optimization parameters
SIGMA = 1e-4
NUM_VIEWS_PER_ITERATION = 2
N_ITER = 2000
PLOT_PERIOD = 250

# Optimization losses configuration
LOSSES = {
    "rgb": {"weight": 1.0, "values": []},
    "silhouette": {"weight": 1.0, "values": []},
    "edge": {"weight": 1.0, "values": []},
    "normal": {"weight": 0.01, "values": []},
    "laplacian": {"weight": 1.0, "values": []},
}


# Check for CUDA availability
def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Load and preprocess the mesh
def load_and_preprocess_mesh(obj_path, device):
    mesh = load_objs_as_meshes([obj_path], device=device)
    verts = mesh.verts_packed()
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-center)
    mesh.scale_verts_((1.0 / float(scale)))
    return mesh.to(device), scale, center


# Set up multiple cameras for viewing the mesh
def setup_cameras(num_views, device):
    elev = torch.linspace(0, 360, num_views)
    azim = torch.linspace(-180, 180, num_views)
    R, T = look_at_view_transform(2.7, elev, azim)
    mv_cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    sv_camera = FoVPerspectiveCameras(device=device, R=R[None, 1, ...], T=T[None, 1, ...])
    return mv_cameras, sv_camera


# Set up point lights for rendering
def setup_lights(device):
    return PointLights(device=device, location=[[0.0, 0.0, -3.0]])


# Configure the silhouette renderer
def silhouette_renderer_settings(sv_camera, image_size):
    raster_settings_silhouette = RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1. / 1e-4 - 1.) * SIGMA,
        faces_per_pixel=50,
        perspective_correct=False
    )
    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=sv_camera, raster_settings=raster_settings_silhouette),
        shader=SoftSilhouetteShader()
    )


# Configure the texture renderer
def texture_renderer_settings(sv_camera, image_size, lights):
    raster_settings_soft = RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1. / 1e-4 - 1.) * SIGMA,
        faces_per_pixel=50,
        perspective_correct=False,
    )
    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=sv_camera, raster_settings=raster_settings_soft),
        shader=SoftPhongShader(device=device, cameras=sv_camera, lights=lights)
    )


# Configure the multi-view renderer
def multi_view_renderer_settings(mv_camera, image_size, lights):
    raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1)
    return MeshRenderer(
        rasterizer=MeshRasterizer(cameras=mv_camera, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=mv_camera, lights=lights),
    )


# Render silhouette images from multiple views
def render_silhouette_images(mesh, mv_cameras, silhouette_renderer, num_views):
    meshes = mesh.extend(num_views)
    silhouette_images = silhouette_renderer(meshes, cameras=mv_cameras)
    return silhouette_images, [silhouette_images[i, ..., 3] for i in range(num_views)]


# Render multi-view images
def render_multi_view(mesh, mv_cameras, mv_renderer, num_views):
    meshes = mesh.extend(num_views)
    target_images = mv_renderer(meshes, cameras=mv_cameras)
    return [target_images[i, ..., :3] for i in range(num_views)]


# Visualize the predicted mesh and target silhouette
def visualize_prediction(predicted_mesh, renderer, target_image, title='', silhouette=False):
    inds = 3 if silhouette else range(3)
    with torch.no_grad():
        predicted_images = renderer(predicted_mesh)
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(predicted_images[0, ..., inds].cpu().detach().numpy())
    plt.subplot(1, 2, 2)
    plt.imshow(target_image.cpu().detach().numpy())
    plt.title(title)
    plt.axis("off")


# Plot the losses during optimization
def plot_losses(losses):
    fig = plt.figure(figsize=(13, 5))
    ax = fig.gca()
    for k, l in losses.items():
        ax.plot(l['values'], label=k + " loss")
    ax.legend(fontsize="16")
    ax.set_xlabel("Iteration", fontsize="16")
    ax.set_ylabel("Loss", fontsize="16")
    ax.set_title("Loss vs iterations", fontsize="16")


# Update mesh shape prior losses
def update_mesh_shape_prior_losses(new_src_mesh, loss):
    loss["edge"] = mesh_edge_loss(new_src_mesh)
    loss["normal"] = mesh_normal_consistency(new_src_mesh)
    loss["laplacian"] = mesh_laplacian_smoothing(new_src_mesh, method="uniform")


# Initialize deformable vertices and optimizer
def deform_mesh_and_texture(src_mesh, device):
    verts_shape = src_mesh.verts_packed().shape
    deform_verts = torch.full(verts_shape, 0.0, device=device, requires_grad=True)
    sphere_verts_rgb = torch.full([1, verts_shape[0], 3], 0.5, device=device, requires_grad=True)
    optimizer = torch.optim.SGD([deform_verts, sphere_verts_rgb], lr=1.0, momentum=0.9)
    return optimizer, sphere_verts_rgb, deform_verts


# Save the mesh to an OBJ file
def save_mesh(new_src_mesh, scale, center, file_path):
    final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
    final_verts = final_verts * scale + center
    save_obj(file_path, final_verts, final_faces)


# Main loop for mesh refinement
def refine_mesh_and_texture(src_mesh, optimizer, deform_verts, sphere_verts_rgb, num_views,
                            texture_renderer, mv_cameras, target_silhouette,
                            target_rgb, scale, center, save_path):

    loop = tqdm(range(N_ITER))
    for i in loop:
        optimizer.zero_grad()
        new_src_mesh = src_mesh.offset_verts(deform_verts)
        new_src_mesh.textures = TexturesVertex(verts_features=torch.clip(sphere_verts_rgb, 0, 1))
        loss = {k: torch.tensor(0.0, device=device) for k in LOSSES}
        update_mesh_shape_prior_losses(new_src_mesh, loss)
        for j in np.random.permutation(num_views).tolist()[:NUM_VIEWS_PER_ITERATION]:
            images_predicted = texture_renderer(new_src_mesh, cameras=mv_cameras[j], lights=lights)
            predicted_silhouette = images_predicted[..., 3]
            loss_silhouette = ((predicted_silhouette - target_silhouette[j]) ** 2).mean()
            loss["silhouette"] += loss_silhouette / NUM_VIEWS_PER_ITERATION
            predicted_rgb = images_predicted[..., :3]
            loss_rgb = ((predicted_rgb - target_rgb[j]) ** 2).mean()
            loss["rgb"] += loss_rgb / NUM_VIEWS_PER_ITERATION

        sum_loss = torch.tensor(0.0, device=device)
        for k, l in loss.items():
            sum_loss += l * LOSSES[k]["weight"]
            LOSSES[k]["values"].append(float(l.detach().cpu()))

        loop.set_description("total_loss = %.6f" % sum_loss)

        if i % PLOT_PERIOD == 0:
            visualize_prediction(new_src_mesh, texture_renderer, target_rgb[1], title="iter: %d" % i,
                                 silhouette=False)
            # plt.show()

        sum_loss.backward()
        optimizer.step()

    save_mesh(new_src_mesh, scale, center, save_path)


# Main function
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python texture_prediction.py <obj_path> <image_views> <image_size> <output_obj_path>")
        sys.exit(1)

    obj_path, image_views, image_size, output_obj_path = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    device = get_device()

    mesh, scale, center = load_and_preprocess_mesh(obj_path, device)
    mv_cameras, sv_camera = setup_cameras(image_views, device)
    lights = setup_lights(device)

    mv_renderer = multi_view_renderer_settings(mv_cameras, image_size, lights)
    target_rgb = render_multi_view(mesh, mv_cameras, mv_renderer, image_views)

    silhouette_renderer = silhouette_renderer_settings(sv_camera, image_size)
    silhouette_images, target_silhouette = render_silhouette_images(mesh, mv_cameras, silhouette_renderer, image_views)

    texture_renderer = texture_renderer_settings(sv_camera, image_size, lights)

    src_mesh = ico_sphere(4, device)
    optimizer, sphere_verts_rgb, deform_verts = deform_mesh_and_texture(src_mesh, device)

    refine_mesh_and_texture(src_mesh, optimizer, deform_verts, sphere_verts_rgb, image_views,
                            texture_renderer, mv_cameras, target_silhouette,
                            target_rgb, scale, center, output_obj_path)
