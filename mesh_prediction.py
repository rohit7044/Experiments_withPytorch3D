import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pytorch3d.utils import ico_sphere
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
)
from plot_image_grid import image_grid

# Constants for rasterization settings and optimization parameters
sigma = 1e-4
num_views_per_iteration = 2
Niter = 2000
plot_period = 250
losses = {
    "silhouette": {"weight": 1.0, "values": []},
    "edge": {"weight": 1.0, "values": []},
    "normal": {"weight": 0.01, "values": []},
    "laplacian": {"weight": 1.0, "values": []},
}


# Check for CUDA availability
def check_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Load and return a mesh from the given OBJ file path
def load_mesh(obj_path, device):
    return load_objs_as_meshes([obj_path], device=device)


# Preprocess the mesh to fit within a unit sphere and center it
def preprocess_mesh(mesh, device):
    verts = mesh.verts_packed()
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-center)
    mesh.scale_verts_((1.0 / float(scale)))
    mesh = mesh.to(device)
    return mesh, scale, center


# Set up multiple cameras for viewing the mesh from different angles
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


# Configure the renderer for silhouette rendering
def silhouette_renderer_settings(sv_camera, image_size):
    raster_settings_silhouette = RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1. / 1e-4 - 1.) * sigma,
        faces_per_pixel=50,
        perspective_correct=False
    )
    renderer_silhouette = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=sv_camera,
            raster_settings=raster_settings_silhouette
        ),
        shader=SoftSilhouetteShader()
    )
    return renderer_silhouette


# Render silhouette images from multiple views
def render_silhouette_images(mesh, mv_cameras, silhouette_renderer, lights, num_views):
    meshes = mesh.extend(num_views)
    silhouette_images = silhouette_renderer(meshes, cameras=mv_cameras, lights=lights)
    target_silhouette = [silhouette_images[i, ..., 3] for i in range(num_views)]
    return silhouette_images, target_silhouette


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
def deform_mesh(src_mesh):
    verts_shape = src_mesh.verts_packed().shape
    deform_verts = torch.full(verts_shape, 0.0, device=device, requires_grad=True)
    optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)
    return optimizer, deform_verts


# Save the mesh to an OBJ file
def save_mesh(new_src_mesh, scale, center, file_path):
    final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
    final_verts = final_verts * scale + center
    save_obj(file_path, final_verts, final_faces)


# Main loop for mesh refinement
def refine_mesh(src_mesh, optimizer, deform_verts, num_views, silhouette_renderer, mv_cameras, target_silhouette, scale,
                center, save_path):
    loop = tqdm(range(Niter))
    new_src_mesh = src_mesh
    for i in loop:
        optimizer.zero_grad()
        new_src_mesh = src_mesh.offset_verts(deform_verts)
        loss = {k: torch.tensor(0.0, device=device) for k in losses}
        update_mesh_shape_prior_losses(new_src_mesh, loss)
        for j in np.random.permutation(num_views).tolist()[:num_views_per_iteration]:
            images_predicted = silhouette_renderer(new_src_mesh, cameras=mv_cameras[j], lights=lights)
            predicted_silhouette = images_predicted[..., 3]
            loss_silhouette = ((predicted_silhouette - target_silhouette[j]) ** 2).mean()
            loss["silhouette"] += loss_silhouette / num_views_per_iteration
        sum_loss = torch.tensor(0.0, device=device)
        for k, l in loss.items():
            sum_loss += l * losses[k]["weight"]
            losses[k]["values"].append(float(l.detach().cpu()))
        loop.set_description("total_loss = %.6f" % sum_loss)
        if i % plot_period == 0:
            visualize_prediction(new_src_mesh, title="iter: %d" % i, silhouette=True, target_image=target_silhouette[1],
                                 renderer=silhouette_renderer)
            plt.show()
        sum_loss.backward()
        optimizer.step()
    save_mesh(new_src_mesh, scale, center, save_path)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python mesh_prediction.py <obj_path> <image_views> <image_size> <output_obj_path>")
        sys.exit(1)
    try:
        obj_path = sys.argv[1]
        images_views = int(sys.argv[2])
        image_size = int(sys.argv[3])
        output_obj_path = sys.argv[4]
    except ValueError:
        print("Error: <image_views> and <image_size> should be integers.")
        sys.exit(1)

    device = check_device()
    obj = load_mesh(obj_path, device)
    mesh, scale, center = preprocess_mesh(obj, device)
    mv_cameras, sv_camera = setup_cameras(images_views, device)
    lights = setup_lights(device)
    silhouette_renderer = silhouette_renderer_settings(sv_camera, image_size)
    silhouette_images, target_silhouette = render_silhouette_images(mesh, mv_cameras, silhouette_renderer, lights,
                                                                    images_views)
    src_mesh = ico_sphere(4, device)
    optimizer, deform_verts = deform_mesh(src_mesh)
    refine_mesh(src_mesh, optimizer, deform_verts, images_views, silhouette_renderer, mv_cameras, target_silhouette,
                scale, center, output_obj_path)
