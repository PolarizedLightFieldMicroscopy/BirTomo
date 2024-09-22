# %% Importing necessary libraries
import time
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.simulations import ForwardModel
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume
from VolumeRaytraceLFM.volumes import volume_args
from VolumeRaytraceLFM.setup_parameters import setup_optical_parameters
from VolumeRaytraceLFM.visualization.plotting_volume import visualize_volume

# %% Setting up the optical system
BACKEND = BackEnds.PYTORCH
optical_info = setup_optical_parameters("config/optical_config_sphere.json")
optical_system = {"optical_info": optical_info}
print(optical_info)

# %% Create the simulator and volume
volume = BirefringentVolume(
    backend=BACKEND,
    optical_info=optical_info,
    volume_creation_args=volume_args.shell_small_args,
)
plotly_figure = volume.plot_lines_plotly(draw_spheres=True)
plotly_figure.show()

# # %% Image the volume
# simulator = ForwardModel(optical_system, backend=BACKEND)
# simulator.rays.prepare_for_all_rays_at_once()
# start_time = time.perf_counter()
# simulator.rays.reset_timing_info()
# simulator.forward_model(volume, all_lenslets=True)
# end_time = time.perf_counter()
# print(f"Forward pass took {end_time - start_time:.2f} seconds to image the volume.")

# # %% View timing information
# simulator.rays.print_timing_info()

# # %% View the images
# simulator.view_images()
# images = simulator.ret_img, simulator.azim_img
