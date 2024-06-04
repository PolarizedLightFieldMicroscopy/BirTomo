import os
import torch
from VolumeRaytraceLFM.birefringence_implementations import (
    BirefringentVolume,
    BirefringentRaytraceLFM,
)

dir = "shell_11x51x51_shift-1_reg-1"
dir = "shell_13x51x51_mask_volshape"
dir = "Oct23/voxel_5x11x11_shere_test"
dir = "Oct23/sphere_15x51x51_shift-1"

output_dir = "reconstructions/" + dir

file_path = f"{output_dir}/parameters.pt"
data = torch.load(file_path)

optical_info = data["optical_info"]
training_params = data["training_params"]
# volume_type = data['volume_type']
vol_args = data["vol_args"]

print("Optical Info:", optical_info)
print("Training Params:", training_params)
# print("Volume Type:", volume_type)
print("Volume creation args:", vol_args)

volume_filename = os.path.join("reconstructions", dir, "volume_ep_200.h5")

volume = BirefringentVolume.init_from_file(
    volume_filename,
    # backend=backend,
    optical_info=optical_info,
)

# Plot volume
with torch.no_grad():
    # Plot the optic axis and birefringence within the volume
    plotly_figure = volume.plot_lines_plotly()
    # Append volumes to plot
    plotly_figure = volume.plot_volume_plotly(
        optical_info, voxels_in=volume.get_delta_n(), opacity=0.02, fig=plotly_figure
    )
    plotly_figure.show()
