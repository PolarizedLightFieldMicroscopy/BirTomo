import time
import matplotlib.pyplot as plt
from plotting_tools import plot_iteration_update, plot_retardance_orientation
from VolumeRaytraceLFM.optic_config import volume_2_projections

def handle_visualization_and_saving(ep, Delta_n_GT, ret_image_measured, azim_image_measured, volume_estimation, figure, output_dir, ret_image_current, azim_image_current, losses, data_term_losses, regularization_term_losses):
    if ep % 10 == 0:
        plt.clf()
        plot_iteration_update(
            volume_2_projections(Delta_n_GT.unsqueeze(0))[0, 0].detach().cpu().numpy(),
            ret_image_measured.detach().cpu().numpy(),
            azim_image_measured.detach().cpu().numpy(),
            volume_2_projections(volume_estimation.get_delta_n().unsqueeze(0))[0, 0].detach().cpu().numpy(),
                ret_image_current.detach().cpu().numpy(),
                azim_image_current.detach().cpu().numpy(),
                losses,
                data_term_losses,
                regularization_term_losses
        )
        figure.canvas.draw()
        figure.canvas.flush_events()
        time.sleep(0.1)
        plt.savefig(f"{output_dir}/Optimization_ep_{'{:02d}'.format(ep)}.pdf")
        time.sleep(0.1)

    if ep % 100 == 0:
        volume_estimation.save_as_file(f"{output_dir}/volume_ep_{'{:02d}'.format(ep)}.h5")
    return
