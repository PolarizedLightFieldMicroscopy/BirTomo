import matplotlib.pyplot as plt


def setup_visualization(window_title, plot_live=True, fig_size=(10, 9)):
    if plot_live:
        plt.ion()
    else:
        plt.ioff()
    figure = plt.figure(figsize=fig_size)
    plt.rcParams["image.origin"] = "upper"
    manager = plt.get_current_fig_manager()
    manager.set_window_title(window_title)
    if False:
        manager = plt.get_current_fig_manager()
        manager.window.setGeometry(4000, 600, 1800, 900)
    return figure
