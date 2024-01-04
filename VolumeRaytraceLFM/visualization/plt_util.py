import matplotlib.pyplot as plt


def setup_visualization(window_title):
    plt.ion()
    fig_size = (10, 9)
    figure = plt.figure(figsize=fig_size)
    plt.rcParams['image.origin'] = 'lower'
    manager = plt.get_current_fig_manager()
    manager.set_window_title(window_title)
    if False:
        manager = plt.get_current_fig_manager()
        manager.window.setGeometry(4000, 600, 1800, 900)
    return figure
