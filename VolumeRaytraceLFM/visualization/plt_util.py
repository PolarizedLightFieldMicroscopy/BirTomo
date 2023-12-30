import matplotlib.pyplot as plt


def setup_visualization():
    plt.ion()
    fig_size = (10, 9)
    figure = plt.figure(figsize=fig_size)
    plt.rcParams['image.origin'] = 'lower'
    if False:
        manager = plt.get_current_fig_manager()
        manager.window.setGeometry(4000, 600, 1800, 900)
    return figure
