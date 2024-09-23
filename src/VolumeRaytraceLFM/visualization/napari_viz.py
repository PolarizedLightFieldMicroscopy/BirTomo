import napari
# from napari_animation import Animation
import numpy as np
# import matplotlib as mpl
import vispy as vs


def load_from_npz(file_path):
    '''Load the birfingence and optic axis data from a npz file'''
    # file_path = r"samples\celery_30_60_140_from2024-05-15_17-40-05_random1.npz"
    npz = np.load(file_path)
    birefringence = npz['birefringence']
    optic_axis = npz['optic_axis']
    return birefringence, optic_axis

def load_np_from_birVol(volume):
        '''Gets the data from the volume we will need for plotting in napari'''
        delta_n = volume.get_delta_n() * 1
        optic_axis = volume.get_optic_axis() * 1
        optical_info = volume.optical_info
        # Check if this is a torch tensor
        if not isinstance(delta_n, np.ndarray):
            try:
                delta_n = delta_n.cpu().detach().numpy()
                optic_axis = optic_axis.cpu().detach().numpy()
            except:
                pass

        
        if "voxel_size_um" not in optical_info:
            optical_info["voxel_size_um"] = [1, 1, 1]
            print(
                "Notice: 'voxel_size_um' was not found in optical_info. Size of [1, 1, 1] assigned."
            )

        um_per_pix = tuple(optical_info["voxel_size_um"])
        
        return delta_n,optic_axis,um_per_pix

def bir_threshold(optic_axis,birefringence, threshold = .001):
    ## make adjust data for ploting in napari
    # no_birefringence = np.abs(birefringence)<=np.percentile(np.abs(birefringence),95)
    no_birefringence = np.abs(birefringence)<= threshold
    optic_axis[:,no_birefringence] = 0
    return optic_axis

def move_comps_to_end(optic_axis):
    return np.moveaxis(optic_axis, 0, -1)
    
def to_Nx2xD(img_like_vect, birefringence = None, omit_zeros = False):
    '''Converts a image like vector field into a list of N vector and position pairs. If omit zeros is true, birefringence != 0  is used as a mask. The values of birefringence are put into a features dic for napari to read in to color the vectors.'''
    img_like_vect = move_comps_to_end(img_like_vect)
    dims = np.array(img_like_vect.shape)
    N = np.prod(dims[:-1])
    pos_vect = np.zeros((N,2,dims[-1]))
    if birefringence is None:
        if omit_zeros:
            nz_count = 0
            for i in range(dims[0]):
                for j in range(dims[1]):
                    for k in range(dims[2]):
                        if np.any(img_like_vect[i,j,k,:]!=0):
                            pos_vect[nz_count,0,:] = np.array([i,j,k])
                            pos_vect[nz_count,1,:] = img_like_vect[i,j,k,:]
                            nz_count += 1
            pos_vect = pos_vect[:nz_count,:,:]
        else:
            for i in range(dims[0]):
                for j in range(dims[1]):
                    for k in range(dims[2]):
                        pos_vect[i*dims[1]*dims[2]+j*dims[2]+k,0,:] = np.array([i,j,k])
                        pos_vect[i*dims[1]*dims[2]+j*dims[2]+k,1,:] = img_like_vect[i,j,k,:]
        return pos_vect
    else:
        bir = np.zeros(N)
        if omit_zeros:
            nz_count = 0
            for i in range(dims[0]):
                for j in range(dims[1]):
                    for k in range(dims[2]):
                        if np.any(img_like_vect[i,j,k,:]!=0):
                            pos_vect[nz_count,0,:] = np.array([i,j,k])
                            pos_vect[nz_count,1,:] = img_like_vect[i,j,k,:]
                            bir[nz_count] = birefringence[i,j,k]
                            nz_count += 1
            pos_vect = pos_vect[:nz_count,:,:]
            bir = bir[:nz_count]
        else:
            for i in range(dims[0]):
                for j in range(dims[1]):
                    for k in range(dims[2]):
                        pos_vect[i*dims[1]*dims[2]+j*dims[2]+k,0,:] = np.array([i,j,k])
                        pos_vect[i*dims[1]*dims[2]+j*dims[2]+k,1,:] = img_like_vect[i,j,k,:]
                        bir[i*dims[1]*dims[2]+j*dims[2]+k] = birefringence[i,j,k]
        features = {
            'bir':bir
        }
        return pos_vect, features

def to_Nx2xD_z_slice(img_like_vect,birefringence = None, omit_zeros = False):
    '''Converts a image like vector field into a list of N vector and position pairs and adds another dimension so we can view slices in along the z axis. At each position on the new axis, only vectors with the same z value are added. We can scroll through this axis in napari to change which z plane is visible. If omit zeros is true, birefringence != 0  is used as a mask. The values of birefringence are put into a features dic for napari to read in to color the vectors.'''
    img_like_vect = move_comps_to_end(img_like_vect)
    dims = np.array(img_like_vect.shape)
    N = np.prod(dims[:-1])
    pos_vect = np.zeros((N,2,dims[-1]+1))
    if birefringence is None:
        if omit_zeros:
            nz_count = 0
            for i in range(dims[0]):
                for j in range(dims[1]):
                    for k in range(dims[2]):
                        if np.any(img_like_vect[i,j,k,:]!=0):
                            pos_vect[nz_count,0,:] = np.array([i,i,j,k])
                            pos_vect[nz_count,1,1:] = img_like_vect[i,j,k,:]
                            nz_count += 1
            pos_vect = pos_vect[:nz_count,:,:]
        else:
            for i in range(dims[0]):
                for j in range(dims[1]):
                    for k in range(dims[2]):
                        pos_vect[i*dims[1]*dims[2]+j*dims[2]+k,0,:] = np.array([i,i,j,k])
                        pos_vect[i*dims[1]*dims[2]+j*dims[2]+k,1,1:] = img_like_vect[i,j,k,:]
        return pos_vect
    else:
        bir = np.zeros(N)
        if omit_zeros:
            nz_count = 0
            for i in range(dims[0]):
                for j in range(dims[1]):
                    for k in range(dims[2]):
                        if np.any(img_like_vect[i,j,k,:]!=0):
                            pos_vect[nz_count,0,:] = np.array([i,i,j,k])
                            pos_vect[nz_count,1,1:] = img_like_vect[i,j,k,:]
                            bir[nz_count] = birefringence[i,j,k]
                            nz_count += 1
            pos_vect = pos_vect[:nz_count,:,:]
            bir = bir[:nz_count]

        else:
            for i in range(dims[0]):
                for j in range(dims[1]):
                    for k in range(dims[2]):
                        pos_vect[i*dims[1]*dims[2]+j*dims[2]+k,0,:] = np.array([i,i,j,k])
                        pos_vect[i*dims[1]*dims[2]+j*dims[2]+k,1,1:] = img_like_vect[i,j,k,:]
                        bir[i*dims[1]*dims[2]+j*dims[2]+k] = birefringence[i,j,k]
        features = {
            'bir':bir
        }
        return pos_vect, features

# print(to_Nx2xD(cropped_optic_axis).shape)
# print(to_Nx2xD(cropped_optic_axis, omit_zeros=True).shape)






def open_viewer():
    viewer = napari.Viewer(ndisplay = 3)
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = 'um'
    viewer.axes.visible = True
    viewer.axes.colored = False
    return viewer

def plot_all_vectors(birefringence, optic_axis, viewer = None, colorlims = None, colormap = 'viridis_r', um_per_pix = 1.7333):
        
    # reshape the image like numpy arrays into a list of vectors and positions that napari wants
    print("Reshaping the birefringence data for napari...")
    all_vects , all_vects_bir= to_Nx2xD(optic_axis, birefringence=birefringence, omit_zeros=True)
    # print('Done!')

    # open up a new viewer if we dont have a current viewer
    if viewer is None:
        print('Creating a new napari viewer...')
        viewer = open_viewer()
    
    # set the
    if colorlims is None:
        colorlims = (.000,np.max(birefringence))

    if type(colormap) is str:
        # treat colormap as the name of the colormap
        # and load it as a vispy colormap
        vs_cmap = vs.color.colormap.MatplotlibColormap(colormap)
    else:
        # try treating the colormap as an existing colormap 
        # and send it to napari as is. This likly will not work 
        # if it is not a vispy colormap, because napari uses vispy colormaps
        vs_cmap = colormap

    if isinstance(um_per_pix, (tuple)) and len(um_per_pix) == 3:
        scale3 = um_per_pix
    elif isinstance(um_per_pix, (float,int)):
        scale3 = (um_per_pix,um_per_pix,um_per_pix)
    else:
        raise TypeError(f"um_per_pix is niether a tuple with len 3 or a float or int. It is a {type(um_per_pix)}, {print(um_per_pix)} and I'm not sure what to do with this")
    print('Adding vectors to the viewer...')
    all_vectors =  viewer.add_vectors(all_vects,features=all_vects_bir,edge_color='bir',vector_style='line',scale=scale3, edge_contrast_limits=colorlims, edge_colormap=vs_cmap, edge_width=.3,length=1,opacity=.75,blending='opaque')
    return viewer,all_vectors


def plot_sliced_vectors(birefringence, optic_axis, viewer = None, colorlims = None, colormap = 'viridis_r', um_per_pix = 1.7333):
        
    # reshape the image like numpy arrays into a list of vectors and positions that napari wants
    print("Reshaping the birefringence data for napari...")
    sliced_vects , sliced_bir= to_Nx2xD_z_slice(optic_axis, birefringence=birefringence, omit_zeros=True)
    print('Done!')
    # open up a new viewer if we dont have a current viewer
    if viewer is None:
        print('Creating a new viewer')
        viewer = open_viewer()
    
    # set the
    if colorlims is None:
        colorlims = (.000,np.max(birefringence))

    if type(colormap) is str:
        # treat colormap as the name of the colormap
        # and load it as a vispy colormap
        vs_cmap = vs.color.colormap.MatplotlibColormap(colormap)
    else:
        # try treating the colormap as an existing colormap 
        # and send it to napari as is. This likly will not work 
        # if it is not a vispy colormap, because napari uses vispy colormaps
        vs_cmap = colormap

    if isinstance(um_per_pix, (tuple)) and len(um_per_pix) == 3:
        scale4 = (um_per_pix[0],)+um_per_pix
    elif isinstance(um_per_pix, (float,int)):
        scale4 = (um_per_pix,um_per_pix,um_per_pix,um_per_pix)
    else:
        raise TypeError(f"um_per_pix is niether a tuple with len 3 or a float or int. It is a {type(um_per_pix)} and I'm not sure what to do with this")

    print('Adding vectors to the viewer')
    sliced_vectors = viewer.add_vectors(sliced_vects,features=sliced_bir,edge_color='bir',vector_style='line',scale=scale4, edge_contrast_limits=colorlims, edge_colormap=vs_cmap, edge_width=.3,length=1,opacity=1,blending='opaque')
    return viewer, sliced_vectors

def viz_vol(volume,viewer = None):
    birefringence,optic_axis,um_per_pix = load_np_from_birVol(volume)
    optic_axis = bir_threshold(optic_axis,birefringence)
    viewer,all = plot_all_vectors(birefringence,optic_axis,viewer=viewer,um_per_pix=um_per_pix)
    napari.run()
    return viewer, all

if __name__ == '__main__':
    # birefringence, optic_axis = load_from_npz(r"C:\Users\trevo\Desktop\celery_30_60_140_from2024-05-15_17-40-05_random1.npz")
    # print(birefringence.shape)
    # print(optic_axis.shape)
    # optic_axis = bir_threshold(optic_axis,birefringence)
    # viewer,sliced = plot_sliced_vectors(birefringence, optic_axis)
    # _,all = plot_all_vectors(birefringence,optic_axis,viewer=viewer)
    # napari.run()
    
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
    # plotly_figure = volume.plot_lines_plotly(draw_spheres=True)
    # plotly_figure.show()
    viz_vol(volume)



# sliced_vects , sliced_vects_bir= to_Nx2xD_z_slice(cropped_optic_axis,birefringence=birefringence, omit_zeros=True)



# sliced_vs = viewer.add_vectors(sliced_vects,features=sliced_vects_bir,edge_color='bir',vector_style='line', scale=scale4, edge_width=.1,length=50,opacity=1,blending='additive')
# all_vs = viewer.add_vectors(all_vects,features=all_vects_bir,edge_color='bir',vector_style='line',scale=scale3, edge_width=.1,length=50,opacity=.1,blending='additive')



# viewer.camera.angles = (13.407327806969164, -34.44222242422218, 160.20054598704667)

# napari.utils.colormaps.colorbar()
'''
import matplotlib.pyplot as plt
import matplotlib as mpl

font = {'size'   : 32}

plt.rc('font', **font)
plt.rc('axes', labelsize=40) 

fig, ax = plt.subplots(figsize=(14, 2), layout='constrained')

mpl_cmap = mpl.colormaps[colormap_name]
# cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=colorlims[0], vmax=colorlims[1])

fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=mpl_cmap),
             cax=ax, orientation='horizontal', label='Birefringence')

ticks = np.linspace(0,.016,5)
tick_labels = [f'{tick}' for tick in ticks]
ax.set_xticks(ticks,tick_labels)
'''