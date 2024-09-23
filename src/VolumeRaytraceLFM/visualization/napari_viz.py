import napari
# from napari_animation import Animation
import numpy as np
import matplotlib as mpl
import vispy as vs


def load_from_numpy(file_path):
    '''Load the birfingence and optic axis data from a npz file'''
    # file_path = r"samples\celery_30_60_140_from2024-05-15_17-40-05_random1.npz"
    npz = np.load(file_path)
    birefringence = npz['birefringence']
    optic_axis = move_comps_to_end(npz['optic_axis'])
    return birefringence, optic_axis

def bir_threshold(optic_axis,birefringence, threshold = .001):
    ## make adjust data for ploting in napari
    # no_birefringence = np.abs(birefringence)<=np.percentile(np.abs(birefringence),95)
    no_birefringence = np.abs(birefringence)<= threshold
    optic_axis[no_birefringence] = 0
    return optic_axis

def move_comps_to_end(optic_axis):
    return np.moveaxis(optic_axis, 0, -1)
    
    

def to_Nx2xD(img_like_vect, birefringence = None, omit_zeros = False):
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
    all_vects , all_vects_bir= to_Nx2xD(optic_axis, birefringence=birefringence, omit_zeros=True)
    
    # open up a new viewer if we dont have a current viewer
    if viewer is None:
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

    if len(um_per_pix) == 3:
        scale3 = um_per_pix
    else:
        scale3 = (um_per_pix,um_per_pix,um_per_pix)

    all_vectors =  viewer.add_vectors(all_vects,features=all_vects_bir,edge_color='bir',vector_style='line',scale=scale3, edge_contrast_limits=colorlims, edge_colormap=vs_cmap, edge_width=.3,length=75,opacity=.75,blending='opaque')
    return all_vectors



if __name__ == '__main__':
    birefringence, optic_axis = load_from_numpy(r"C:\Users\trevo\Desktop\celery_30_60_140_from2024-05-15_17-40-05_random1.npz")
    print(birefringence.shape)
    print(optic_axis.shape)
    optic_axis = bir_threshold(optic_axis,birefringence)
    plot = plot_all_vectors(birefringence, optic_axis)



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