import numpy as np

def rays_through_vol(pixels_per_ml, naObj, nMedium, volCtr):
    '''Identifies the rays that pass through the volume and the central lenslet
    Parameters:
        pixels_per_ml (int): number of pixels per microlens in one direction,
                                preferrable to be odd integer so there is a central
                                lenslet
        naObj (float): numerical aperture of the objective lens
        nMedium (float): refractive index of the volume
        volCtr (np.array): 3D vector containing the coordinates of the center of the
                            volume in volume space units (um)
    Returns:
        ray_enter (np.array): (3, X, X) array where (3, i, j) gives the coordinates 
                                within the volume ray entrance plane for which the 
                                ray that is incident on the (i, j) pixel with intersect
        ray_exit (np.array): (3, X, X) array where (3, i, j) gives the coordinates 
                                within the volume ray exit plane for which the 
                                ray that is incident on the (i, j) pixel with intersect
        ray_diff (np.array): (3, X, X) array giving the direction of the rays through 
                                the volume
    
    '''
    # Units are in pixel indicies, referring to the pixel that is centered up 0.5 units
    #   Ex: if ml_ctr = [8, 8], then the spatial center pixel is at [8.5, 8.5]
    ml_ctr = [(pixels_per_ml - 1)/ 2, (pixels_per_ml - 1)/ 2]
    ml_radius = pixels_per_ml / 2
    i = np.linspace(0, pixels_per_ml - 1, pixels_per_ml)
    j = np.linspace(0, pixels_per_ml - 1, pixels_per_ml)
    jv, iv = np.meshgrid(i, j)
    dist_from_ctr = np.sqrt((iv - ml_ctr[0]) ** 2 + (jv - ml_ctr[1]) ** 2)

    # Angles that reach the pixels
    cam_pixels_azim = np.arctan2(jv - ml_ctr[1], iv - ml_ctr[0])
    cam_pixels_azim[dist_from_ctr > ml_radius] = np.NaN
    dist_from_ctr[dist_from_ctr > ml_radius] = np.NaN #
    cam_pixels_tilt = np.arcsin(dist_from_ctr / ml_radius * naObj / nMedium)

    # Positions of the ray in volume coordinates
    # assuming rays pass through the center voxel
    ray_enter_x = np.zeros([pixels_per_ml, pixels_per_ml])
    ray_enter_y = volCtr[0] * np.tan(cam_pixels_tilt) * np.sin(cam_pixels_azim) + volCtr[1]
    ray_enter_z = volCtr[0] * np.tan(cam_pixels_tilt) * np.cos(cam_pixels_azim) + volCtr[2]
    ray_enter_x[np.isnan(ray_enter_y)] = np.NaN
    ray_enter = np.array([ray_enter_x, ray_enter_y, ray_enter_z])
    vol_ctr_grid_tmp = np.array([np.full((pixels_per_ml, pixels_per_ml), volCtr[i]) for i in range(3)])
    ray_exit = ray_enter + 2 * (vol_ctr_grid_tmp - ray_enter)

    # Direction of the rays at the exit plane
    ray_diff = ray_exit - ray_enter
    ray_diff = ray_diff / np.linalg.norm(ray_diff, axis=0)
    return ray_enter, ray_exit, ray_diff

def main():
    # Volume shape
    voxNrX = 5
    voxNrYZ = 5
    voxPitch = 1
    axialPitch = voxPitch
    # voxCtr = np.array([(voxNrX - 1) / 2, (voxNrYZ - 1) / 2, (voxNrYZ - 1) / 2]) # in index units
    voxCtr = np.array([voxNrX / 2, voxNrYZ / 2, voxNrYZ / 2]) # in index units
    volCtr = [voxCtr[0] * axialPitch, voxCtr[1] * voxPitch, voxCtr[2] * voxPitch]   # in vol units (um)
    ray_enter, ray_exit, ray_diff = rays_through_vol(17, 1.2, 1.52, volCtr)

if __name__ == '__main__':
    main()