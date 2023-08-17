# Description
The object directory contains h5 files of objects/volumes that can be used in the forward model.

27 Dec 2022:  
### Bundle objects by Rudolf:
Structure of h5 files of bundles of different orientations  
<img width="378" alt="image" src="https://user-images.githubusercontent.com/2894530/207461377-c79d1554-406b-4283-a3ed-99366ee99749.png">

Over Christmas, Saint Nick (SN) has dropped four new HDF5 files in the stream branch:  
![bundles_SN](https://user-images.githubusercontent.com/2894530/209747520-7644656a-080b-406e-abd2-ecbcb075aafe.jpeg)
In each bundle object, the delta_n data (birefringence) are given as a 3D array delta_n[Z,Y,X] with indices identifying voxels along the Z-, Y-, and X-axis.  
The optic_axis data are given as a four dimensional array optic_axis[oa,Z,Y,X] with oa representing the optic axis as a 3D unit vector for each voxel [Z,Y,X]. The order of the vector components is [oaZ,oaY,oaX].

bundle radius = 2.5 voxels  
bundle length = 7 voxels  
bundle birefringence = -0.01 and optic axis direction parallel to bundle axis
