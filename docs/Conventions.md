# Geometric and Jones-calculus conventions

These conventions are relied on throughout the forward model. Changing one of
them requires changing the others consistently.

## Axial direction: index 0 is the objective side

Volume arrays are indexed `[axial, y, x]` (axis 0 is along the microscope's
optical axis), and the optic-axis vector is stored in the same component order.

**Axial index 0 is the slice nearest the objective (camera side).** Illumination
from the condenser enters the volume at the last axial slice and leaves through
slice 0 toward the objective. Light therefore propagates toward *decreasing*
axial index.

The ray geometry is traced backward from the camera:

- `ray_entry` lies on the axial-0 plane and `ray_exit` on the far plane.
- `ray_direction` and `ray_direction_basis[0]` point toward increasing axial
  index, i.e. *opposite* to the propagation of the light.
- The voxel lists from Siddon's algorithm (`ray_vol_colli_indices`) run from
  the objective side to the condenser side.

Nothing else in the geometry fixes this choice. The native focal plane is at the
volume center, so the ray geometry is symmetric in depth. When comparing with a
real specimen, or loading a volume from elsewhere, make sure slice 0 is the one
nearest the objective. Flipping a volume along the axial direction leaves
retardance and azimuth images unchanged, but changes intensity images wherever
the optic axis twists with depth.

## Order of Jones-matrix multiplication

For each ray the cumulative matrix is

```
J_ray = J_0 @ J_1 @ ... @ J_n
```

with `J_0` the voxel nearest the objective. The rightmost factor is the
condenser-side voxel, which the light crosses first, so this is the physical
order for light travelling toward the objective. The field at the camera is

```
E_out = analyzer @ J_ray @ polarizer @ E_in
```

## Transverse basis of the Jones vectors

`calc_ray_direction` returns `[ray, ray_perp1, ray_perp2]`. They are obtained by
rotating the lab axes `[axial, y, x]` (with the last axis negated) about
`ray x axial`, so for an on-axis ray `ray_perp1 = +y` and `ray_perp2 = -x`.
They satisfy `ray_perp1 x ray_perp2 = -ray`, so **(`ray_perp1`, `ray_perp2`) is
a right-handed transverse frame for the direction in which light propagates.**

The per-voxel Jones matrices are built with component 0 along `ray_perp2` and
component 1 along `ray_perp1`: the slow-axis angle is
`atan2(a . ray_perp1, a . ray_perp2)`. That frame is the mirror image
(left-handed). This does not affect linear retarders, whose matrix
`R(theta) diag(e^{i delta/2}, e^{-i delta/2}) R(-theta)` is valid in any
orthonormal frame. It does matter for elements with a circular handedness.

## Polarizer and analyzer matrices

`optical_info["polarizer"]`, `optical_info["analyzer"]`, and the LC-PolScope
compensator from `JonesMatrixGenerators` are written in the right-handed frame
(`ray_perp1`, `ray_perp2`). `intensity_images` converts them to the
(`ray_perp2`, `ray_perp1`) frame of the material matrices by swapping the two
basis vectors (`np.flip(analyzer, (0, 1))` and `np.flip(pol_hor)`). Since
`|S A S J S p| = |A (S J S) p|` for the swap `S`, the result equals the
standard calculation in the right-handed frame. Do not remove these flips
unless you also change the frame of the material matrices.

`BirefringentRaytraceLFM.apply_polarizers` does **not** apply this swap and is
currently unused.

## Retardance and azimuth

- Per voxel: half-retardance `pi * |dn| * (1 - (a . ray)^2) * ell / wavelength`
  (PyTorch backend) or full retardance (NumPy backend). The slow axis is the
  projection of the optic axis `a` onto the transverse plane, rotated by 90
  degrees when `dn < 0`.
- Per ray: retardance `2 * acos(Re J_00)` and azimuth
  `0.5 * atan2(Im J_00, Im J_01) - pi/4 (mod pi)`, followed by
  `transpose_and_flip` of the assembled images.
