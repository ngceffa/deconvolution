import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from scipy.special import j0, j1
import mayavi.mlab as mlab
import sys
import time
sys.path.append('Users/ngc/Desktop')
from numba import jit
import my_math_functions as mtm
import plotly.graph_objects as go
import mayavi.mlab as mlab
from scipy.signal import argrelextrema
from skimage.feature import peak_local_max
import scipy as sp
import optics_funcions as of
from importlib import reload
reload(mtm)
reload(of)


plt.ion()

# objectives parameters
NA_det = 1.
refr_index = 1.33
wavelength_det = .54  # um
focal_det = 9 * 10**3  # um
wavelength_det = wavelength_det/ refr_index

NA_ex = .24
wavelength = .488  # um
focal_ex = 20 * 10**3  # um
wavelength =  wavelength/refr_index # not in vacuum!
excitation_axial_length = 300  # um, this is also the desired FOV dimension

# image paratmeters
image_pixels = 500 # a.
images = 500 # b. these parameters can ensure a square image
pixel = .195  # um
z_step = .15 # um
zoom_factor = 1  # to better sample the spaces. Go back in the final step to
                 # match the image dimensions for the deconvolution algorithm
range_factor = 1  # explore a larger range during the simulation

# image grid
xy_sampling = np.linspace(-image_pixels * pixel / 2 * range_factor,
                          image_pixels * pixel / 2 * range_factor,
                          image_pixels * zoom_factor)
z_sampling = np.linspace(-images * z_step / 2 * range_factor,
                          images * z_step / 2 * range_factor,
                          images * zoom_factor)
xy_step = np.abs(xy_sampling[0]-xy_sampling[1])
z_step = np.abs(z_sampling[0]-z_sampling[1])

# pupil grid (derived by the image grid)
xy_sampling_pupil_det = np.linspace(-1 / xy_step / 2 * wavelength * focal_det,
                                1 / xy_step / 2 * wavelength * focal_det,
                                len(xy_sampling))
z_sampling_pupil_det = np.linspace(-1 / z_step / 2 * wavelength * focal_det,
                                1 / z_step / 2 * wavelength * focal_det,
                                len(z_sampling))
xy_sampling_pupil_ex = np.linspace(-1 / xy_step / 2 * wavelength * focal_ex,
                                1 / xy_step / 2 * wavelength * focal_ex,
                                len(xy_sampling))
z_sampling_pupil_ex = np.linspace(-1 / z_step / 2 * wavelength * focal_ex,
                                1 / z_step / 2 * wavelength * focal_ex,
                                len(z_sampling))

# probably to be cancelled
# Re, Ri, theta, phi = of.mask_geometry(200, .3, wavelength, 20000, refr_index)
# annulus = np.zeros((image_pixels, image_pixels))
# x, y = np.linspace(-1 / xy_step / 2 * wavelength * 20000,
#                                 1 / xy_step / 2 * wavelength * 20000, 
#                                 len(xy_sampling)), \
#         np.linspace(-1 / xy_step / 2 * wavelength * 20000,
#                                 1 / xy_step / 2 * wavelength * 20000, 
#                                 len(xy_sampling))
# x, y = np.meshgrid(x,y)
# annulus[x**2 + y**2 < Re**2] = 1
# annulus[x**2 + y**2 < Ri**2] = 0
# shell = Re-Ri

# if shell < np.sqrt(xy_step**2 + xy_step**2  + z_step**2):
#     shell = np.sqrt(xy_step**2  + xy_step**2 + z_step**2)

pupil_det = of.circular_pupil(xy_sampling_pupil_det,
                          z_sampling_pupil_det,
                          focal_det,
                          NA_det,
                          refr_index)
psf_det = np.abs(mtm.FT3(pupil_det))**2
val_max, val_min = psf_det.max(), psf_det.min()
psf_det = (psf_det - val_min)/(val_max - val_min)

pupil_ex = of.circular_pupil(xy_sampling_pupil_ex,
                          z_sampling_pupil_ex,
                          focal_ex,
                          NA_ex,
                          refr_index)
annular = of.annular_pupil(xy_sampling_pupil_ex,
                             z_sampling_pupil_ex,
                             excitation_axial_length,
                             wavelength,
                             focal_ex,
                             NA_ex,
                             refr_index,
                             pupil_ex)
psf_bessel = np.abs(mtm.FT3(annular))**2
val_max, val_min = psf_bessel.max(), psf_bessel.min()
psf_bessel = (psf_bessel - val_min)/(val_max - val_min)

lattice, mask = of.lattice_mask(xy_sampling_pupil_ex,
                                z_sampling_pupil_ex,
                                xy_sampling,
                                z_sampling,
                                annular,
                                psf_bessel,
                                wavelength,
                                focal_ex)
psf_lattice = np.abs(mtm.FT3(lattice))**2
val_max, val_min = psf_lattice.max(), psf_lattice.min()
psf_lattice = (psf_lattice - val_min)/(val_max - val_min)

scan = np.average(psf_lattice, axis = 1)

full_exc = np.zeros((psf_lattice.shape))
for i in range(full_exc.shape[1]):
    full_exc[:, :, i] = scan
psf_lattice_scanned = np.rot90(full_exc, 1, (0, 2))

psf_total = psf_lattice_scanned *  psf_det
val_max, val_min = psf_total.max(), psf_total.min()
psf_total_norm = (psf_total - val_min)/(val_max - val_min)

psf_for_dec = psf_total_norm[125:375, 125:375, :]
psf_for_dec = sp.ndimage.zoom(psf_for_dec, [1, 1, 1/2])
psf_for_dec_final = psf_for_dec[:,:,]

data = tiff.imread('cell_6_substack.tif')
organized = np.zeros((data.shape[1], data.shape[2], data.shape[0]))
for i in range(data.shape[0]):
    organized[:,:,i] = data[i,:,:]

# psf_zoomed = sp.ndimage.zoom(psf_total_norm, [1, 1, 1/4])    
# psf_for_dec = np.zeros((organized.shape))
# psf_for_dec[:,:,:] = psf_zoomed[int(
# psf_zoomed.shape[0]/2-organized.shape[0]/2):
#                         int(psf_zoomed.shape[0]/2+organized.shape[0]/2),
#                          int(psf_zoomed.shape[1]/2-organized.shape[1]/2):
#                             int(psf_zoomed.shape[1]/2+organized.shape[1]/2),
#                                 :]

result = of.deconvolve(organized, psf_for_dec_final, 5, 1)

val_max, val_min = result.max(), result.min()
result = (result - val_min)/(val_max - val_min)
saving = np.zeros((data.shape[0], data.shape[1], data.shape[2]),
                  dtype = np.uint16)

# result = of.deconvolve(organized, psf_for_dec, 70, 1) 
for i in range(data.shape[0]):
    saving[i,:,:] = result[:,:,i].astype(np.uint16)

tiff.imsave('cel_6_sub_dec.tif', saving)





# for i in range(pupil.shape[2]):
#     pupil[:,:,i] = pupil[:,:,i] * annulus


# bessel_pupil = of.annular_pupil(xy_sampling_pupil,
#                                 z_sampling_pupil,
#                                 20000, 
#                                 .3, 
#                                 refr_index, 
#                                 theta, 
#                                 phi, 
#                                 shell)
# psf_bessel = np.abs(mtm.FT3(pupil))**2
# val_max, val_min = psf_bessel.max(), psf_bessel.min()
# psf_bessel = (psf_bessel - val_min)/(val_max - val_min)

# lattice_mask, stripes_mask = of.lattice_mask(xy_sampling_pupil,
#                                              z_sampling_pupil,
#                                              xy_sampling, 
#                                              z_sampling, 
#                                              pupil,
#                                              psf_bessel,
#                                              .488, 
#                                              20000)
# pupil = of.circular_pupil(xy_sampling_pupil,
#                           z_sampling_pupil, 
#                           focal=0000, 
#                           NA=.3, 
#                           refr_index=1.33)
# detection_psf = np.abs(mtm.FT3(pupil))**2
# val_max, val_min = detection_psf.max(), detection_psf.min()
# detection_psf = (detection_psf - val_min)/(val_max - val_min)

# #psf_zoomed = sp.ndimage.zoom(detection_psf, 1/zoom_factor)

# excitation_profile = tiff.imread('scanned_profile.tif')
# excitation_profile_1D = np.average(excitation_profile, axis = 0)

# # correct z sampling
# samp = pixel / z_step
# excitation_profile_1D = sp.ndimage.zoom(excitation_profile_1D, samp)

# maxarg = np.argmax(excitation_profile_1D)
# centered_profile = np.zeros((detection_psf.shape[2]))
# centered_profile[:] = excitation_profile_1D[\
#                       int(maxarg-centered_profile.shape[0]/2):\
#                          int(maxarg+centered_profile.shape[0]/2)]

# val_max, val_min = centered_profile.max(), centered_profile.min()
# centered_profile = (centered_profile - val_min)/(val_max - val_min)

# excitation_psf = np.zeros((detection_psf.shape))
# for i in range(detection_psf.shape[0]):
#     for j in range(detection_psf.shape[1]):
#         excitation_psf[i, j, :] = centered_profile[:]

# total_psf = excitation_psf * detection_psf

# data = tiff.imread('double_a.tif')
# organized = np.zeros((data.shape[1], data.shape[2], data.shape[0]))
# for i in range(data.shape[0]):
#     organized[:,:,i] = data[i,:,:]

# result = of.deconvolve(organized, total_psf, 10, 1) 

# saving = np.zeros((data.shape[0], data.shape[1], data.shape[2]), 
#                   dtype = np.uint16)
# for i in range(data.shape[0]):
#     saving[i,:,:] = result[:,:,i].astype(np.uint16)
# tiff.imsave('double_z_dev.tif', saving)



















