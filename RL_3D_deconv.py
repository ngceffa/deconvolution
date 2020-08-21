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

plt.ion()

def mask_geometry (axial_length, NA, wavelength, focal, refr_index, \
                    pictures = 'n'):
    """It gives the geometry of an annular mask, given some parameters:
            - axial_lenllllgth = half the distance between 
                excitation zeros;
            -
            -
    """

    # i'm using the correct refractive index for a water immersion objective
    # McCutchen calcs
    a = wavelength / axial_length / refr_index
    R_ext = NA * focal / refr_index  # correct ?

    X = np.arcsin(R_ext / focal * refr_index)
    cosY = a + np.cos(X)
    Y = np.arccos(cosY)
    R_int = np.sin(Y)*focal/refr_index

    print   ('\nRequired length = ', axial_length, 'um;\n'\
            '\nR_ext:', round(R_ext, 2), 'um' \
            ' <--->  R_int:', round(R_int, 2), 'um' \
            '\nDelta R = ', round(R_ext-R_int, 2), 'um')

    # Inde calcs
    length = wavelength * focal**2 / (R_ext-R_int) \
                            / (R_ext - ((R_ext-R_int)/2)) / refr_index
    print ('\nDouble check with Indebeteouw:', round(length, 2), 'um')

    # Lateral profile = thickness
    r = np.linspace(-1., 13, 444)
    profile = j1(np.pi * R_ext * r / wavelength / focal * refr_index)/r - \
                j1(np.pi * R_int * r / wavelength / focal * refr_index)/r

    if (pictures == 'y'):
        plt.figure('profile')
        plt.fill_between(r, 0, profile**2/np.amax(profile**2),\
                         facecolor = 'deepskyblue')
        plt.xlabel(r'$ r \; [\mu m]$')
        plt.ylabel(r'$Normalized intensity \; [a.u.]$')
        plt.grid()
        plt.show()

    return R_ext, R_int, X, Y
#@jit(nopython=True, parallel=True)
def deconvolve(stack, 
               psf, 
               iterations, 
               R_factor, 
               space_sampling, 
               pupil_sampling):
    o = np.copy(stack).astype(complex)
    X, Y, Z = o.shape
    # lucy-richardson in the for loop
    for k in range (iterations):
        step_0 = stack/(mtm.IFT3(mtm.FT3(o)*mtm.FT3(psf)))
        step_1 = mtm.IFT3(mtm.FT3(step_0)*np.conj(mtm.FT3(psf)))
        o *= step_1
        print(k)
    return np.real(o)

@jit(nopython=True, parallel=True)
def circular_pupil(spatial_sampling_xy, 
                   spatial_sampling_z, 
                   focal, 
                   NA, 
                   refr_index):
    """
    Parameters
    ----------
    spatial_sampling : TYPE
        DESCRIPTION.
    focal : TYPE
        DESCRIPTION.
    NA : TYPE
        DESCRIPTION.
    ref_index : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    """
    x, y, z = np.copy(spatial_sampling_xy),\
              np.copy(spatial_sampling_xy),\
              np.copy(spatial_sampling_z)
    extent_xy = len(spatial_sampling_xy)
    extent_z = len(spatial_sampling_z)
    step = np.abs(x[1]-x[0])
    shell_thickness = step * np.sqrt(2)
    theta = np.arcsin((NA/refr_index))
    pupil, inner = np.zeros((extent_xy, extent_xy, extent_z)),\
                   np.zeros((extent_xy, extent_xy, extent_z))
    focal = focal / refr_index
    for i in range(extent_xy):
        for j in range(extent_xy):
            for k in range(extent_z):
                r = np.sqrt(x[i]**2+y[j]**2+z[k]**2)
                if(r < focal):
                    pupil[i, j, k] = 1
                if(r<(focal - shell_thickness)):
                    inner[i, j, k] = 1
                pupil[i, j, k] -= inner[i, j, k]
                if(np.abs(np.arccos(z[k]/r)) > theta):
                    pupil[i, j, k] = 0
    return pupil

@jit(nopython=True, parallel=True)
def annular_pupil(spatial_sampling_xy, 
                  spatial_sampling_z, 
                  focal, 
                  NA, 
                  refr_index, 
                  theta, 
                  phi, 
                  shell):
    """
    Parameters
    ----------
    spatial_sampling : TYPE
        DESCRIPTION.
    focal : TYPE
        DESCRIPTION.
    NA : TYPE
        DESCRIPTION.
    ref_index : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    """
    x, y, z = np.copy(spatial_sampling_xy),\
              np.copy(spatial_sampling_xy),\
              np.copy(spatial_sampling_z)
    extent_xy = len(spatial_sampling_xy)
    extent_z = len(spatial_sampling_z)
    # use the larger dimension for the tickness cell, like np.argmax(...)
    #step = np.abs(z[1]-z[0])
    shell_thickness = shell * (np.sqrt(2)*3) # why 4???
    #theta = np.arcsin((NA/refr_index))
    pupil, inner = np.zeros((extent_xy, extent_xy, extent_z)),\
                   np.zeros((extent_xy, extent_xy, extent_z))
    focal = focal / refr_index
    for i in range(extent_xy):
        for j in range(extent_xy):
            for k in range(extent_z):
                r = np.sqrt(x[i]**2+y[j]**2+z[k]**2)
                if(r < focal):
                     pupil[i, j, k] = 1
                if(r<(focal - shell_thickness)):
                    inner[i, j, k] = 1
                pupil[i, j, k] -= inner[i, j, k]
                if(np.abs(np.arccos(z[k]/r)) > theta):
                    pupil[i, j, k] = 0
                if(np.abs(np.arccos(z[k]/r)) < phi):
                    pupil[i, j, k] = 0
    return pupil

@jit(nopython=True, parallel=True)
def annular_pupil_2(spatial_sampling_xy, 
                  spatial_sampling_z, 
                  Re,
                  Ri,
                  gaus_pupil):
    x, y, z = np.copy(spatial_sampling_xy),\
              np.copy(spatial_sampling_xy),\
              np.copy(spatial_sampling_z)
    extent_xy = len(spatial_sampling_xy)
    extent_z = len(spatial_sampling_z)
    mask_2D = np.zeros((extent_xy, extent_xy))
    for i in range(mask_2D.shape[0]):
        for j in range(mask_2D.shape[1]):
            if np.sqrt(x[i]**2+y[j]**2) < Re:
                mask_2D[i,j] = 1
    if Re-Ri < np.abs(x[0]-x[1])*2:
        Ri = Re-(np.abs(x[0]-x[1])*2)
    for i in range(mask_2D.shape[0]):
        for j in range(mask_2D.shape[1]):
            if np.sqrt(x[i]**2+y[j]**2) < Ri:
                mask_2D[i,j] = 0
    pupil = np.zeros((extent_xy, extent_xy, extent_z))
    for k in range(len(z)):
        pupil[:, :, k] = mask_2D[:, :]
    bessel_pupil = gaus_pupil * pupil
    return bessel_pupil, mask_2D

#@jit(nopython=True, parallel=True)
def lattice_mask(pupil_sampling_xy,
                 pupil_sampling_z,
                 sample_sampling_xy, 
                 sample_sampling_z, 
                 annulus, 
                 psf_bessel, 
                 wavelength, 
                 f_obj ):
    first_min = argrelextrema(psf_bessel[int(psf_bessel.shape[0]/2),\
                                         int(psf_bessel.shape[1]/2+1):,\
                                         int(psf_bessel.shape[2]/2)],
                                         np.less)
    first_min = sample_sampling_xy[
                    int(first_min[0][0]+len(sample_sampling_xy)/2)]
    first_max = argrelextrema(psf_bessel[int(psf_bessel.shape[0]/2),\
                                         int(psf_bessel.shape[1]/2+1):,\
                                         int(psf_bessel.shape[2]/2)],
                                         np.greater)
    first_max = sample_sampling_xy[int(
                first_max[0][0]+len(sample_sampling_xy)/2)]
    print ('\nfirst min:', first_min, 'um')
    print ('\nfirst max: ', first_max, 'um')
    print ('\n\'Total\' central peak: ', first_min * 2., 'um')
    
    seed = first_max * 2
    reciprocal_seed = wavelength * f_obj / seed
    stripes_3D = np.zeros((annulus.shape))
    stripes_2D = np.zeros((annulus.shape[0], annulus.shape[1]))
    stripes_2D = np.zeros((annulus.shape[0], annulus.shape[1]))
    step = np.abs(pupil_sampling_xy[1] - pupil_sampling_xy[0])
    for i in range (len(sample_sampling_xy)):
        for j in range (0, 20):
            if (np.abs(np.abs(pupil_sampling_xy[i]) - j*reciprocal_seed)\
                    <= step/2.):
                stripes_2D[:, i] = 1
    for i in range(stripes_3D.shape[2]):
        stripes_3D[:, :, i] = stripes_2D[:, :]
    lattice_mask = annulus * stripes_3D
    return lattice_mask, stripes_2D

exc_wavelength = .488  # um
exc_obj_f = 20000  # um
sampling = np.linspace(-40*10**3, 40*10**3, 500)
sampling_step = np.abs(sampling[1] - sampling[0])
# sampling_sample_xy = np.linspace(-1 / sampling_step / 2 \
#                               * exc_wavelength * exc_obj_f,
#                               1  / sampling_step / 2. \
#                               * exc_wavelength * exc_obj_f,
#                               len(sampling))
# sampling_sample_z = np.linspace(-1 / sampling_step / 2 \
#                               * exc_wavelength * exc_obj_f,
#                               1  / sampling_step / 2. \
#                               * exc_wavelength * exc_obj_f,
#                               len(sampling)

# From the image shape, times 4 (use scipy zoom after)
sampling_sample_xy = np.linspace(-300*.195/2,
                                 300*.195/2,
                                 600)
sampling_sample_z = np.linspace(-100.*.75/2,
                                 100*.75/2,
                                 300)

sampling_step_xy = np.abs(sampling_sample_xy[1] - sampling_sample_xy[0])
sampling_step_z = np.abs(sampling_sample_z[1] - sampling_sample_z[0])
pupil_sampling_xy = np.linspace(-1 / sampling_step_xy / 2 \
                              * exc_wavelength * exc_obj_f,
                              1  / sampling_step_xy / 2. \
                              * exc_wavelength * exc_obj_f,
                              len(sampling_sample_xy))
pupil_sampling_z = np.linspace(-1 / sampling_step_z / 2 \
                              * exc_wavelength * exc_obj_f,
                              1  / sampling_step_z / 2. \
                              * exc_wavelength * exc_obj_f,
                              len(sampling_sample_z))
pupil_sampling_step_xy = np.abs(pupil_sampling_xy[1] - pupil_sampling_xy[0])
pupil_sampling_step_z = np.abs(pupil_sampling_z[1] - pupil_sampling_z[0])  
  
pupil_sampling_xy_det = np.linspace(-1 / sampling_step_xy / 2 \
                              * .54 * 9000,
                              1  / sampling_step_xy / 2. \
                              * .54 * 9000,
                              len(sampling_sample_xy))
pupil_sampling_z_det = np.linspace(-1 / sampling_step_z / 2 \
                              * .54 * exc_obj_f,
                              1  / sampling_step_z / 2. \
                              * .54 * 9000,
                              len(sampling_sample_z))

gaus = circular_pupil(pupil_sampling_xy_det, 
                      pupil_sampling_z_det, 
                      9000, 
                      1, 
                      1.33)
psf_gaus = np.abs(mtm.FT3(gaus))**2
Re, Ri, theta, phi = mask_geometry(200., .25, .488, 20000, 1.33)

annulus = np.zeros((gaus.shape[0], gaus.shape[1]))
X, Y = np.meshgrid(pupil_sampling_xy, pupil_sampling_xy)
annulus[X**2+Y**2 <=Re**2] = 1.
annulus[X**2+Y**2 <=Ri**2] = 0.

KX, KY = np.meshgrid(sampling_sample_xy, sampling_sample_xy)

bessel = np.abs(mtm.FT2(annulus))

#------------------------------------------------------------------------------
first_min = argrelextrema(bessel[int(bessel.shape[0]/2),\
                                            int(bessel.shape[1]/2+1):],np.less)
first_min = sampling_sample_xy[int(first_min[0][0]+len(sampling_sample_xy)/2)]
first_max = argrelextrema(bessel[int(bessel.shape[0]/2),\
                    int(bessel.shape[1]/2+1):],\
                                                                    np.greater)
first_max = sampling_sample_xy[int(first_max[0][0]+len(sampling_sample_xy)/2-1)]
seed = first_max * 2
reciprocal_seed = .488 * 20000 / seed

stripes = np.zeros((annulus.shape))
stripes[:, int(stripes.shape[0]/2)] = 1.
for i in range (len(sampling_sample_xy)):
    for j in range (0, 20):
        if (np.abs(np.abs(pupil_sampling_xy[i]) - j*reciprocal_seed) \
                                            <= pupil_sampling_step_xy/2):
            stripes[:, i] = 1.
            
profile = np.abs(mtm.FT2(annulus*stripes))**2

scan = np.average(profile, axis = 1)

full_exc = np.zeros((psf_gaus.shape))
for i in range(full_exc.shape[2]):
    for j in range(full_exc.shape[1]):
        full_exc[:,j,i] = scan

psf_lattice_scanned = np.rot90(full_exc, 1, (1, 2))

resized_gaus = psf_gaus[:, 150:450,:]
resized_psf_lattice = psf_lattice_scanned[:,:,150:450]

psf_total_lattice_exc = resized_gaus * resized_psf_lattice
psf_total_lattice_exc = psf_total_lattice_exc/np.amax(psf_total_lattice_exc)

psf_cut = psf_total_lattice_exc[150:450,:,100:200]
psf_zoomed = sp.ndimage.zoom(psf_cut, .5)
 
shell = Re - Ri
print(shell)
gaus_exc = circular_pupil(pupil_sampling_xy, 
                      pupil_sampling_z, 
                      20000, 
                      .3, 
                      1.33)
if shell < np.sqrt(2*pupil_sampling_step_xy**2 +pupil_sampling_step_z**2) :
    shell = np.sqrt(2*pupil_sampling_step_xy**2 +pupil_sampling_step_z**2)
bessel= annular_pupil(pupil_sampling_xy, 
                        pupil_sampling_z, 
                        20000, 
                        .3,
                        1.33,
                        theta, phi, shell)


psf_gaus = np.abs(mtm.FT3(gaus))**2
psf_bessel = np.abs(mtm.FT3(bessel))**2
lattice, mask = lattice_mask(pupil_sampling_xy,
                  pupil_sampling_z,
                  sampling_sample_xy, 
                  sampling_sample_z, 
                  bessel, 
                  psf_bessel, 
                  .488, 
                  20000)
psf_lattice = np.abs(mtm.FT3(lattice))**2
# psf_lattice_scanned = np.zeros((psf_lattice.shape))
# for i in range(psf_lattice.shape[2]):
#     psf_lattice_scanned[:,i,:] = np.average(psf_lattice, axis = 1)
# psf_lattice_scanned = np.rot90(psf_lattice_scanned, 1, (1, 2))
# #psf_total_lattice_exc = psf_gaus * psf_lattice_scanned




















