import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0, j1
import mayavi.mlab as mlab
import sys
sys.path.append('Users/ngc/Desktop')
from numba import jit
import my_math_functions as mtm
import mayavi.mlab as mlab
from scipy.signal import argrelextrema
from skimage.feature import peak_local_max
import scipy as sp

plt.ion()


def mask_geometry (axial_length, NA, wavelength, EFL, refr_index, \
                    pictures = 'n'):
    # i'm using the correct refractive index for a water immersion objective
    # McCutchen calcs
    focal = EFL * refr_index
    a = wavelength / axial_length / refr_index # ok from theory
    R_ext = NA * EFL # ok from theory

    X = np.arcsin(R_ext / EFL / refr_index)
    # also X = np.arcsin(R_ext / focal)
    cosY = a + np.cos(X)
    Y = np.arccos(cosY)
    R_int = np.sin(Y) * focal

    print   ('\nRequired length = ', axial_length, 'um;\n'\
            '\nR_ext:', round(R_ext, 2), 'um' \
            ' <--->  R_int:', round(R_int, 2), 'um' \
            '\nDelta R = ', round(R_ext-R_int, 2), 'um')

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

    return R_ext, R_int

def deconvolve(stack,
               psf, 
               iterations, 
               R_factor):
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
                   EFL,
                   NA,
                   refr_index):
    x, y, z = np.copy(spatial_sampling_xy),\
              np.copy(spatial_sampling_xy),\
              np.copy(spatial_sampling_z)
    focal = EFL * refr_index
    extent_xy = len(spatial_sampling_xy)
    extent_z = len(spatial_sampling_z)
    step = np.sqrt((x[1] - x[0])**2 + (y[1]-y[0])**2 + (z[1]-z[0])**2)
    shell_thickness = step * np.sqrt(2)
    theta = np.arcsin((NA/refr_index))
    pupil = np.zeros((extent_xy, extent_xy, extent_z))
    for i in range(extent_xy):
        for j in range(extent_xy):
            for k in range(extent_z):
                r = np.sqrt(x[i]**2+y[j]**2+z[k]**2)
                if(r < (focal + (shell_thickness / 2)) and 
                   r > (focal - (shell_thickness / 2))):
                    pupil[i, j, k] = 1
                if(np.abs(np.arccos(z[k]/r)) > theta):
                    pupil[i, j, k] = 0
    return pupil

def lattice_mask(pupil_sampling_xy,
                 sample_sampling_xy, 
                 annulus,
                 psf_bessel,
                 wavelength,
                 EFL):
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
    reciprocal_seed = wavelength * EFL / seed
    stripes_3D = np.zeros((annulus.shape))
    stripes_2D = np.zeros((annulus.shape[0], annulus.shape[1]))
    stripes_2D = np.zeros((annulus.shape[0], annulus.shape[1]))
    step = np.abs(pupil_sampling_xy[1] - pupil_sampling_xy[0])
    for i in range (len(sample_sampling_xy)):
        for j in range (0, 20):
            if (np.abs(np.abs(pupil_sampling_xy[i]) - j*reciprocal_seed)\
                    <= step/2.):
                stripes_2D[:, i] = 1
    for i in range(annulus.shape[2]):
        stripes_3D[:, :, i] = stripes_2D[:, :]
    return annulus * stripes_3D, stripes_2D

#@jit(nopython=True, parallel=True)
def annular_pupil(sampling_xy,
                  length,
                  wavelength,
                  focal,
                  NA,
                  refr_index,
                  full_pupil):
    Re, Ri = mask_geometry(length,
                           NA,
                           wavelength,
                           focal,
                           refr_index)
    annulus = np.zeros((len(sampling_xy), len(sampling_xy)))
    x, y = np.meshgrid(sampling_xy, sampling_xy)
    annulus[x**2 + y**2 < Re**2] = 1
    annulus[x**2 + y**2 < Ri**2] = 0
    cyl = np.zeros((full_pupil.shape))
    for i in range(full_pupil.shape[2]):
        cyl[:, :, i] = annulus[:, :]
    return full_pupil * cyl
