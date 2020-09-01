import numpy as np
import optics_functions as of
import matplotlib.pyplot as plt
import my_math_functions as mtm
from importlib import reload
reload(of)
reload(mtm)


plt.ion()
# |> objectives' parameters
NA_det = 1.
refr_index = 1.33
wavelength_det = .54 # um
EFL_det = 9 * 10**3  # um
NA_ex = .29
wavelength = .488  # um
EFL_ex = 20 * 10**3  # um
excitation_axial_length = 150  # um
# image paratmeters
image_pixels = 500
images = 500
pixel = .19  # um
z_step = .15 # um
zoom_factor = 1  # to better sample the spaces. Go back in the final step to
                 # match the image dimensions for the deconvolution algorithm
range_factor = 1  # explore a larger range during the simulation
xy_sampling = np.linspace(-image_pixels * pixel / 2 * range_factor,
                          image_pixels * pixel / 2 * range_factor,
                          image_pixels * zoom_factor)
z_sampling = np.linspace(-images * z_step / 2 * range_factor,
                         images * z_step / 2 * range_factor,
                         images * zoom_factor)
xy_step = np.abs(xy_sampling[0]-xy_sampling[1])
z_step = np.abs(z_sampling[0]-z_sampling[1])
# pupil grid (derived by the image grid)
xy_sampling_pupil_det = np.linspace(-1 / xy_step / 2 * wavelength * EFL_det,
                                    1 / xy_step / 2 * wavelength * EFL_det,
                                    len(xy_sampling))
z_sampling_pupil_det = np.linspace(-1 / z_step / 2 * wavelength * EFL_det,
                                   1 / z_step / 2 * wavelength * EFL_det,
                                   len(z_sampling))
xy_sampling_pupil_ex = np.linspace(-1 / xy_step / 2 * wavelength * EFL_ex,
                                   1 / xy_step / 2 * wavelength * EFL_ex,
                                   len(xy_sampling))
z_sampling_pupil_ex = np.linspace(-1 / z_step / 2 * wavelength * EFL_ex,
                                  1 / z_step / 2 * wavelength * EFL_ex,
                                  len(z_sampling))
# -><- -><-
pupil_det = of.circular_pupil(xy_sampling_pupil_det,
                              z_sampling_pupil_det,
                              EFL_det,
                              NA_det,
                              refr_index)
psf_det = np.abs(mtm.FT3(pupil_det))**2
val_max, val_min = psf_det.max(), psf_det.min()
psf_det_norm = (psf_det - val_min)/(val_max - val_min)

pupil_ex = of.circular_pupil(xy_sampling_pupil_ex,
                             z_sampling_pupil_ex,
                             EFL_ex,
                             NA_ex,
                             refr_index)
annular = of.annular_pupil(xy_sampling_pupil_ex,
                           excitation_axial_length,
                           wavelength,
                           EFL_ex,
                           NA_ex,
                           refr_index,
                           pupil_ex)
psf_bessel = np.abs(mtm.FT3(annular))**2
val_max, val_min = psf_bessel.max(), psf_bessel.min()
psf_bessel_norm = (psf_bessel - val_min)/(val_max - val_min)

lattice, mask = of.lattice_mask(xy_sampling_pupil_ex,
                                xy_sampling,
                                annular,
                                psf_bessel,
                                wavelength,
                                EFL_ex)
psf_lattice = np.abs(mtm.FT3(lattice))**2
val_max, val_min = psf_lattice.max(), psf_lattice.min()
psf_lattice_norm = (psf_lattice - val_min)/(val_max - val_min)

