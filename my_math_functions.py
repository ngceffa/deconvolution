import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fft2, ifft2, fftn, ifftn
from scipy.fftpack import fftshift, ifftshift
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize as opt
from skimage.feature import peak_local_max


def sampleVideo (matrix, frames, pause = .05):
    """Show a video from a 3Dd matrix.
        - matrix = [x, y, t]
        - frames = number of frames to show
        - pause = interval between frames
    """
    im = plt.imshow(np.abs(matrix[:,:,0])/np.amax(np.abs(matrix)),\
                        cmap = 'gray', interpolation = 'none')
    for i in range (frames):
        im.set_data(np.abs(matrix[:,:,i])/np.amax(np.abs(matrix)))
        plt.pause(0.5)
    plt.show()
    return None

from inspect import currentframe
def debug_hint():
    """It prints to screen the position at which it is called.
    """
    frameinfo = currentframe()
    print( '\n{line: ', frameinfo.f_back.f_lineno,'}\n')
    return None

def FT(f, ax = 0):
    return fftshift(fft(ifftshift(f), axis = ax))

def FT2(f):
    return fftshift(fft2(ifftshift(f)))

def FT3(f):
    return fftshift(fftn(ifftshift(f)))

def IFT(F, ax = 0):
    return ifftshift(ifft(fftshift(F), axis = ax))

def IFT2(F):
    return ifftshift(ifft2(fftshift(F)))

def IFT3(F):
    return ifftshift(ifftn(fftshift(F)))

def gaus(x, p=[0, 1]):
    """Gaussian function
        - x = domain;
        - p[0] = center;
        - p[1] = width.
    """
                      
    return np.exp(-np.pi*(x-p[0])**2/(p[1]**2))
      
def gaus_2d(X, Y, p=[1, 0,  0., .5, .5]):
    """ 2D Gaussian
        - X, Y = axes
        - p[0] = amplitude
        - p[1] = center X
        - p[3] = sigma X
        - p[2] = center Y
        - p[4] = sigma Y
    """
    X, Y = np.meshgrid(X, Y)
    return p[0] * \
            np.exp(-(np.pi*(X-p[1])**2)/(np.abs(p[3])**2)-\
                                (np.pi*(Y-p[2])**2)/(np.abs(p[4])**2)) 

def gaus_2d_forFit(mesh_grid, p=[1, 0,  0., .5, .5, .1]):
    (X, Y) = mesh_grid
    return np.ravel((p[0] * np.exp(-(np.pi*(X-p[1])**2)/(p[3]**2)-\
                          (np.pi*(Y-p[2])**2)/(p[4]**2))+p[5]))

def double_gaus_2d_forFit(X, Y, p=[0, 1, 2, 3, 4, 5, 6, 7, 1, 1]):
    return (p[8]*np.exp(-(np.pi*(X-p[0])**2)/\
                                    (p[1]**2)-(np.pi*(Y-p[2])**2)/(p[3]**2))\
            +p[9]*np.exp(-(np.pi*(X-p[4])**2)/\
                                    (p[5]**2)-(np.pi*(Y-p[6])**2)/(p[7]**2)))\
            .ravel()

def step(x):
    """Step function:
        - x = domain.
    """
    return .5*(1+sp.sign(x))

def rect(x, p=[0.,1.]):
    """Rectangular function
                      - x = domain;
                      - p[0] = center;
                      - p[1] = width.
    """
    return step((x-p[0])/p[1]+.5)-step((x-p[0])/p[1]-.5)

def rect_2d(x, y, p=[0.,0.,1.,1.]):
    """Rectangular function
        - x, y = 2d grid domain;
        - p[0] = center_x;
        - p[1] = center_y;
        - p[2] = width_x;
        - p[3] = width_y;
    """
    return (step((x[:]-p[0])/p[2]+.5)-step((x[:]-p[0])/p[2]-.5))*\
                        (step((y[:]-p[1])/p[3]+.5)-step((y[:]-p[1])/p[3]-.5))

def circ(x, y, center, D):
    """Circ function
        - x, y = 2d meshgrid, domain;
        - D = mask diamenter.
    """
    out = np.zeros((x.shape[0], y.shape[1]))
    for i in range (out.shape[0]):
        for j in range (out.shape[1]):
            if((x[i,j]-center)**2+(y[i,j]-center)**2)< (D/2.)**2 :
                out[i,j] = 1.
    return out

def comb(x, x0, b):
    result = np.zeros((len(x)))
    threshold = np.abs(x[1]-x[0])
    for i in range(len(result)):
        for j in range(len(result)):
            if(np.abs((x[i]-x0)%(j*b)) < threshold):
                result[i] = 1.
    return result

def comb_2d(x, y, x0, y0, a, b):
    M, N = len(x), len(y)
    X, Y = np.meshgrid(x,y)
    matrix, result = np.zeros((M ,N)), np.zeros((M ,N))
    matrix = (np.cos(2*np.pi*(X-x0)/a)+1) * (np.cos(2*np.pi*(Y-y0)/b)+1)
    peaks = peak_local_max(matrix)
    for i in range (peaks.shape[0]):
        result[peaks[i][0], peaks[i][1]] = 1.
    return result, matrix

def crossCorr (f, g):
    """Cross-correlation function, with padding. For flows: it will
        highligh a flow of particle f --> g;
                        - f = first signal;\
                        - g = second signal.
        """
    
    N = len(f)
    one, two = np.pad(np.copy(f),\
                    (N/2),\
                            mode = 'constant', constant_values=(0)),\
               np.pad(np.copy(g),\
                    (N/2),\
                            mode = 'constant', constant_values=(0))
    F, G = FT(one), FT(two)
    
    cross = np.real(ifft(ifftshift(F)*np.conj(ifftshift(G))))[:N]
    
    for i in range (len(cross)):
        cross[i] = cross[i]/(N-i)
    
    return cross

def crossCorrFluct (f, g):
    """Cross-correlation function of the
        fluctuations of two signals, with padding, normalized.
        For flows: it will highligh a flow of particle f --> g;
                        - f = first signal;
                        - g = second signal.
        """
    
    N = len(f)
    mean_f, mean_g = np.mean(f), np.mean(g)
    one, two = np.pad(np.copy(f-mean_f),\
                    (N/2),\
                            mode = 'constant', constant_values=(0)),\
               np.pad(np.copy(g-mean_g),\
                    (N/2),\
                            mode = 'constant', constant_values=(0))
    F, G = fft(two), fft(one)
    
    cross = np.real(ifft(F*np.conj(G)))[:N]
    
    for i in range (len(cross)):
        cross[i] = cross[i]/(N-i)/mean_f/mean_g
    
    return cross

def spatial_Xcorr_2(matrix_1, matrix_2):
    """
    To be tested
    """
    M, N = matrix_1.shape[0], matrix_1.shape[1]
    
    one, two = np.pad(np.copy(matrix_1),\
                    ((int(M/2), int(M/2)),(int(N/2), int(N/2))),\
                            mode = 'constant', constant_values=(0,0)),\
               np.pad(np.copy(matrix_2),\
                    ((int(M/2), int(M/2)),(int(N/2), int(N/2))),\
                            mode = 'constant', constant_values=(0,0))
    ONE, TWO =   FT2(one), FT2(two)
    
    spatial_cross = ifftshift(ifft2(ifftshift(ONE) * np.conj(ifftshift(TWO))))

    return spatial_cross[int(M/2) :int(M/2)+matrix_1.shape[0],\
                        int(N/2) : int(N/2)+matrix_1.shape[1]]

def convolution_2D(matrix_1, matrix_2):
    """
    To be tested
    """
    M, N = matrix_1.shape[0], matrix_1.shape[1]
    
    one, two = np.pad(np.copy(matrix_1),\
                    ((int(M/2), int(M/2)),(int(N/2), int(N/2))),\
                            mode = 'constant', constant_values=(0,0)),\
               np.pad(np.copy(matrix_2),\
                    ((int(M/2), int(M/2)),(int(N/2), int(N/2))),\
                            mode = 'constant', constant_values=(0,0))
    ONE, TWO =   FT2(one), FT2(two)
    
    spatial_cross = ifftshift(ifft2(ifftshift(ONE) * (ifftshift(TWO))))

    return spatial_cross[int(M/2) :int(M/2)+matrix_1.shape[0],\
                        int(N/2) : int(N/2)+matrix_1.shape[1]]
def gausFit (vector, x_step = 1., plot = 'yes', title = 'titolo'):
    """Fitting with a gaussian:
                      - vector = what to fit;
                      - x_step (1.) = independent variable increment;
                      - plot ('yes') = show/hide a plot of the result;
                      - titel ('titolo') = name of the function.
                      - return = {[amplitude, center, sigma], iterations}
    """
    N = len(vector)
    x = np.arange(0., N, 1.)
    
    p0 = [0.,0.,0.] # A, x0, sigma
    p0[0] = np.amax(vector)
    p0[1] = np.argmax(vector)
    p0[2] = p0[0]/2.
    
    fit_func = lambda p,x: p[0]*np.exp(-np.pi*(x-p[1])**2/(p[2]**2))
    err_func = lambda p,x,y: fit_func(p,x)-y
    p1,cov,info1, info2, success = sp.optimize.leastsq(err_func, p0, \
                        args=(x, vector), \
                        full_output = 1)
    
    if (plot=='yes'):
        plt.figure('FITResults '+ title)
        plt.title('gausFit'+' '+title)
        plt.ylabel('Function'), plt.xlabel('x')
        
        plt.plot(x*x_step, vector,'ro',label = 'data')
        
        plt.plot(x*x_step,fit_func(p1, x),'b--',label = 'fit')
        
        plt.ylim (np.amin(vector)-.05, np.amax(vector)+.2)
        
        plt.annotate('Amplitude: '+str(p1[0])+'\n'\
                    +'Center: '+str(p1[1])+'\n'+
                    'Sigma:'+str(p1[2])+'\n',\
                    [p1[1],p1[0]], [1,p1[0]-.5])
        plt.grid(which = 'minor')
        plt.legend()
        plt.show()
        
    return p1, success
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def gausFit_2D (matrix, mesh, plot = 'yes', title = 'fit'):
    """Fitting with a gaussian:
                      - vector = what to fit;
                      - x_step (1.) = independent variable increment;
                      - plot ('yes') = show/hide a plot of the result;
                      - titel ('titolo') = name of the function.
                      - return = {[amplitude, center, sigma], iterations}
    """
    vector = np.ravel(matrix)
    print(vector.shape)
    (x, y) = mesh
    
    # Initial guesses
    p0 = [1., 2. , 2., 4., 4., .5] # Amplitude, x0, y0, sigmaX, sigmaY
    p0[0] = np.amax(vector)
    p0[2], p0[1] = np.unravel_index(np.argmax(matrix), matrix.shape)
    p0[5] = np.amin(vector)
    
    print(p0)
    
    print(type(p0))
    
    #p0 = tuple(p0)
    popt = 0

    popt, pcov = opt.curve_fit(gaus_2d_forFit, mesh, vector, p0 = p0)
    errs = np.sqrt(np.diag(pcov))
    
    amp, xc, yc, a, b, offset = popt

    if (plot=='yes'):
        after = gaus_2d(mesh, amp, xc, yc, a, b, offset)
        plt.figure('FIT Results '+ title)
        plt.title(title)
        plt.ylabel('x'), plt.xlabel('y')
        
        plt.imshow(matrix, label = 'data', cmap = 'gray', \
            extent = (np.amin(y), np.amax(y), np.amin(x), np.amax(x)))
        levels = (np.linspace(np.amin(after), np.amax(after), 6 ))
        plt.contour(after[::-1,:], label = 'fit', cmap = 'hot', alpha = .5,\
                    levels = levels,\
                    extent = (np.amin(y), np.amax(y), np.amin(x), np.amax(x)))
        
        plt.annotate(\
                r'$A\,exp(-\frac{\pi(x-x_{0})^{2}}{\sigma_{x}^{2}}\,-\,$'+\
                r'$\frac{\pi(y-y_{0})^{2}}{\sigma_{y}^{2}})$'+'\n'+
                r'$A$= '+str(round(popt[0],3))+\
                r'$\,\pm\,$'+str(round(errs[0],4))+'\n'+\
                r'$x_{0}$= '+str(round(popt[1],4))+\
                r'$\,\pm\,$'+str(round(errs[1],4))+'\n'+\
                r'$y_{0}$= '+str(round(popt[2],4))+\
                r'$\,\pm\,$'+str(round(errs[2],4))+'\n'+\
                r'$\sigma_{x}= $'+str(round(popt[3],3))+\
                r'$\,\pm\,$'+str(round(errs[3],4))+'\n'+\
                r'$\sigma_{y}= $'+str(round(popt[4],3))+\
                r'$\,\pm\,$'+str(round(errs[4],3))+'\n',\
                    [1,1], xytext = (np.amax(x)+.1, np.amax(y)-5))
        plt.grid(which = 'minor')
        plt.legend()

        plt.show()
        
    return popt, errs

def smoothing (vector, points = 3):
    """Smoothing macro.
                    - vector = input signal;
                    - points = smoothing range (odd).
    """
    if (int(points%2)==0):
        print( '\nOdd number of points needed. Smoothing NOT done.\n')
        return None
    else:
        smoothed = np.copy(vector)
        for i in range (points, len(vector)-points):
            smoothed[i] = 0.
            for j in range (int(-1*points/2+1),int(points/2+1)):
                smoothed[i] += vector[i+j]/float(points)
        return smoothed

def crossCorrCirc(f,g):
    """Cross-correlation function of the
        fluctuations of two signals, without padding, normalized.
        For flows: it will highligh a flow of particle f --> g;
                        - f = first signal;
                        - g = second signal.
        """
    
    N = len(f)
    
    F, G = fft(f), fft(g)
    
    cross = np.real(ifft(F*np.conj(G)))[:N]
    
    for i in range (len(cross)):
        cross[i] = cross[i]/(N-i)
    
    return cross

def quadratic_phase_signal(x, y, z, wavelenght):
    """ From Gaskill: ignoring the lens diameter and thickness
    (the lens is considered thin and paraxial approximantion makes it appear
    lage), this can be used to model field propagation.
    """
    return np.exp(np.pi*1j*(x**2 + y**2)/(wavelenght * z))

def exp_fit (vector, x_step = 1., plot = 'yes', title = 'titolo'):
    """Fitting with a gaussian:
                      - vector = what to fit;
                      - x_step (1.) = independent variable increment;
                      - plot ('yes') = show/hide a plot of the result;
                      - titel ('titolo') = name of the function.
                      - return = {[amplitude, center, sigma], iterations}
    """
    N = len(vector)
    x = np.arange(0., N, 1.)
    
    p0 = [0., 0., 0.] # A, x0, sigma
    p0[0] = vector[0] # amplitude
    p0[1] = 300#vector[int(len(vector)/2)] * x_step
    p0[2] = np.amin(vector) # end value?
    
    fit_func = lambda p,x: (p0[0]+p0[2])*np.exp(-x/p0[1])+p0[2]
    err_func = lambda p,x,y: fit_func(p,x)-y
    p1,cov,info1, info2, success = sp.optimize.leastsq(err_func, p0, \
                        args=(x, vector), \
                        full_output = 1)
    
    if (plot=='yes'):
        plt.figure('FITResults '+ title)
        plt.title('gausFit'+' '+title)
        plt.ylabel('Function'), plt.xlabel('x')
        plt.plot(x*x_step, vector,'ro',label = 'data')
        plt.plot(x*x_step,fit_func(p1, x),'b--',label = 'fit')
        plt.ylim (np.amin(vector)-.05, np.amax(vector)+.2)
        plt.grid(which = 'minor')
        plt.legend()
        plt.show()
        
    return p1, success

def polyfit_subtraction(x, y, deg, smooth=0, plot='y'):
    fit = np.polyfit(x, y, deg)
    fit_func = np.zeros((len(y)))
    for i in range(len(fit)):
        fit_func += x**(deg-i)*fit[i]
    result = y - fit_func
    result /= np.amax(result)
    if smooth!=0 and smooth%2!=0: result = smoothing(result, smooth)
    if plot=='y':
        fig = plt.figure('ployfit_subtraction result')
        fig.add_subplot(211)
        plt.plot(x, y, 'ro', alpha=.5)
        plt.plot(x, fit_func, 'b-')
        fig.add_subplot(212)
        plt.plot(x, result, 'k.-')
        plt.grid()
        plt.show()
    return result

def flatten_peaks(vector):
    """Flatten function peaks.
        They are substituted with a constant value, the same as the 
        one the function had before starting the peak slope.
    """
    flattened = np.copy(vector)
    for i in range(1, len(vector)):
        if(vector[i]>vector[i-1]):
            flattened[i] = flattened[i-1]
        else:
            flattened[i] = vector[i]
    return flattened