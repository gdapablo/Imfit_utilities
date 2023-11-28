from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import pyimfit
from photutils.isophote import EllipseGeometry, Ellipse, build_ellipse_model
from photutils import EllipticalAperture
from tqdm import tqdm
from matplotlib import gridspec
import models
import pdb
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator

plt.ion()

# Plotting image, model and residuals
def image(intensity, model, noise_lv):
    fig, ax = plt.subplots(1,3,figsize=(20,5))
    ax.ravel()

    noise = np.zeros_like(model) + noise_lv # Creating syntethic noise image to add to our model

    im1 = ax[0].imshow(np.log10(intensity), vmin=-4, vmax=2, cmap='rainbow', origin='lower')
    im2 = ax[1].imshow(np.log10(model+noise), vmin=-4, vmax=2, cmap='rainbow', origin='lower')
    im3 = ax[2].imshow((intensity - model - noise), cmap='rainbow', vmin=-.2,vmax=.2, origin='lower')

    ax[0].set_title('Data', fontsize=15); ax[1].set_title('Model', fontsize=15); ax[2].set_title('Residuals', fontsize=15)

    ax[0].tick_params(axis='x',labelsize=15); ax[0].tick_params(axis='y',labelsize=15)
    ax[1].tick_params(axis='x',labelsize=15); ax[1].tick_params(axis='y',labelsize=15)
    ax[2].tick_params(axis='x',labelsize=15); ax[2].tick_params(axis='y',labelsize=15)

    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    #ticks = MaxNLocator(nticks).tick_values(vmin, vmax)
    cb1 = fig.colorbar(im1, cax=cax, orientation='vertical')
    cb1.set_label(label=r'$\log(Flux)$',fontsize=15)
    cb1.ax.tick_params(labelsize=15)

    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    #ticks = MaxNLocator(nticks).tick_values(vmin, vmax)
    cb2 = fig.colorbar(im2, cax=cax, orientation='vertical')
    cb2.set_label(label=r'$\log(Flux)$',fontsize=15)
    cb2.ax.tick_params(labelsize=15)

    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    #ticks = MaxNLocator(nticks).tick_values(vmin, vmax)
    cb3 = fig.colorbar(im3, cax=cax, orientation='vertical')
    cb3.set_label(label=r'$Flux_{data}-Flux_{model}$',fontsize=15)
    cb3.ax.tick_params(labelsize=15)

    plt.subplots_adjust(wspace=0.5)

    input('Press ENTER to continue: ')
    plt.close()

# Mask stars
def mask_stars(image,xe,ye,width,length,alpha):

    image[np.isnan(image)] = 0.0; s = np.shape(image)
    x, y   = np.arange(s[1]), np.arange(s[0])
    xx, yy = np.meshgrid(x,y)
    x, y   = xx.reshape(s[0]*s[1]), yy.reshape(s[0]*s[1])
    spec   = image.reshape(s[0]*s[1])

    elip_mask_gal = (((x-xe) * np.cos(alpha) + (y-ye) * np.sin(alpha)) / (width/2)) ** 2 + (((x-xe) * np.sin(alpha) - (y-ye) * np.cos(alpha)) / (length/2)) ** 2 <= 50.0

    #idx = np.sqrt( (x - x0)**2 + (y - y0)**2 ) <= rad
    spec[elip_mask_gal] = np.nan

    return spec.reshape([s[0],s[1]])

# Mask for detecting the level of SKY
def substract_sky(image,min,max):

    where_are_NaNs = np.isnan(image)

    vector = image[~where_are_NaNs]
    num = len(vector)
    vector.sort()
    sel = vector[int(num*min) : int(num*max)]
    min_sel = sel.min()
    max_sel = sel.max()

    mask = np.where( (image >= min_sel) & (image <= max_sel), 0, 1 )

    mask[where_are_NaNs] = 1

    return mask

# FUNCTION to constrain to the centre of the galaxy
def extractor(x0, y0, n_pix, data):

    data[np.isnan(data)] = 0.0; s = np.shape(data)
    x, y   = np.arange(s[1]), np.arange(s[0])
    xx, yy = np.meshgrid(x,y)
    x, y   = xx.reshape(s[0]*s[1]), yy.reshape(s[0]*s[1])
    spec   = data.reshape(s[0]*s[1])

    # Centering the coordinates
    x, y = x - x0, y - y0

    boundx = (x <= n_pix) & (x >= -n_pix)
    x, y, spec = x[boundx], y[boundx], spec[boundx]
    boundy = (y <= n_pix) & (y >= -n_pix)
    x, y, spec = x[boundy], y[boundy], spec[boundy]

    idx = spec.argmax()
    xc, yc = x[idx], y[idx]

    section = np.copy(spec)
    dim  = len(section)
    data = section.reshape(int(np.sqrt(dim)),int(np.sqrt(dim)))
    s = np.shape(data)

    x, y   = np.arange(s[1]), np.arange(s[0])
    xx, yy = np.meshgrid(x,y)
    x, y   = xx.reshape(s[0]*s[1]), yy.reshape(s[0]*s[1])
    spec   = data.reshape(s[0]*s[1])

    idx = spec.argmax()
    xc, yc = x[idx], y[idx]

    return data, xc, yc

def ellipse_fitting(data, comp, model, x0, y0, sma, eps, pa):

    print('\nFitting the data with ellipses')

    data[np.isnan(data)] = 0.0; s = np.shape(data)
    x, y   = np.arange(s[1]), np.arange(s[0])
    xx, yy = np.meshgrid(x,y)
    x, y   = xx.reshape(s[0]*s[1]), yy.reshape(s[0]*s[1])
    spec   = data.reshape(s[0]*s[1])

    # Centering the coordinates
    x, y = x - x0, y - y0

    #data, xcen, ycen = extractor(x, y, sma, spec)

    # Defining the geometry of this
    #geometry = EllipseGeometry(xcen, ycen, sma - 20, eps, pa)
    geometry = EllipseGeometry(x0, y0, sma, eps, pa)
    ellipse  = Ellipse(data,geometry=geometry)

    # Fitting this rubbish
    fit = ellipse.fit_image()

    ellipse  = Ellipse(model,geometry=geometry)
    fit_model = ellipse.fit_image()

    # Getting the brightness of individual MODEL components
    bright = []; labels = []
    for i in tqdm(range(len(comp))):
        geometry = EllipseGeometry(x0, y0, sma, eps, pa)
        ellipse  = Ellipse(comp[i],geometry=geometry)
        if len(ellipse.fit_image()) > 0:
            bright.append(ellipse.fit_image())
            labels.append(names[i])
        else:
            continue

    #fig, ax = plt.subplots(1,3,figsize=(30,10))
    #ax.ravel()
    #ax[0].imshow(np.log10(data), cmap='plasma')

    # Defining the aperture
    #aper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma, geometry.sma*(1.0 - geometry.eps), geometry.pa)

    #aper.plot(ax = ax[0], color='white')

    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax = ax.ravel()
    ax[0].plot(fit.sma,fit.intens,label='data'); ax[1].plot(fit.sma,fit.intens,label='data')
    ax[1].plot(fit_model.sma,fit_model.intens,label='total model')

    for i in range(len(bright)):
        ax[0].plot(bright[i].sma,bright[i].intens,label=labels[i])

    ax[0].set_xscale('log'); ax[0].set_yscale('log')
    ax[1].set_xscale('log'); ax[1].set_yscale('log')

    ax[0].legend(); ax[1].legend()

    '''fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
    ax0 = plt.subplot(gs[0])
    ax0.plot(fit.sma,fit.intens,label='data');
    for i in range(len(bright)):
        ax0.plot(bright[i].sma,bright[i].intens,label=labels[i])

    ax1 = plt.subplot(gs[1])
    ax1.plot(fit.sma,fit.intens,label='data')
    ax1.plot(fit_model.sma,fit_model.intens,label='total model')

    # RESIDUALS
    ax2 = plt.subplot(gs[2])
    ax2.plot(fit.sma,fit.intens,label='data');
    for i in range(len(bright)):
        ax0.plot(bright[i].sma,bright[i].intens,label=labels[i])

    ax1 = plt.subplot(gs[1])
    ax1.plot(fit.sma,fit.intens,label='data')
    ax1.plot(fit_model.sma,fit_model.intens,label='total model')

    ax0.set_xscale('log'); ax0.set_yscale('log')
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax0.legend(); ax1.legend()

    plt.tight_layout()'''

    # Modelling
    #print('\nFitting the model')
    #model_image = build_ellipse_model(data.shape, fit)

    #ax[1].imshow(np.log10(model_image), cmap='plasma')
    #ax[2].imshow(data - model_image, cmap='plasma')

    input('Press ENTER to continue: ')
    plt.close()

    return fit, model_image

def get_PSF(filename, x0, y0):
    PSF_image = fits.getdata(filename)

    # Using the oversample PSF
    dim_PSF   = int(len(PSF_image)/2) + 1
    if dim_PSF % 2 == 1:
        lim1, lim2, lim3, lim4 = y0-dim_PSF, y0+dim_PSF-1, x0-dim_PSF, x0+dim_PSF-1
        #psfOsamp = pyimfit.MakePsfOversampler(PSF_image, 10, (lim1,lim2, lim3,lim4))
        psfOsamp = pyimfit.MakePsfOversampler(PSF_image, 10, (int(lim1),int(lim2), int(lim3),int(lim4)))
    else:
        lim1, lim2, lim3, lim4 = y0-dim_PSF, y0+dim_PSF, x0-dim_PSF, x0+dim_PSF
        psfOsamp = pyimfit.MakePsfOversampler(PSF_image, 10, (lim1,lim2, lim3,lim4))

    osampleList = [psfOsamp]
    return osampleList, PSF_image

def brightness_fitting(x0,y0,data,model,comps,ang,labels):

    step = 500
    s = np.shape(data)
    x, y   = np.arange(s[1]), np.arange(s[0])
    xx, yy = np.meshgrid(x,y)
    x, y   = xx.reshape(s[0]*s[1]), yy.reshape(s[0]*s[1])
    x, y   = (x - x0)*0.05, (y - y0)*0.05
    xcord,ycord = x*np.cos(ang) - y*np.sin(ang), x*np.sin(ang) + y*np.cos(ang)
    spec_data  = data.reshape(s[0]*s[1])
    spec_model = model.reshape(s[0]*s[1])

    spec_comps = []
    for i in range(len(comps)):
        fit_comp   = pyimfit.Imfit(comps[i])
        model_comp = fit_comp.getModelImage(shape = s)
        spec_comps.append(model_comp.reshape(s[0]*s[1]))

    apertures = np.linspace(0.0,100.0,step)
    flux_data, flux_model = [], []
    for i in range(len(apertures)-1):
        idx = (ycord <= 0.1) & (ycord >= -0.1)
        xbin, ybin, spec_data_bin, spec_model_bin = xcord[idx], ycord[idx], spec_data[idx], spec_model[idx]
        idx = (xbin >= apertures[i]) & (xbin <= apertures[i+1])
        xbin, ybin, spec_data_bin, spec_model_bin = xbin[idx], ybin[idx], spec_data_bin[idx], spec_model_bin[idx]
        flux_data.append(np.sum(spec_data_bin))
        flux_model.append(np.sum(spec_model_bin))

    #fig,ax = plt.subplots(2,2,figsize=(15,9))
    #ax = ax.ravel()

    fig, ax = plt.subplots(2, 2, gridspec_kw={'height_ratios': [2, 1]}, figsize=(15,9), sharex=True)
    ax = ax.ravel()

    ax[0].plot(apertures[:-1],-2.5*np.log10(flux_data),label='data')
    ax[0].plot(apertures[:-1],-2.5*np.log10(flux_model),label='model')
    ax[2].plot(apertures[:-1],np.array(-2.5*np.log10(flux_data))-np.array(-2.5*np.log10(flux_model)),'.')

    #ax[2].plot(range(5), range(5, 10))
    #ax[3].plot(range(5), range(10, 5, -1))

    #ax[0].plot(apertures[:-1],-2.5*np.log10(flux_data),label='data')
    #ax[0].plot(apertures[:-1],-2.5*np.log10(flux_model),label='model')
    #ax[1].plot(apertures[:-1],np.array(-2.5*np.log10(flux_data))-np.array(-2.5*np.log10(flux_model)),'.')

    ap_min = np.linspace(0.0,20.0,step)
    flux_data, flux_model = [], []
    for i in range(len(ap_min)-1):
        idx = (xcord <= 0.1) & (xcord >= -0.1)
        xbin, ybin, spec_data_bin, spec_model_bin = xcord[idx], ycord[idx], spec_data[idx], spec_model[idx]
        idx = (ybin >= ap_min[i]) & (ybin <= ap_min[i+1])
        xbin, ybin, spec_data_bin, spec_model_bin = xbin[idx], ybin[idx], spec_data_bin[idx], spec_model_bin[idx]
        flux_data.append(np.sum(spec_data_bin))
        flux_model.append(np.sum(spec_model_bin))

    ax[1].plot(apertures[:-1],-2.5*np.log10(flux_data))#,label='data')
    ax[1].plot(apertures[:-1],-2.5*np.log10(flux_model))#,label='model')
    ax[3].plot(apertures[:-1],np.array(-2.5*np.log10(flux_data))-np.array(-2.5*np.log10(flux_model)),'.')

    for j in range(len(spec_comps)):
        flux_model = []
        for i in range(len(apertures)-1):
            idx = (ycord <= 0.1) & (ycord >= -0.1)
            xbin, ybin, spec_model_bin = xcord[idx], ycord[idx], spec_comps[j][idx]
            idx = (xbin >= apertures[i]) & (xbin <= apertures[i+1])
            xbin, ybin, spec_model_bin = xbin[idx], ybin[idx], spec_model_bin[idx]
            flux_model.append(np.sum(spec_model_bin))

        ax[0].plot(apertures[:-1],-2.5*np.log10(flux_model),label=labels[j])

    for j in range(len(spec_comps)):
        flux_model = []
        for i in range(len(ap_min)-1):
            idx = (xcord <= 0.1) & (xcord >= -0.1)
            xbin, ybin, spec_model_bin = xcord[idx], ycord[idx], spec_comps[j][idx]
            idx = (ybin >= ap_min[i]) & (ybin <= ap_min[i+1])
            xbin, ybin, spec_model_bin = xbin[idx], ybin[idx], spec_model_bin[idx]
            flux_model.append(np.sum(spec_model_bin))

        ax[1].plot(apertures[:-1],-2.5*np.log10(flux_model))#,label=labels[j])

    ax[0].set_ylim(5,-8.0); ax[1].set_ylim(5,-8.0)
    ax[0].set_xscale('log'); ax[1].set_xscale('log')
    ax[2].set_xscale('log'); ax[3].set_xscale('log')
    #ax[0].set_xlim(0.0,100.0); ax[2].set_xlim(0.0,100.0)
    ax[0].legend(); ax[1].legend()

    ax[0].set_ylabel(r'$\mu (mag/arcsec^{2})$',fontsize=15); ax[1].set_ylabel(r'$\mu (mag/arcsec^{2})$',fontsize=15)
    #ax[0].set_xlabel('x (arcsec)',fontsize=15); ax[1].set_xlabel('x (arcsec)',fontsize=15)
    ax[2].set_xlabel('x (arcsec)',fontsize=15); ax[3].set_xlabel('x (arcsec)',fontsize=15)

    ax[0].tick_params(axis='x',labelsize=15); ax[0].tick_params(axis='y',labelsize=15)
    ax[1].tick_params(axis='x',labelsize=15); ax[1].tick_params(axis='y',labelsize=15)
    ax[2].tick_params(axis='x',labelsize=15); ax[2].tick_params(axis='y',labelsize=15)
    ax[3].tick_params(axis='x',labelsize=15); ax[3].tick_params(axis='y',labelsize=15)

    ax[0].set_title('Semi-major axis',fontsize=15); ax[1].set_title('Semi-minor axis',fontsize=15)

    fig.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9,
                        wspace=0.2, hspace=0.001)

    input('Press ENTER to continue: ')
    plt.close()

def make_cube(model,Re,pix2kpc):

    s = np.shape(model)
    x,y = np.arange(s[1]), np.arange(s[0])
    xx,yy = np.meshgrid(x,y)
    x,y = xx.reshape(s[0]*s[1]), yy.reshape(s[0]*s[1])

    model_list = model.reshape(s[0]*s[1])
    idx = model_list.argmax()

    x,y = (x-x[idx])*pix2kpc,(y-y[idx])*pix2kpc

    rad = np.sqrt(x**2+y**2)
    idx = rad <= 5*Re*0.1

    return model_list[idx]

file = 'FCC153_HST_croped'
intensity = fits.getdata(file + '.fits') #/(4.0*np.pi*exptime*pixelsize*pixelsize)
s = np.shape(intensity)

#x0, y0   = 200.5, 200.5
#ang      = np.radians(0.0)
#model_desc, components, labels = models.galaxy_model.FCC170_bulge()
#filename = 'oversamplePSF.fits'

if file[:6] == 'FCC083':
    model_desc, components, labels = models.galaxy_model.FCC083()
    filename = 'HST/j90x08010/oversamplePSF.fits'
    x0, y0   = 991, 1547
    ang      = np.radians(112)

if file[:6] == 'FCC153':
    model_desc, components, labels = models.galaxy_model.FCC153()
    filename = 'HST/j90x13020/oversamplePSF.fits'
    x0, y0   = 1627, 724
    ang      = np.radians(97.2 - 90.0)
    noise_lv = 0.001871
    Re       = 19.8 # From Iodice et al. 2019 (in arcsec)
    distance = 21.73 * 1000

if file[:6] == 'FCC170':
    model_desc, components, labels = models.galaxy_model.FCC170()
    filename = 'HST/j90x12010/oversamplePSF.fits'
    x0, y0   = 1068, 1752
    ang      = np.radians(-338.0 + 90.0)
    noise_lv = 0.00124
    Re       = 15.9
    distance = 19.57 * 1000 # Distance towards the galaxy in kpc

if file[:6] == 'FCC177':
    model_desc, components, labels = models.galaxy_model.FCC177()
    filename = 'HST/j90x14020/oversamplePSF.fits'
    x0, y0   = 1126, 1649
    ang      = np.radians(-208.9 - 90.0)
    intensity = mask_stars(intensity,1167,1861,35.0,15.0,15.0)
    noise_lv = 0.00263
    Re       = 35.9
    distance = 17.80 * 1000

arcsec2kpc = distance * np.tan(np.pi / 648000)
pix2kpc    = 0.05*arcsec2kpc # Coonversion factor from HST ACS

osampleList, PSF_image = get_PSF(filename, x0, y0)

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
imfit_fitter = pyimfit.Imfit(model_desc)

# Computing the preliminar model
model = imfit_fitter.getModelImage(shape = s)

light_model = []
aux1, xx, yy = [], [], []
for i in range(len(components)):
    fit_comp   = pyimfit.Imfit(components[i])
    model_comp = fit_comp.getModelImage(shape = s)
    light_model.append(make_cube(model_comp,Re*0.1,pix2kpc))
    models.galaxy_model.flux_mass(file[:6],model_comp,labels[i])

labels = np.array(labels)
light_model = np.array(light_model)
# Getting the thin-disc to total light contribution
thin = np.where(labels == 'thin_disk')[0]; tot = np.where(labels != 'thin_disk')[0]
D_T = np.sum(light_model[thin])/np.sum(light_model[tot])
print('\n- Disc to total light contribution within 5Re: ',D_T,' -')

# Plotting the maps
image(intensity, model, noise_lv)

# Plotting surface brightness profiles
brightness_fitting(x0,y0,intensity,model,components,ang,labels)

signal = np.copy(intensity)
signal[signal < 0.0] = 0.0
signal[np.isnan(signal)] = 0.0
errors = np.sqrt(signal)

# Avoiding those regions with NaNs or Inf
mask = ~np.isfinite(intensity)

imfit_fitter.loadData(intensity, psf_oversampling_list=osampleList, mask = mask, error = errors)

imfit_fitter.fit(intensity, mask = mask, verbose = 1, error = errors)
bestfit_parameters = imfit_fitter.getRawParameters()

if imfit_fitter.fitConverged is True:
    print("Fit converged: chi^2 = {0}, reduced chi^2 = {1}".format(imfit_fitter.fitStatistic,
        imfit_fitter.reducedFitStatistic))
    bestfit_params = imfit_fitter.getRawParameters()
    print("Best-fit parameter values:", bestfit_params)

save = input('\nSave paremeterfile? y/n: ')
if save == 'y':
    np.savetxt('parameterfile.txt', bestfit_params)

# Model image
model = imfit_fitter.getModelImage()
# Total flux and individual fluxes of the model
(totalFlux, componentFluxes) = imfit_fitter.getModelFluxes()

# 2D images of fitted model
image(intensity, model, noise_lv)
# Plotting surface brightness profiles of last model
#brightness_fitting(x0,y0,intensity,model,components,ang,labels)
