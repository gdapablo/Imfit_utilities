import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
from astropy.io import fits
from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from astropy.nddata import NDData
from astropy.visualization import simple_norm
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import SqrtStretch
from photutils import DAOStarFinder
from photutils.psf import extract_stars
from photutils import EPSFBuilder
import numpy as np
from photutils import find_peaks
from photutils import CircularAperture

def source_detector(data):
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    print((mean, median, std))
    daofind = DAOStarFinder(fwhm=5.0, threshold=50)
    sources = daofind(data - median)
    for col in sources.colnames:
        sources[col].info.format = '%.8g'  # for consistent table output
        print(sources)
    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    apertures = CircularAperture(positions, r=20.)
    norm = ImageNormalize(stretch=SqrtStretch())
    plt.figure(figsize=(10,10))
    plt.imshow(np.log10(data), cmap='rainbow', vmin=-4, vmax=2., origin='lower', norm=norm)
    apertures.plot(color='blue', lw=1.5, alpha=0.5)
    stars_tbl = Table()
    stars_tbl['x'] = positions.T[0]
    stars_tbl['y'] = positions.T[1]
    return stars_tbl


plt.ion()

# Openning the HST image
filename = 'HST/j90x08010/j90x08010_drc.fits'
#filename = 'FCC170_bulge_decomp.fits'
hdu   = fits.open(filename)
data  = hdu[1].data

# Initial plot
#fig, ax = plt.subplots(1, 1, figsize=(10,10))
#ax.imshow(np.log10(data), cmap = 'rainbow')

# We extract the peaks above a treshold in order to get the brigthest stars
#peaks_tbl = find_peaks(data, threshold=100.)
#peaks_tbl['peak_value'].info.format = '%.8g'  # for consistent table output
#print(peaks_tbl)

# Defining circular apertures around the stars
#aper = CircularAperture((peaks_tbl['x_peak'], peaks_tbl['y_peak']), 25)
#aper.plot(ax = ax, color='white')

# We mask the stars close to the borders in order to avoid errors
#size = 25
#hsize = (size - 1) / 2
#x = peaks_tbl['x_peak']
#y = peaks_tbl['y_peak']
#mask = ((x > hsize) & (x < (data.shape[1] -1 - hsize)) &
#         (y > hsize) & (y < (data.shape[0] -1 - hsize)))

# Creating the table of good star positions
#stars_tbl = Table()
#x = x[mask]
#y = y[mask]

# Eliminating similar values
#rad = np.sqrt(x**2 + y**2); aux = [True]
#for i in range(len(rad)-1):
#    aux.append(abs(rad[i] - rad[(i+1):]) >= 10.0)

#idx = []
#for i in range(len(aux)):
#    if type(aux[i]) == bool:
#        idx.append(aux[i])
#    else:
#        idx.append(aux[i][0])

# Creating the table of good star positions
#stars_tbl['x'] = x[idx]
#stars_tbl['y'] = y[idx]
opt = str(input('Do you want to introduce the coordinates throught a param file? y/n '))
if opt == 'y':
    stars_tbl = Table()
    positions = np.loadtxt(filename[:14] + 'positions.param')
    stars_tbl['x'] = positions.T[0]
    stars_tbl['y'] = positions.T[1]
    plt.figure(figsize=(10,10))
    plt.imshow(np.log10(data), cmap='rainbow', vmin=-4, vmax=2., origin='lower')
    apertures = CircularAperture(positions, r=20.)
    apertures.plot(color='blue', lw=1.5, alpha=0.5)
else:
    stars_tbl = source_detector(data)

# Subtracting the background from the image (this step is useless for HST images but we do it just in case...)
mean_val, median_val, std_val = sigma_clipped_stats(data, sigma=2.)
data -= median_val

# Extracting the stars and calculating the PSF
nddata = NDData(data=data)
stars = extract_stars(nddata, stars_tbl, size=20)
nrows = 2
ncols = 4
fig2, ax2 = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20),
                        squeeze=True)
ax2 = ax2.ravel()
for i in range(nrows*ncols):
     norm = simple_norm(stars[i], 'log', percent=99.)
     ax2[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')

# Building the PSF
epsf_builder = EPSFBuilder(oversampling=10, maxiters=5,
                            progress_bar=True)
epsf, fitted_stars = epsf_builder(stars)
norm = simple_norm(epsf.data, 'log', percent=99.)
plt.figure(3,figsize=(10,10))
plt.imshow(epsf.data, norm=norm, origin='lower', cmap='viridis')
plt.colorbar()

# Saving the PSF into a fits file
hdu = fits.PrimaryHDU()
hdu.data = epsf.data
hdu.writeto(filename[:14] + 'oversamplePSF.fits', overwrite=True)
