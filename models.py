import numpy as np
import pyimfit
from acstools import acszpt

class galaxy_model():

    def flux_mass(galaxy,model,label):

        flux = 1.7886880e-19 # HST flux conversion

        if galaxy == 'FCC083':
            q = acszpt.Query(date='2004-11-01', detector='WFC', filt='F850LP')
            mod = 31.26; M_sun = 4.01
            M_L_ratio = 2.5

        if galaxy == 'FCC153':
            q = acszpt.Query(date='2004-11-19', detector='WFC', filt='F475W')
            mod = 31.588; M_sun = 5.21
            M_L_ratio = 1.5

        if galaxy == 'FCC170':
            q = acszpt.Query(date='2004-11-01', detector='WFC', filt='F850LP')
            mod = 31.705; M_sun = 4.01
            M_L_ratio = 1.5

        if galaxy == 'FCC177':
            q = acszpt.Query(date='2004-10-23', detector='WFC', filt='F475W')
            mod = 31.509; M_sun = 5.21
            M_L_ratio = 3

        filter_zpt = q.fetch()
        Intensity  = np.sum(model)

        mag   = -2.5 * np.log10(Intensity) + filter_zpt['VEGAmag'][0].value
        Mabs  = mag - mod
        Lum   = 10**(-0.4*(Mabs - M_sun))
        print('\nMass of the '+label+': ', "{:e}".format(Lum*M_L_ratio))

    def FCC083():

        # Total model
        model = pyimfit.SimpleModelDescription()

        components = []
        labels = ['bulge', 'disc', 'NSD', 'nucleus']

        # define the limits on X0 and Y0 as +/-10 pixels relative to initial values
        model.x0.setValue(992, [990,995])
        model.y0.setValue(1548, [1545,1552])

        # BULGE COMPONENT
        components.append(pyimfit.SimpleModelDescription())
        components[0].x0.setValue(992, [990,995])
        components[0].y0.setValue(1548, [1545,1552])

        bulge = pyimfit.make_imfit_function('Exponential', label=labels[0])
        bulge.PA.setValue(157.39, [150,160])
        bulge.ell.setValue(0.43, [1e-5,.7])
        bulge.I_0.setValue(0.87, [1e-4,1])
        bulge.h.setValue(288.7, [100,500])

        components[0].addFunction(bulge)

        # DISC COMPONENT
        components.append(pyimfit.SimpleModelDescription())
        components[1].x0.setValue(992, [990,995])
        components[1].y0.setValue(1548, [1545,1552])

        disc = pyimfit.make_imfit_function('EdgeOnDisk', label=labels[1])
        disc.PA.setValue(159.6,       [150,160])
        disc.L_0.setValue(9.8e-3,       [5e-4,10])
        disc.h.setValue(64.75,        [50,300])
        disc.n.setValue(1,          fixed = True)
        disc.z_0.setValue(34.79,         [10,50])

        components[1].addFunction(disc)

        # NSD COMPONENT
        components.append(pyimfit.SimpleModelDescription())
        components[2].x0.setValue(992, [990,995])
        components[2].y0.setValue(1548, [1545,1552])

        NSD = pyimfit.make_imfit_function('EdgeOnDisk', label=labels[2])
        NSD.PA.setValue(159.56,       [155,160])
        NSD.L_0.setValue(0.153,       [5e-2,5])
        NSD.h.setValue(5,        [5,30])
        NSD.n.setValue(1,          [0.5,10])
        NSD.z_0.setValue(2.48,         [2,15])

        components[2].addFunction(NSD)

        # NUCLEUS COMPONENT
        components.append(pyimfit.SimpleModelDescription())
        components[3].x0.setValue(992, [990,995])
        components[3].y0.setValue(1548, [1545,1552])

        nucleus = pyimfit.make_imfit_function('Sersic', label=labels[3])
        nucleus.PA.setValue(158.04,      [150,160])
        nucleus.ell.setValue(0.32,	      [0.001,0.5])
        nucleus.n.setValue(2.22,      	[1,4])
        nucleus.I_e.setValue(2.23,	    [1e-1,10])
        nucleus.r_e.setValue(56.82,	        [2,150])

        components[3].addFunction(nucleus)

        model.addFunction(bulge)
        model.addFunction(disc)
        model.addFunction(NSD)
        model.addFunction(nucleus)

        return model, components, labels


    def FCC153():

        # Total model
        model = pyimfit.SimpleModelDescription()

        components = []
        labels = ['bulge', 'thin_disk', 'thick_disk']

        # define the limits on X0 and Y0 as +/-10 pixels relative to initial values
        model.x0.setValue(1627.7, [1620,1630])
        model.y0.setValue(724.63, [720,730])

        # BULGE COMPONENT
        components.append(pyimfit.SimpleModelDescription())
        components[0].x0.setValue(1627, [1620,1630])
        components[0].y0.setValue(724, [715,73])

        bulge = pyimfit.make_imfit_function('Sersic', label=labels[0])
        bulge.PA.setValue(86.764, [0,180])
        bulge.ell.setValue(0.36, [1e-5,0.5])
        bulge.n.setValue(5.92, [0.5,10])
        bulge.I_e.setValue(0.03, [1e-4,10])
        bulge.r_e.setValue(215.7, [1,300])

        components[0].addFunction(bulge)

        # THIN DISC COMPONENT
        components.append(pyimfit.SimpleModelDescription())
        components[1].x0.setValue(1627, [1620,1630])
        components[1].y0.setValue(724, [715,73])

        thin_disk = pyimfit.make_imfit_function('EdgeOnDisk', label=labels[1])
        thin_disk.PA.setValue(84.64,       [80,90])
        thin_disk.L_0.setValue(4.28e-3,       [1e-4,5])
        thin_disk.h.setValue(218.15,        [200,250])
        thin_disk.n.setValue(1,          fixed = True)
        thin_disk.z_0.setValue(15.59,         [5,20])

        components[1].addFunction(thin_disk)

        # THICK DISC COMPONENT
        components.append(pyimfit.SimpleModelDescription())
        components[2].x0.setValue(1627, [1620,1630])
        components[2].y0.setValue(724, [715,73])

        thick_disk = pyimfit.make_imfit_function('EdgeOnDisk', label=labels[2])
        thick_disk.PA.setValue(83.17,       [80,90])
        thick_disk.L_0.setValue(8.12e-4,       [5e-5,5])
        thick_disk.h.setValue(277.35,        [250,300])
        thick_disk.n.setValue(1.,          fixed = True)
        thick_disk.z_0.setValue(62.98,         [55,70])

        components[2].addFunction(thick_disk)

        model.addFunction(bulge)
        model.addFunction(thin_disk)
        model.addFunction(thick_disk)

        return model, components, labels


    def FCC170():

        # Total model
        model = pyimfit.SimpleModelDescription()

        components = []
        labels = ['bulge', 'thin_disk', 'thick_disk', 'NSD']

        # Define the centre of the croped HSt image as an initial guess within a range of values, in this case +- 10 pixels.
        model.x0.setValue(1068.96,   [1068,1070])
        model.y0.setValue(1752.46,   [1751,1753])

        # BULGE COMPONENT
        components.append(pyimfit.SimpleModelDescription())
        components[0].x0.setValue(1068.96,   [1068,1070])
        components[0].y0.setValue(1752.49,   [1751,1753])

        bulge = pyimfit.make_imfit_function('Sersic', label=labels[0])
        bulge.PA.setValue(157.67, fixed=True)
        bulge.ell.setValue(0.137,   [0.,0.5])
        bulge.n.setValue(2., [2,2.5])
        bulge.I_e.setValue(1.61, [1e-4,5])
        bulge.r_e.setValue(85.69, [50,200])

        components[0].addFunction(bulge)

        # THICK DISC COMPONENT
        components.append(pyimfit.SimpleModelDescription())
        components[1].x0.setValue(1068.96,   [1068,1070])
        components[1].y0.setValue(1752.49,   [1751,1753])

        thin_disk = pyimfit.make_imfit_function('EdgeOnDisk', label=labels[1])
        thin_disk.PA.setValue(159.8, [150,165])
        thin_disk.L_0.setValue(2e-3, [1e-4,1])
        thin_disk.h.setValue(264.65, [230,350])
        thin_disk.n.setValue(1, fixed = True)
        thin_disk.z_0.setValue(33.77, [15,70])

        components[1].addFunction(thin_disk)

        # THIN DISC COMPONENT
        components.append(pyimfit.SimpleModelDescription())
        components[2].x0.setValue(1068.96,   [1068,1070])
        components[2].y0.setValue(1752.49,   [1751,1753])

        thick_disk = pyimfit.make_imfit_function('EdgeOnDisk', label=labels[2])
        thick_disk.PA.setValue(159.68, [158,161])
        thick_disk.L_0.setValue(8.5e-4, [5e-5,5])
        thick_disk.h.setValue(322.47, [270,400])
        thick_disk.n.setValue(1, fixed = True)
        thick_disk.z_0.setValue(82.93, [50,120])

        components[2].addFunction(thick_disk)

        # NSD COMPONENT
        components.append(pyimfit.SimpleModelDescription())
        components[3].x0.setValue(1068.96,   [1068,1070])
        components[3].y0.setValue(1752.49,   [1751,1753])

        ''' As we base our NSD decomposition on Morelli et al. (in preparation) study, we need to constrain the parameters
            with respect to their results as to get a reliable NSD that seems like the observed one.'''

        NSD = pyimfit.make_imfit_function('EdgeOnDisk', label=labels[3])
        NSD.PA.setValue(157.67, [157,161])
        NSD.L_0.setValue(0.48, [0.4,1.5])
        NSD.h.setValue(14.52, fixed = True)
        NSD.n.setValue(1., fixed = True)
        NSD.z_0.setValue(5, [5,12])

        components[3].addFunction(NSD)

        model.addFunction(bulge)
        model.addFunction(thin_disk)
        model.addFunction(thick_disk)
        model.addFunction(NSD)

        return model, components, labels


    def FCC177():

        # Total model
        model = pyimfit.SimpleModelDescription()

        components = []
        labels = ['bulge', 'thin_disk', 'thick_disk']

        # define the limits on X0 and Y0 as +/-10 pixels relative to initial values
        model.x0.setValue(1126.57, [1120,1130])
        model.y0.setValue(1649.13, [1640,1660])

        # BULGE COMPONENT
        components.append(pyimfit.SimpleModelDescription())
        components[0].x0.setValue(1126, [1120,1130])
        components[0].y0.setValue(1649, [1640,1660])

        bulge = pyimfit.make_imfit_function('Sersic', label=labels[0])
        bulge.PA.setValue(205.38, [200,215])
        bulge.ell.setValue(0, [0.0, 0.8])
        bulge.n.setValue(2, [.5,10.0])
        bulge.I_e.setValue(0.7, [1e-5,1])
        bulge.r_e.setValue(30, [5,100])

        components[0].addFunction(bulge)

        # THIN DISC COMPONENT
        components.append(pyimfit.SimpleModelDescription())
        components[1].x0.setValue(1126, [1120,1130])
        components[1].y0.setValue(1649, [1640,1660])

        thin_disk = pyimfit.make_imfit_function('EdgeOnDisk', label=labels[1])
        thin_disk.PA.setValue(210.15, [200,215])
        thin_disk.L_0.setValue(6e-4,       [2e-5,3])
        #thin_disk.h.setValue(313.49,        [150,350])
        thin_disk.h.setValue(249.81,        [200,250])
        thin_disk.n.setValue(1., fixed = True)
        thin_disk.z_0.setValue(16.91,         [5,25])
        #thin_disk.z_0.setValue(9.66,         [2,100])

        components[1].addFunction(thin_disk)

        # THICK DISC COMPONENT
        components.append(pyimfit.SimpleModelDescription())
        components[2].x0.setValue(1126, [1120,1130])
        components[2].y0.setValue(1649, [1640,1660])

        thick_disk = pyimfit.make_imfit_function('EdgeOnDisk', label=labels[2])
        thick_disk.PA.setValue(210.15, [200,215])
        thick_disk.L_0.setValue(7.15e-4,       [1e-5,1])
        thick_disk.h.setValue(263.26,        [250,300])
        #thick_disk.h.setValue(261.45,        [200,350])
        thick_disk.n.setValue(1., fixed = True)
        thick_disk.z_0.setValue(74.39,         [50,100])

        components[2].addFunction(thick_disk)

        model.addFunction(bulge)
        model.addFunction(thin_disk)
        model.addFunction(thick_disk)

        return model, components, labels
