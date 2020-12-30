'''Functions to test dust mechanics'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import glob
from pygadgetreader import *
from matplotlib import animation
import basic_astro_functions as b
import astrophysical_constants_cgs as c
import Quick_Render as Q
from scipy.stats import norm

#Font styles, sizes and weights
def set_rcparams(fsize=12):
    font = {'weight' : 'normal',
            'size'   : fsize,
            'serif' : 'Minion Pro'}
    plt.rc('font', **font)
    plt.rc('legend', fontsize=fsize)
    plt.rc('lines',linewidth = 2)
    
    return
set_rcparams()


#Code units
code_M     = c.Msol                  #Msol in g
code_L     = 100*c.AU                #100 AU in cm
code_V     = 297837.66               #v_K @ 100 AU #in cm/s
code_time  = code_L/code_V           #seconds #~160 Yrs
code_rho   = code_M/code_L**3
code_E     = code_M*code_V**2
code_G     = c.G /code_L**3*code_M*code_time**2
AU_scale   = 100


snapprefix = 'snapshot_'
savedir = '/rfs/TAG/rjh73/save_data/'
run_cols = ['#2EC4C6', '#CD6622', '#440044', '#FFC82E','#008822','#111144' ,'#8800bb','#0088ee']

#runfols = 'Sergei_runs'
#runfols = 'O_M2_runs'
#runfols = 'NSG_disc'
#runfols = 'Pow_law'
#runfols = 'SWD_2017'
#runfols = 'Gio_aav'
#runfols = 'Poly_test'
#runfols = 'Gio_N1e5_tests'
runfols = 'Gio_N1e6'
#runfols = 'Gio_N1e6_planets'



if runfols == 'Sergei_runs':
    filepath = '/rfs/TAG/rjh73/dust_tests_2017/'
    runfolders = ['O_M2_beta01_n1e6_a1/',
                  'O_M2_beta01_n1e6_a01/',
                  'O_M2_beta01_n1e6_a001/',
                  'O_M2_beta01_n1e6_a0001/'
    ]
    grain_as = [1.,
                0.1,
                0.01,
                0.001
    ]

elif runfols == 'O_M2_runs':
    filepath = '/rfs/TAG/rjh73/dust_accretion_runs/'
    runfolders = ['O_M2_beta10_a001/',
                  'O_M2_beta10_a01/',
                  'O_M2_beta10_a1/',
                  #'O_M2_beta10_a10/'
    ]
    grain_as = [0.01,
                0.1,
                1.,
                10.
    ]
    



elif runfols == 'NSG_disc':
    filepath = '/rfs/TAG/rjh73/dust_tests_2017/'
    runfolders = ['Gio_disc_a01/',
                  'Gio_disc_a001/',
                  'Gio_disc_a0001/',
                  'Gio_disc_a00001/'
    ]
    grain_as = [0.1,
                0.01,
                0.001,
                0.0001
    ]


elif runfols == 'Pow_law':
    filepath = '/rfs/TAG/rjh73/dust_tests_2017/'
    runfolders = ['Disc_N1e6_R0130_M002_beta01_dust/',
                  'Disc_N1e6_R0130_M002_beta01_dust_Z1/',
                  'Disc_N1e6_R0130_M002_beta01_dust_Z100/']
    Rin = 0.44
    Rout= 0.46
    snapid = 100
    aavs = [1,1,1]

elif runfols == 'ring_test':
    filepath = '/rfs/TAG/rjh73/dust_tests_2017/'
    runfolders = ['Disc_N1e6_R0130_M002_beta01_dustring/']

elif runfols == 'SWD_2017':
    filepath = '/rfs/TAG/rjh73/dust_accretion_runs/'
    runfolders = ['Sergei_Weidendust_2017/']
    Rin = 0.73
    Rout = 0.8
    snapid=63
    aavs = [1]


elif runfols == 'Gio_aav':
    filepath = '/rfs/TAG/rjh73/Gio_disc/'
    runfolders = [#'Gio_N1e6_aav1_amucm/','Gio_N1e6_aav01_amucm/',
                  'Gio_N1e6_aav1/',
                  'Gio_N1e6_aav01/',
                  'Gio_N1e6_aav01_altvisc/',#'Gio_N1e6_aav10/'
                  'Gio_N1e6_aav01_betavisc/',
                  'Gio_N1e6_aav01_convvisc/',
        'Gio_N1e6_aav01_R00120/'
    ]
    snapid = 25
    Rin = 0.44
    Rout= 0.46
    aavs = [#1,0.1,
            1,0.1,0.1,0.1,0.1,0.1]


elif runfols == 'Poly_test':
    filepath = '/rfs/TAG/rjh73/Clump_project/'
    runfolders = ['Poly_N1e5_M5_R3_rho1e12_T30_r60/',
                  'Poly_N1e5_M5_R3_rho1e12_T50_r60/',
                  'Poly_N1e5_M5_R3_rho2e12_T30_r60/',
                  'Poly_N1e5_M5_R3_rho2e12_T50_r60/',
                  #'Poly_N1e5_M5_R3_evap_rho1e12_T30/',
                  #'Poly_N1e5_M5_R3_evap_rho1e12_T50/',
                  'Disc_N1e5_M0005_R01200_aav1_beta5/'
    ]


elif runfols == 'Gio_N1e5_tests':
    filepath = '/rfs/TAG/rjh73/Gio_disc/'
    runfolders = [#'Gio_N1e5_aav01/',
                  'Gio_N1e5_aav01_dustin/',
                  #'Gio_N1e5_aav01_inteq/',
                  #'Gio_N1e5_aav01_conv_inteq/',
                  #'Gio_N1e5_aav01_conv/'
    ]


elif runfols == 'Gio_N1e6':
    filepath = '/rfs/TAG/rjh73/Gio_disc/'
    runfolders = [#'Gio_N1e6_aav01_R00120_MD001_Z10/',
                  'Gio_N1e6_aav01_R00120_MD01_Z10/',
                  #'Gio_N1e6_aav01_R00120_MD001_dustreadin_Zset/',
                  #'Gio_N1e6_aav01_R00120_S100_MD001_Z10/',
                  #'Gio_N1e6_aav01_R00120_S100_MD01_Z10/',
                  #'Gio_N1e6_aav01_R00120_MD001_Z10_soft/',
                  #'Gio_N1e6_aav01_R00120_MD001_Z10_a00001/',
                  'Gio_N1e6_aav01_R00120_MD01_Z10_errF00005/',
                  'Gio_N1e6_aav01_R00120_MD01_Z10_errI00005/'
    ]

elif runfols == 'Gio_N1e6_planets':
    filepath = '/rfs/TAG/rjh73/Gio_disc/'
    runfolders = [#'Gio_N1e6_aav01_R00120_MD001_Z10/',
                  #'Gio_N1e6_aav01_R00120_MD001_Z10_MP03/',
                  #'Gio_N1e6_aav01_R00120_MD001_Z10_MP1/',
                  #'Gio_N1e6_aav01_R00120_MD001_Z10_MP3/',
                  #'Gio_N1e6_aav01_R00120_MD01_Z10/',
                  'Gio_N1e6_aav01_R00120_MD01_Z10_MP03/',
                  'Gio_N1e6_aav01_R00120_MD01_Z10_MP1/',
                  'Gio_N1e6_aav01_R00120_MD01_Z10_MP3/',
    ]


    
def vertical_profile(anim_mode=False,anim_ind=0):
    '''Animate and plot vertical Height scale for gas and dust distributions'''
    H_max=0.2
    num_bins = 200
    bins = np.linspace(0,2*H_max,num_bins+1)-H_max
    bin_mids = (bins[1:]+bins[:-1])/2
    dbin = bins[1]-bins[0]
    anim_dict = {}
    #Rin=0.5#1#0.5
    #Rout=0.6#1.05#0.6
    R_mid = (Rin+Rout)/2
    
    for j in range(len(runfolders)):
        #==== Load histograms ====#
        try:
            #1/0
            load_array = np.load(savedir+runfolders[j].rstrip('//')+'_tGD_Hs.npy')
            anim_dict[runfolders[j].rstrip('//')+'_tGD_Hs'] = load_array

        except:
            #Compute H arrays
            num_snaps = len(glob.glob(filepath+runfolders[j]+'snap*'))
            hist_array = np.zeros((num_snaps,6+2*num_bins))

            for i in range(num_snaps):
                snapid = str(i).zfill(3)

                #==== Load sph data ====#
                load_dict    = Q.load_Gsnap(filepath,runfolders[j],snapid,'gas',bonus_arg='u',)
                headertime   = load_dict['headertime']
                N_sph        = load_dict['N_sph']       
                M_sph        = load_dict['M_sph']      
                M_star       = load_dict['M_star']
                sph_pos      = load_dict['sph_pos']
                sph_vel      = load_dict['sph_vel']
                sph_u        = load_dict['sph_A']

                #==== Load dust data ====#
                load_dust    = Q.load_Gsnap(filepath,runfolders[j],snapid,'disk',bonus_arg='u',)
                N_dust       = load_dust['N_sph']       
                M_dust       = load_dust['M_sph']      
                dust_pos     = load_dust['sph_pos']
                
                #Slice Ring
                r_gas  = np.sqrt(sph_pos[:,0]**2+sph_pos[:,1]**2)
                r_dust = np.sqrt(dust_pos[:,0]**2+dust_pos[:,1]**2)
                gas_ring  = sph_pos[(r_gas<Rout)&(r_gas>Rin),:]
                dust_ring = dust_pos[(r_dust<Rout)&(r_dust>Rin),:]

                gas_bins  = np.histogram(gas_ring[:,2],bins=bins)[0]
                dust_bins = np.histogram(dust_ring[:,2],bins=bins)[0] 
                mu_gas, std_gas   = norm.fit(gas_ring[:,2])
                mu_dust, std_dust = norm.fit(dust_ring[:,2])

                hist_array[i,0] = headertime*code_time/c.sec_per_year #Years
                hist_array[i,1] = std_gas
                hist_array[i,2] = std_dust
                hist_array[i,3] = M_sph
                hist_array[i,4] = M_dust
                hist_array[i,5] = M_star
                hist_array[i,6:6+num_bins] = gas_bins
                hist_array[i,6+num_bins:]  = dust_bins
                print 'H Gas,', std_gas
                print 'H Dust', std_dust
                
            np.save(savedir+runfolders[j].rstrip('//')+'_tGD_Hs',hist_array)
            anim_dict[runfolders[j].rstrip('//')+'_tGD_Hs'] = hist_array
        

    #=== Plotting ===#
    if anim_mode == True:
        fig1 = plt.figure(1,facecolor='white',figsize=(6,10))
        ax1  = fig1.add_axes([0.12,0.12,0.83,0.83])
        ax1.set_ylim(0,1500)

        hist_array = anim_dict[runfolders[anim_ind].rstrip('//')+'_tGD_Hs']
        num_snaps = len(hist_array[:,0])
        plot_gas,  = ax1.plot(bin_mids,hist_array[0,6:6+num_bins])
        plot_dust, = ax1.plot(bin_mids,hist_array[0,6+num_bins:])

        def animate(i):
            plot_gas.set_ydata(hist_array[i,6:6+num_bins])
            plot_dust.set_ydata(hist_array[i,6+num_bins:])
            return plot_gas, plot_dust,
        ani = animation.FuncAnimation(fig1, animate, interval=80, frames=num_snaps, blit=True, repeat=True)
        plt.show()


    #Plot Hdust_Hgas wrt simulation time
    plt.figure(2)
    #plt.semilogy()
    for i in range(len(runfolders)):
        hist_array = anim_dict[runfolders[i].rstrip('//')+'_tGD_Hs']
        HD_HG = hist_array[:,2]/hist_array[:,1]
        plt.plot(hist_array[:,0],HD_HG,color=run_cols[i],label='a = '+str(grain_as[i]))
        #plt.plot(hist_array[:,0],hist_array[:,2],color=run_cols[i],ls='--')

    #Gas histograms
    plt.figure(5,facecolor='w')
    M_sph = hist_array[0,3]
    M_star = hist_array[0,5]
    ring_vol = dbin*2*np.pi*(Rout-Rin)*R_mid
    for i in range(len(runfolders)):
        #Plot SPH vertical density profile
        hist_array = anim_dict[runfolders[i].rstrip('//')+'_tGD_Hs']
        rho_bins = hist_array[0,6:6+num_bins]*M_sph/ring_vol * code_M/code_L**3
        plt.plot(bin_mids, rho_bins/np.max(rho_bins))

        #Calculate analytic expectation values for NSG disc
        Sig    = M_sph * np.sum(hist_array[0,6:6+num_bins]) / (np.pi*(Rout**2-Rin**2))
        Om_K   = b.v_kepler(M_star*code_M,R_mid*code_L)/code_L*code_time/R_mid
        T      = b.T_profile(R=R_mid,T0=20,R0=1,power=-0.5)
        cs     = np.sqrt(c.kb*T/(c.mu*c.mp))/code_L*code_time
        rho_0  = Sig*Om_K / (np.sqrt(2*np.pi)*cs)
        H = cs/Om_K

        #Plotting expected NSG disc vertical density profile
        mu_gas, std_gas   = norm.fit(rho_bins)
        Zs   = np.linspace(-0.2,0.2,100)
        rho_gauss = rho_0*np.exp(-Zs**2/2/H**2)
        plt.plot(Zs,0.9*rho_gauss/np.max(rho_gauss))
        
    return


def reorder_IDs(anim_dict,num_folders,num_snaps):
    '''Reorder anim_dict into tracks for each particle'''
    track_dict = {}

    for i in range(num_folders):
        #Find Particle IDs
        PIDs = []
        for j in range(num_snaps):
            temp = anim_dicts[str(i)]['ID'+str(j)]
            for k in range(len(temp)):
                if temp[k] not in PIDs:
                    PIDs.append(temp[k])
        track_dict[str(i)+'_PIDs'] = PIDs

        #Reorder dictionaries based on IDs
        for k in PIDs:
            track_store = np.zeros((2,num_snaps))
            for j in range(num_snaps):
                tempZs  = anim_dicts[str(i)]['Z'+str(j)]
                tempVZs = anim_dicts[str(i)]['VZ'+str(j)]
                tempIDs = anim_dicts[str(i)]['ID'+str(j)]
                for l in range(len(tempIDs)):
                    if k == tempIDs[l]:
                        track_store[0,j] = tempZs[l]
                        track_store[1,j] = tempVZs[l]
                        
            track_dict[str(i)+'_'+str(k)] = track_store
                
    return track_dict







def settling_test(runfolders,snapid,Rin=1.0,Rout=1.02,anim_mode=False,mean_bins=False,
                  Nframes=0,rerun=False,paper_plots=False,ppi=0,write=False):
    '''Check settling velocities of dust grains'''

    #==== For non animation ====#    
    num_snaps = 1
    snap_offset = snapid
    anim_dicts = {}
    R_mid  = (Rin+Rout)/2

    #==== Animation settings =====#
    if (anim_mode == True) or (paper_plots == True):
        num_snaps = Nframes
        if Nframes == 0:
            num_snaps = len(glob.glob(filepath+runfolders[0]+'snap*')) 
        snap_offset=0
        
        
    #============ Load data ============#
    #-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-#
    
    for i in range(len(runfolders)):
        anim_dict = {}
        runfolder = runfolders[i]
        print runfolder
        time_store = np.zeros(num_snaps) 
        
        #Attempt to load dictionaries. Otherwise compute
        try:
            if rerun == True:
                1/0
            elif (anim_mode == False) & (paper_plots == False):
                1/0
            anim_dict = np.load(savedir+runfolder.rstrip('//')+'_animdict.npy')
        except:
            for j in range(num_snaps):
                snapid = str(j+snap_offset).zfill(3)
                print filepath+runfolder
                print snapid
                #==== Load sph data ====#
                load_dict    = Q.load_Gsnap(filepath,runfolder,snapid,'gas',bonus_arg='u',)
                headertime   = load_dict['headertime']
                N_sph        = load_dict['N_sph']       
                M_sph        = load_dict['M_sph']      
                M_star       = load_dict['M_star']
                sph_pos      = load_dict['sph_pos']
                sph_vel      = load_dict['sph_vel']
                sph_u        = load_dict['sph_A']

                #==== Load dust data ====#
                load_dust    = Q.load_Gsnap(filepath,runfolder,snapid,'disk',bonus_arg='u',)
                N_dust       = load_dust['N_sph']       
                M_dust       = load_dust['M_sph']      
                dust_pos     = load_dust['sph_pos']
                dust_vel     = load_dust['sph_vel']
                dust_IDs     = load_dust['sph_ID']

                #Slice Ring
                r_gas  = np.sqrt(sph_pos[:,0]**2+sph_pos[:,1]**2)
                gas_ring    = sph_pos[(r_gas<Rout)&(r_gas>Rin),:]
                gas_ringv   = sph_vel[(r_gas<Rout)&(r_gas>Rin),:]
                
                if j == 0:
                    r_dust = np.sqrt(dust_pos[:,0]**2+dust_pos[:,1]**2)
                    test_dust = dust_pos[(r_dust<Rout)&(r_dust>Rin),2]
                    test_dustv = dust_vel[(r_dust<Rout)&(r_dust>Rin),2]
                    test_dustIDs = dust_IDs[(r_dust<Rout)&(r_dust>Rin)]
                    ini_dustIDs = test_dustIDs.copy()

                if j != 0:
                    test_dust    = []
                    test_dustv   = []
                    test_dustIDs = []
                    #Only choose particles that are in initial ring
                    for k in range(len(dust_IDs)):
                        if dust_IDs[k] in ini_dustIDs:
                            test_dust.append(dust_pos[k,2])
                            test_dustv.append(dust_vel[k,2])
                            test_dustIDs.append(dust_IDs[k])
                    test_dust    = np.array(test_dust)
                    test_dustv   = np.array(test_dustv)
                    test_dustIDs = np.array(test_dustIDs)

                #Settling analysis
                Sig    = M_sph * len(gas_ring[:,0]) / (np.pi*(Rout**2-Rin**2))
                Om_K   = b.v_kepler(M_star*code_M,R_mid*code_L)/code_L*code_time/R_mid
                T      = b.T_profile(R=R_mid,T0=20,R0=1,power=-0.5)
                cs     = np.sqrt(c.kb*T/(c.mu*c.mp))/code_L*code_time
                rho_0  = Sig*Om_K / (np.sqrt(2*np.pi)*cs)            
                a      = grain_as[i]/code_L
                rho_a  = 5 /code_M*code_L**3
                print 'Rho_0', rho_0*code_M/code_L**3
                t_stop = a*rho_a /rho_0/cs *np.sqrt(np.pi/8)
                print 'tau', t_stop*Om_K


                Zbins   = np.linspace(-0.2,0.2,51)
                Zbin_mids = (Zbins[1:]+Zbins[:-1])/2
                dZbin = Zbins[1]-Zbins[0]
                Z_Rs = Zbin_mids/R_mid
                VZ_an = -Om_K**2*t_stop*Zbin_mids * np.exp(Zbin_mids**2/2 * (Om_K/cs)**2)

                #Calculate rho from sph
                gas_bins = np.histogram(gas_ring[:,2],bins=Zbins)[0]
                ring_vol = dZbin*(Rout-Rin)*2*np.pi*R_mid
                rho_bins = M_sph*gas_bins/ring_vol
                VZ_Gad = -Om_K**2*a*rho_a*Zbin_mids / (cs*rho_bins) * np.sqrt(np.pi/8)
                
                
                #Store information in dictionaries for later animation
                anim_dict['Z'+str(j)]     = test_dust[:]/R_mid
                anim_dict['VZ'+str(j)]    = test_dustv[:]
                anim_dict['ID'+str(j)]    = test_dustIDs
                anim_dict['Zgas'+str(j)]  = gas_ring[:,2]/R_mid
                anim_dict['VZgas'+str(j)] = gas_ringv[:,2]
                time_store[j] = headertime*code_time/c.sec_per_year
                if j == 0:
                    anim_dict['a'] = grain_as[i]
                    anim_dict['tau']  = t_stop*Om_K
                    anim_dict['an_Z_R'] = Z_Rs
                    anim_dict['an_VZs'] = VZ_an
                    anim_dict['VZ_Gad'] = VZ_Gad
                    
            #Save anim dictionaries        
            if anim_mode == True:
                anim_dict['time'] = time_store
                np.save(savedir+runfolder.rstrip("//")+'_animdict.npy', anim_dict)
            else:
                print anim_dict['a']
                anim_dicts[str(i)] = anim_dict

                

    #============== Plotting =================#
    #-----------------------------------------#

    bins = np.linspace(-0.2,0.2,51) #in H/R
    bin_mids = (bins[1:]+bins[:-1])/2

    #==== Non Animation plotting ====#
    #Establish figures 
    fig3 = plt.figure(3,facecolor='white',figsize=(6,10))
    ax1  = fig3.add_axes([0.12,0.12,0.83,0.83])
    ax1.set_xlabel('Z/R')
    ax1.set_ylabel('VZ Dust []')
    ax1.set_ylim(-0.15,0.15)
    
    for i in range(len(runfolders)):
        if (anim_mode == False) and (paper_plots == False):

            print 'i',i
            dustZ   = anim_dicts[str(i)]['Z0']
            dustVZ  = anim_dicts[str(i)]['VZ0']
            gasZ    = anim_dicts[str(i)]['Zgas0']
            gasVZ   = anim_dicts[str(i)]['VZgas0']
            an_Z_Rs = anim_dicts[str(i)]['an_Z_R']
            an_VZs  = anim_dicts[str(i)]['an_VZs']
            tau     = anim_dicts[str(i)]['tau']
            
            #Plot analytic curves and text
            ax1.plot(an_Z_Rs,an_VZs,color=run_cols[i],label='a = '+str(grain_as[i])+'[cm]\n'+r'$\tau$ = {:.4f}'.format(tau))
            if mean_bins == True:
                mean_VZ, std_VZ = b.calc_binned_data(dustVZ,dustZ,bins=bins)
                ax1.errorbar(bin_mids,mean_VZ,yerr=std_VZ,color=run_cols[i],fmt='o',mew=0)
            else:
                ax1.scatter(dustZ,dustVZ,color=run_cols[i],alpha=0.6)
                ax1.scatter(gasZ,gasVZ,color=run_cols[i],alpha=0.1,s=2)

    #==== Paper plots ====#
    if paper_plots == True:
        bins = np.linspace(-0.2,0.2,31) #in H/R
        bin_mids = (bins[1:]+bins[:-1])/2
        vK = b.v_kepler(c.Msol,R_mid*code_L)/code_L*code_time
    
        #Build Figure
        plt.close()
        fig3 = plt.figure(ppi,facecolor='white',figsize=(3,8))
        ax1  = fig3.add_axes([0.29,0.66,0.68,0.29])
        ax2  = fig3.add_axes([0.29,0.36,0.68,0.29],sharex=ax1)
        ax3  = fig3.add_axes([0.29,0.06,0.68,0.29],sharex=ax1)
        axes = [ax1,ax2,ax3]
        ax1.set_xlim(-0.25,0.25)
        ax3.set_xlabel('Z/R')
        ax2.set_ylabel('100xVZ / VK')
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        
        Times = [50,100,150]
        anim_dicts[str(ppi)] = np.load(savedir+runfolders[ppi].rstrip("//")+'_animdict.npy').item()
        time_store = anim_dicts[str(ppi)]['time']

        ax1.text(0.06,1.1,'a = '+str(grain_as[ppi])+' cm', transform=ax1.transAxes)
        for i in range(len(Times)):
            ti  = np.argmin((time_store-Times[i])**2)
            axes[i].text(0.65,0.9,'{:.0f}'.format(time_store[ti]) + ' Yrs', transform=axes[i].transAxes)

            dustZ   = anim_dicts[str(ppi)]['Z'+str(ti)]
            dustVZ  = anim_dicts[str(ppi)]['VZ'+str(ti)]/vK*100
            gasZ    = anim_dicts[str(ppi)]['Zgas'+str(ti)]
            gasVZ   = anim_dicts[str(ppi)]['VZgas'+str(ti)]/vK*100
            an_Z_Rs = anim_dicts[str(ppi)]['an_Z_R']
            an_VZs  = anim_dicts[str(ppi)]['an_VZs']/vK*100
            tau     = anim_dicts[str(ppi)]['tau']
            if i == 0:
                ax1.text(0.06,1.02,r'$\tau$ = {:.4f}'.format(tau),transform=ax1.transAxes)

            axes[i].plot(an_Z_Rs,an_VZs,color=run_cols[ppi],label='a = '+str(grain_as[ppi])+'[cm]\n'+r'$\tau$ = {:.4f}'.format(tau))
            mean_VZ, std_VZ = b.calc_binned_data(dustVZ,dustZ,bins=bins)
            axes[i].errorbar(bin_mids,mean_VZ,yerr=std_VZ,color=run_cols[ppi],fmt='o',ms=3,mew=1,mec=run_cols[ppi],alpha=0.5)

            #Set y scales
            mean_VZ[np.isnan(mean_VZ)] = 0
            std_VZ[np.isnan(std_VZ)]   = 0
            ylim = 1.2*np.max(mean_VZ+std_VZ)
            print ylim
            axes[i].set_ylim(-ylim,ylim)


    
    #==== Animation plotting ====#
    if anim_mode == True:
        #Multiplot option for Figures
        if len(runfolders) != 1:
            plt.close()
            fig3 = plt.figure(3,facecolor='white',figsize=(10,10))
            ax1  = fig3.add_axes([0.1,0.55,0.38,0.38])
            ax2  = fig3.add_axes([0.6,0.55,0.38,0.38])
            ax3  = fig3.add_axes([0.1,0.1,0.38,0.38])
            ax4  = fig3.add_axes([0.6,0.1,0.38,0.38])
            axes = [ax1,ax2,ax3,ax4]
            for j in axes:
                j.set_xlabel('Z/R')
                j.set_ylabel('VZ Dust []')
                j.set_ylim(-0.15,0.15)
            plt.yscale('symlog')

        else:
            axes = [ax1,ax1,ax1,ax1]
            
        scatter_dict = {}
        for i in range(len(runfolders)):
            
            #Load animation data
            anim_dicts[str(i)] = np.load(savedir+runfolders[i].rstrip("//")+'_animdict.npy').item()
            if i == 0:
                time_store = anim_dicts['0']['time']
                timetext = ax1.text(0.06,0.91,'Time: {:.2f}'.format(time_store[0]) + ' Years', transform=ax1.transAxes)

            if mean_bins == True:
                mean_VZ, std_VZ = b.calc_binned_data(anim_dicts[str(i)]['VZ'+str(0)],anim_dicts[str(i)]['Z'+str(0)],bins=bins)
                line, ( bottoms, tops), verts = axes[i].errorbar(bin_mids,mean_VZ,yerr=std_VZ,color=run_cols[i],fmt='o',ms=4,mew=1,mec=run_cols[i],alpha=0.5)
                verts[0].remove() # remove the vertical lines
                scatter_dict[str(i)+'line']    = line
                scatter_dict[str(i)+'bottoms'] = bottoms
                scatter_dict[str(i)+'tops']    = tops

            else:
                scatter_dict[str(i)] = axes[i].scatter(anim_dicts[str(i)]['Z'+str(0)],anim_dicts[str(i)]['VZ'+str(0)],color=run_cols[i],alpha=0.1)

            ylim = 1.2*np.max(anim_dicts[str(i)]['VZ'+str(6)])
            axes[i].set_ylim(-ylim,ylim)
            
            axes[i].plot(anim_dicts[str(i)]['an_Z_R'],anim_dicts[str(i)]['an_VZs'],color=run_cols[i],label='a = '+str(anim_dicts[str(i)]['a'])+' cm')
            axes[i].plot(anim_dicts[str(i)]['an_Z_R'],anim_dicts[str(i)]['VZ_Gad'],color=run_cols[i],ls='--')
            axes[i].legend(frameon=False)

            
        def animate(j):
            output = []
            for i in range(len(runfolders)):
                #Animate timer
                if i == 0:
                    timetext.set_text('Time: {:.2f}'.format(time_store[j]) + ' Years')
                    output.append(timetext)

                if mean_bins == True:
                    mean_VZ, std_VZ = b.calc_binned_data(anim_dicts[str(i)]['VZ'+str(j)],anim_dicts[str(i)]['Z'+str(j)],bins=bins)
                    scatter_dict[str(i)+'line'].set_ydata(mean_VZ)
                    scatter_dict[str(i)+'bottoms'].set_ydata(mean_VZ - std_VZ)
                    scatter_dict[str(i)+'tops'].set_ydata(mean_VZ + std_VZ)
                    output.append(scatter_dict[str(i)+'line'] )
                    output.append(scatter_dict[str(i)+'bottoms'])
                    output.append(scatter_dict[str(i)+'tops'])

                else:
                    scatter_dict[str(i)].set_offsets(np.c_[anim_dicts[str(i)]['Z'+str(j)],anim_dicts[str(i)]['VZ'+str(j)]])
                    output.append(scatter_dict[str(i)])
            return output
        
        ani = animation.FuncAnimation(fig3, animate, interval=90, frames=num_snaps, blit=False, repeat=True)

        if write == True:
            writer = animation.writers['ffmpeg'](fps=5)
            savename = runfols
            if mean_bins == True:
                savename = runfols + '_meanbins'
            ani.save(savename+'.mp4',writer=writer,dpi=200)
        plt.show()

        
    '''
    if allplot_mode == True:            
        track_dicts = reorder_IDs(anim_dicts,len(runfolders),num_snaps)
        for i in range(len(runfolders)):
            PIDs = track_dicts[str(i)+'_PIDs']
            for j in range(len(PIDs)):
                plotZs,plotVZs  = track_dicts[str(i)+'_'+str(PIDs[j])]
                plotZs[plotZs==0]   = np.nan
                plotVZs[plotVZs==0] = np.nan
                
                if abs(plotZs[0]) > 0.15:
                    axes[i].plot(plotZs,plotVZs,alpha=0.3,color=run_cols[i])
                     
            axes[i].text(0.72,0.9,'a = '+str(grain_as[i]), transform=axes[i].transAxes)
    '''    
    return



def ring_test(runid,it=0):
    '''Test to examine how a ring of dust spreads out in a gas disc. Testing SPH velocity field noise'''
    snapids = ['005','006']
    runfolder = runfolders[runid]

    plt.figure(7,facecolor='w')
    vr_bins = np.linspace(-0.03,0.03,100)
    headertimes = []

    for i in range(len(snapids)):
        
        #==== Load sph data ====#
        load_dict    = Q.load_Gsnap(filepath,runfolder,snapids[i],'gas',bonus_arg='u',)
        headertime   = load_dict['headertime']
        N_sph        = load_dict['N_sph']       
        M_sph        = load_dict['M_sph']      
        M_star       = load_dict['M_star']
        sph_pos      = load_dict['sph_pos']
        sph_vel      = load_dict['sph_vel']
        sph_u        = load_dict['sph_A']
        sph_r        = np.sqrt(sph_pos[:,0]**2+sph_pos[:,1]**2)

        #==== Load dust data ====#
        load_dust    = Q.load_Gsnap(filepath,runfolder,snapids[i],'disk',bonus_arg='u',)
        N_dust       = load_dust['N_sph']       
        M_dust       = load_dust['M_sph']      
        dust_pos     = load_dust['sph_pos']
        dust_vel     = load_dust['sph_vel']
        dust_IDs     = load_dust['sph_ID']
        dust_r       = np.sqrt(dust_pos[:,0]**2+dust_pos[:,1]**2)
        dust_vrs     = np.zeros(N_dust)
        gas_vrs      = np.zeros(N_dust)
        headertimes.append(headertime)

        if i == 0:
            r0s = dust_r 
        elif i == len(snapids)-1:
            r1s = dust_r
            vr_bulk = np.mean(r1s-r0s)/(headertimes[-1]-headertimes[0])
            plt.axvline(vr_bulk,lw=2,color=run_cols[it])

            #Find Weidenschilling analytic prediction
            code_a = grain_as[it]/code_L
            code_rhoa = 5*code_L**3/code_M
            
            meanr = np.mean(r1s)
            dr = 0.01 #1AU
            sph_ring = sph_r[np.sqrt((sph_r-meanr)**2) < dr]
            u_ring   = np.mean(sph_u[np.sqrt((sph_r-meanr)**2) < dr])
            
            Gas_sig = len(sph_ring)*M_sph / (2*np.pi*meanr*2*dr)
            Om_K_r = np.sqrt(M_star*code_G/meanr**3)
            rtau = code_a*code_rhoa * np.pi / (2*Gas_sig*Om_K_r)
            print 'rtau',rtau
            print 'meanr', meanr
            eta = 11/4*u_ring*(c.gamma_mono-1)/(Om_K_r*meanr)**2

            Wvr = -eta*Om_K_r*meanr / (rtau+1/rtau)
            plt.axvline(Wvr,lw=2,color=run_cols[it],ls='--')

            
        for j in range(N_dust):
            dust_vrs[j] = np.dot(dust_vel[j],dust_pos[j]) / dust_r[j]
        for k in range(N_sph):
            gas_vrs[k] = np.dot(sph_vel[k],sph_pos[k]) / sph_r[k]
        if i != 0:
            plt.hist(dust_vrs,alpha=0.3,bins=vr_bins,lw=0,color=run_cols[it],label=r'$\tau$={:.2}'.format(rtau))
            plt.hist(gas_vrs,alpha=0.5,bins=vr_bins,lw=0,color=run_cols[it])

        plt.legend(frameon=False)
    return

        
def weidenschilling_r_mig(snapid,runid,bonus_plots=False,it=0,Rin=0.44,Rout=0.46):
    '''Dust behaviour tests. Youdin, Weidenshcilling'''
    runfolder = runfolders[runid]
    snapid = str(snapid).zfill(3)
    print runfolder

    #==== Load sph data ====#
    load_dict    = Q.load_Gsnap(filepath,runfolder,snapid,'gas',bonus_arg='u',)
    headertime   = load_dict['headertime']
    print headertime*code_time/c.sec_per_year, 'Years'
    M_sph        = load_dict['M_sph']      
    M_star       = load_dict['M_star']
    sph_pos      = load_dict['sph_pos']
    sph_vel      = load_dict['sph_vel']
    sph_u        = load_dict['sph_A']
    sph_h        = load_dict['sph_h']

    #==== Load dust data ====#
    load_dust    = Q.load_Gsnap(filepath,runfolder,snapid,'disk',bonus_arg='u',)
    M_dust       = load_dust['M_sph']      
    dust_pos     = load_dust['sph_pos']
    dust_vel     = load_dust['sph_vel']
    if runfols == 'SWD_2017':
        dust_a   = load_dust['dust_M']
    else:
        dust_a       = load_dust['dust_a']

    
    #==== Isolate ring data ====#
    R_mid = (Rin+Rout)/2

    r_gas        = np.sqrt(sph_pos[:,0]**2+sph_pos[:,1]**2)
    M_int        = M_sph*len(r_gas[r_gas<R_mid])
    gas_ring     = sph_pos[(r_gas<Rout)&(r_gas>Rin),:]
    r_ring       = r_gas[(r_gas<Rout)&(r_gas>Rin)]
    z_ring       = sph_pos[(r_gas<Rout)&(r_gas>Rin),2]
    v_ring       = sph_vel[(r_gas<Rout)&(r_gas>Rin),:]
    u_ring       = sph_u[(r_gas<Rout)&(r_gas>Rin)]
    mean_h       = np.mean(sph_h[(r_gas<Rout)&(r_gas>Rin)])
    ring_z_mu, ring_H = norm.fit(z_ring)
    vz_ring      = v_ring[:,2]
    vr_ring, vazi_ring  = b.v_r_azi(gas_ring,v_ring)
    vK_ring = b.v_kepler((M_star+M_int)*code_M,r_ring*code_L)*code_time/code_L
    vazi_ring_rel = vazi_ring - vK_ring

    
    #Gas r bin vel
    vr_gas,vazi_gas = b.v_r_azi(sph_pos,sph_vel)
    r_bins  = np.linspace(0.1,2.0,100)
    vK_bins = b.v_kepler(M_star*code_M,r_bins*code_L)*code_time/code_L
    vr_bins,vr_stds  = b.calc_binned_data(vr_gas,r_gas,r_bins)
    vr_gas_Rmid = vr_bins[np.argmin((r_bins-R_mid)**2)]
    
    
    print 'Ring H', ring_H
    
    #==== Calculate dust ring data ====#
    r_dust          = np.sqrt(dust_pos[:,0]**2+dust_pos[:,1]**2)
    dust_ring       = dust_pos[(r_dust<Rout)&(r_dust>Rin),:]
    r_dust_ring     = r_dust[(r_dust<Rout)&(r_dust>Rin)]
    v_dust_ring     = dust_vel[(r_dust<Rout)&(r_dust>Rin),:]
    vK_dust_ring    = b.v_kepler((M_star+M_int)*code_M,r_dust_ring*code_L)*code_time/code_L
    
    dust_ring_a     = dust_a[(r_dust<Rout)&(r_dust>Rin)]
    vr_dust_ring,vazi_dust_ring = b.v_r_azi(dust_ring,v_dust_ring)
    vazi_dust_ring_rel = vazi_dust_ring - vK_dust_ring
    
    #==== Dynamical time ====#
    t_dyn = np.sqrt(R_mid**3/ (M_star*code_G))
    t_sim = headertime/t_dyn

    print 'old mu2_vz', np.mean(vz_ring**2,axis=0)
    print 'old mu2_vr', np.mean(vr_ring**2,axis=0)

    #==== Radial and vertical data ====#
    vz_ring = b.Gadget_smooth(gas_ring,vz_ring,M_sph,gas_ring)

    vr_ring = b.Gadget_smooth(gas_ring,vr_ring,M_sph,gas_ring)
    print 'vr', np.shape(vr_ring)
    print 'z', np.shape(z_ring)
    
    #vz_ring = vz_ring[abs(gas_ring[:,2]<0.1*ring_H)]
    mu2_vzgas   = np.mean(vz_ring**2)
    mu2_vrgas   = np.mean(vr_ring**2)
    mu2_vazigas = np.mean(vazi_ring_rel**2)
    print 'mu2_vz, mu2_vr', mu2_vzgas, mu2_vrgas
    
    vz2_dust = v_dust_ring[:,2]**2
    vr2_dust = vr_dust_ring**2


    
    
    #==== Analytic quantities for Youdin (2007) dispersion and Weidenschilling ====#
    a      = dust_ring_a/code_L
    rho_a  = 1/code_M*code_L**3 *3#5
    Sig    = M_sph * len(gas_ring[:,0]) / (np.pi*(Rout**2-Rin**2))
    Om_K   = b.v_kepler((M_star+M_int)*code_M,R_mid*code_L)/code_L*code_time/R_mid
    T      = b.T_profile(R=R_mid,T0=20,R0=1,power=-0.5)
    cs     = np.sqrt(c.kb*T/(c.mu*c.mp))/code_L*code_time
    H      = cs/Om_K
    rho_0  = Sig*Om_K / (np.sqrt(2*np.pi)*cs)
    eta    = 11/4 * cs**2/(Om_K*R_mid)**2

    #==== Alpha Calculations ====#
    alpha_AV = aavs[runid]
    alpha  = 0.1*alpha_AV * mean_h/2/H #Lodato & Price 2010

    #==== Stopping Times ====#
    force_fac = 1 
    if runfols == 'SWD_2017':
        force_fac = 8e5 #Sergei enhanced drag forces to compensate for low disc mass
    t_stop   = a*rho_a /rho_0/cs *np.sqrt(np.pi/8) /force_fac
    tau      = t_stop*Om_K
    taus     = np.logspace(np.log10(np.min(tau)),np.log10(np.max(tau)),50)
    tau_eddyz = alpha*cs**2/mu2_vzgas
    tau_eddyr = alpha*cs**2/mu2_vrgas
    print 'tau eddy z, tau eddy r', tau_eddyz, tau_eddyr
    Stz       = taus/tau_eddyz
    Str       = taus/tau_eddyr

    
    #Vertical analytic formula
    an_vp_vgz = 1/ (1 + Stz*(1+tau_eddyz**2)) #youdin20
    epsilon = 1 + taus*tau_eddyz**2/(taus+tau_eddyz)
    HdHg = np.sqrt(alpha/(alpha+taus))
    HdHg_youdin28 =  np.sqrt(alpha/(alpha+taus)) * epsilon**-0.5 #youdin28

    #Radial analytic formula
    d2 = (1+taus**2)* (taus**2 + (1 + Str)**2)
    termA = 1 + Str*(1+taus**2/2)
    termB = 2*taus**2 * (2+Str)*mu2_vazigas/mu2_vrgas
    termC = 2*taus*(2+Str)*alpha*cs**2/mu2_vrgas
    an_vp_vgr = (termA + termB + termC)/d2 #youdin33a

    vd_vgz = (vz2_dust)/mu2_vzgas
    vd_vgr = (vr2_dust)/mu2_vrgas

    mean_vd_vgz,sig_vd_vgz = b.calc_binned_data(vd_vgz,tau,taus)
    mean_vd_vgr,sig_vd_vgr = b.calc_binned_data(vd_vgr,tau,taus)

    vr_gas = -3/2 * alpha*cs*H/R_mid #does it?
    print 'gas Vran, gas Vrsim', vr_gas, vr_gas_Rmid
    Wei_vr_vK = (vr_gas/(taus*Om_K*R_mid)-eta) / (taus + 1/taus)

    
    #======== Plotting ========#

    scatcol = '#ffaa00'
    ancol = '#00ddff'
    datacol = '#660099'
    ms = 4
    alph = 0.2

    #==== VZ dispersion plot ====#
    fig1 = plt.figure(1+it,facecolor='white',figsize=(6,6))
    ax1  = fig1.add_axes([0.18,0.1,0.78,0.85])
    plt.ylabel(r'sqrt($<VZ_{Dust}^2>/<VZ_{gas}^2>$)')
    plt.xlabel(r'Stokes Number')
    plt.semilogx()
    plt.semilogy()
    plt.xlim(np.min(tau)*0.9,1.1*np.max(tau))

    plt.scatter(tau,np.sqrt(vd_vgz),color=scatcol,alpha=alph,s=ms,label='Dust')
    plt.plot(taus[:-1],np.sqrt(mean_vd_vgz),color=datacol,label='Mean dust')
    plt.plot(taus[:-1],np.sqrt(mean_vd_vgz-sig_vd_vgz),ls='--',color=datacol) 
    plt.plot(taus[:-1],np.sqrt(mean_vd_vgz+sig_vd_vgz),ls='--',color=datacol)
    #plt.vlines(t_sim,1e-6,200,color=ancol,linestyle='--') #Dynamical times
    plt.plot(taus,np.sqrt(an_vp_vgz),color=ancol,label='Analytic result') #Analytic
    plt.legend(frameon=False)
    
    #==== VR dispersion plot ====#
    fig2 = plt.figure(2+it,facecolor='white',figsize=(6,6))
    ax1  = fig2.add_axes([0.18,0.1,0.78,0.85])
    plt.ylabel(r'sqrt($<VR_{Dust}^2>/<VR_{gas}^2>$)')
    plt.xlabel(r'Stokes Number')
    plt.semilogx()
    plt.semilogy()
    plt.xlim(np.min(tau)*0.9,1.1*np.max(tau))

    plt.scatter(tau,np.sqrt(vd_vgr),color=scatcol,alpha=alph,s=ms,label='Dust')
    plt.plot(taus[:-1],np.sqrt(mean_vd_vgr),color=datacol,label='Mean dust')
    plt.plot(taus[:-1],np.sqrt(mean_vd_vgr-sig_vd_vgr),ls='--',color=datacol) 
    plt.plot(taus[:-1],np.sqrt(mean_vd_vgr+sig_vd_vgr),ls='--',color=datacol) 
    #plt.vlines(t_sim,1e-6,200,color=ancol,linestyle='--') #Dynamical times
    plt.plot(taus,np.sqrt(an_vp_vgr),color=ancol,label='Analytic Result')
    plt.legend(frameon=False)

    #==== Bonus plots ====#
    if bonus_plots == True:
        
        #==== Z coord plot ====#
        ZHbins = np.linspace(-3,3,100)
        dZH = ZHbins[1]-ZHbins[0]
        rho_gauss = rho_0*np.exp(-ZHbins[:-1]**2/2)
        gas_profile = np.histogram(gas_ring[:,2]/ring_H,bins=ZHbins)[0]
        gas_rhoZ = gas_profile * M_sph / (dZH*H*(Rout-Rin)*2*np.pi*R_mid)

        ZH_data = dust_ring[:,2]/ring_H
        ZH_data2 = ZH_data**2
        ZHmeans, tmp   = b.calc_binned_data(ZH_data,tau,taus)
        ZH2means, tmp2 = b.calc_binned_data(ZH_data2,tau,taus)
        ZHstds = np.sqrt(ZH2means-ZHmeans**2)
        ZHstd_max = ZHmeans+ZHstds
        ZHstd_min = ZHmeans-ZHstds

        fig3 = plt.figure(3+it,facecolor='white',figsize=(6,6))
        ax1  = fig3.add_axes([0.2,0.1,0.7,0.85])
        ax2  = fig3.add_axes([0.1,0.1,0.1,0.85],sharey=ax1)
        
        #Plot Data
        ax1.scatter(tau,ZH_data,color=scatcol,alpha=alph,s=ms)
        ax1.plot(taus[:-1],ZHstd_max,color=datacol)
        ax1.plot(taus[:-1],ZHstd_min,color=datacol)
        ax1.plot(taus,-HdHg,ls=':',color=datacol)
        ax1.plot(taus,HdHg,ls=':',color=datacol)
        ax1.plot(taus,-HdHg_youdin28,ls='--',color=datacol)
        ax1.plot(taus,HdHg_youdin28,ls='--',color=datacol)
        ax1.set_xlim(np.min(tau)*0.9,1.1*np.max(tau))

        #Plot gas profile
        ax2.plot(gas_rhoZ,ZHbins[:-1],color=datacol)
        ax2.plot(rho_gauss,ZHbins[:-1],color=ancol)
        ax1.semilogx()
        ax1.set_xlim(np.min(tau)*0.9,1.1*np.max(tau))
        ax1.set_ylabel('Z/H')
        ax1.set_xlabel(r'Stokes Number')
        ax1.set_ylim(-2.5,2.5)


        #==== Weidenschilling VR_VK plots ====#
        fig4 = plt.figure(4+it,facecolor='white',figsize=(6,6))
        ax1  = fig4.add_axes([0.18,0.1,0.78,0.85])
        vr_vK = -vr_dust_ring/vK_dust_ring
        plt.scatter(tau,vr_vK,color=scatcol,s=ms,alpha=alph,label='Dust')
        ax1.set_xlim(np.min(tau)*0.9,1.1*np.max(tau))
        plt.semilogx()
        plt.semilogy()
        #plt.yscale('symlog')
        plt.ylabel(r'V$_R$/V$_K$')
        plt.xlabel(r'Stokes Number')

        vr_vK_means, tmp   = b.calc_binned_data(vr_vK,tau,taus)
        vr_vK2_means, tmp2 = b.calc_binned_data(vr_vK**2,tau,taus)
        std_vr_vK = np.sqrt(vr_vK2_means-vr_vK_means**2)
        plt.plot(taus[:-1],vr_vK_means,color=datacol,label='Dust mean')
        plt.plot(taus[:-1],vr_vK_means-std_vr_vK,color=datacol,ls='--')
        plt.plot(taus[:-1],vr_vK_means+std_vr_vK,color=datacol,ls='--')
        plt.plot(taus,-Wei_vr_vK,color=ancol,label='Weidenschilling solution')
        gas_disp = np.sqrt(mu2_vrgas)/Om_K/R_mid
        plt.hlines(gas_disp,1e-3,1e3,color=ancol,linestyle='--',label=r'Gas VR$_{RMS}$') 
        plt.legend(frameon=False,loc=2)

        #==== VZ/VK plots ====#
        fig5 = plt.figure(5+it,facecolor='white',figsize=(6,6))
        ax1  = fig5.add_axes([0.18,0.1,0.78,0.85])
        vz_vK = abs(v_dust_ring[:,2])/vK_dust_ring
        plt.scatter(tau,vz_vK,color=scatcol,s=ms,alpha=alph,label='Dust')
        ax1.set_xlim(np.min(tau)*0.9,1.1*np.max(tau))
        plt.semilogx()
        plt.semilogy()
        #plt.yscale('symlog')
        plt.ylabel(r' |V$_Z$|/V$_K$')
        plt.xlabel(r'Stokes Number')

        vz_vK_means, tmp   = b.calc_binned_data(vz_vK,tau,taus)
        vz_vK2_means, tmp2 = b.calc_binned_data(vz_vK**2,tau,taus)
        std_vz_vK = np.sqrt(vz_vK2_means-vz_vK_means**2)
        plt.plot(taus[:-1],vz_vK_means,color=datacol,label='Dust mean')
        plt.plot(taus[:-1],vz_vK_means-std_vz_vK,color=datacol,ls='--')
        plt.plot(taus[:-1],vz_vK_means+std_vz_vK,color=datacol,ls='--')
        plt.hlines(gas_disp,1e-3,1e3,color=ancol,linestyle='--',label=r'Gas VR$_{RMS}$')
        plt.legend(frameon=False)

        '''
        plt.figure(13)
        plt.hist(np.log10(dust_a),bins=200,color='#ffaa00',alpha=0.5)
        plt.hist(np.log10(dust_ring_a),bins=200,color='#ee00aa',alpha=0.5)

        plt.figure(17)
        plt.plot(r_bins[:-1],vr_bins/vK_bins[:-1])
        '''
    return





def alpha_SS(snapid,runid):
    runfolder = runfolders[runid]
    snapid = str(snapid).zfill(3)

    #==== Load sph data ====#
    load_dict    = Q.load_Gsnap(filepath,runfolder,snapid,'gas',bonus_arg='u',)
    headertime   = load_dict['headertime']
    M_sph        = load_dict['M_sph']      
    M_star       = load_dict['M_star']
    sph_pos      = load_dict['sph_pos']
    sph_vel      = load_dict['sph_vel']
    sph_u        = load_dict['sph_A']
    sph_h        = load_dict['sph_h']

    sph_r  = np.sqrt(sph_pos[:,0]**2 + sph_pos[:,1]**2)
    sort_r = np.sort(sph_r)
    rin    = sort_r[int(len(sph_r)*0.001)]
    rout   = sort_r[int(len(sph_r)*0.99)]
    r_bins = np.linspace(rin,rout,201)
    r_bin_mids = (r_bins[1:]+r_bins[:-1])/2

    H68s      = b.calc_binned_data(sph_pos[:,2],sph_r,r_bins,H68_mode=True)
    hs,h_stds = b.calc_binned_data(sph_h,sph_r,r_bins)
    alpha_AV  = aavs[runid]
    alpha_SS = alpha_AV/10 *hs/H68s
    
    plt.figure(14)
    plt.plot(r_bin_mids*100,alpha_SS,color='b')
    plt.plot(r_bin_mids*100,hs/H68s,color='g')
    plt.xlabel('R [AU]')
    plt.ylabel('alpha_SS')


    



def cross_section(runfolders,anim_mode=False,anim_ind=0,rerun=False,vr_mode=False,norm_mode=False):
    '''Animate and plot Sigma for gas and an array of dust sizes'''
    if np.shape(runfolders) == ():
        runfolders = [runfolders]
        
    #==== Arrange data binning ==#
    num_Rbins  = 200
    num_abins = 5
    
    Rbins = np.linspace(0.01,2,num_Rbins+1)
    Rbin_mids = (Rbins[1:]+Rbins[:-1])/2
    dRbin = Rbins[1]-Rbins[0]
    Rbin_areas = Rbin_mids*2*np.pi*dRbin

    amin = -4
    amax = 2
    abins = np.logspace(amin,amax,num_abins+1)
    abin_mids = (abins[1:]+abins[:-1])/2

    anim_dict = {}

    for runid in range(len(runfolders)):
        #==== Load histograms ====#
        try:
            if rerun == True:
                1/0
            anim_dict[str(runid)] = np.load(savedir+runfolders[runid].rstrip('//')+'_dust_Sigmas.npy')
            print 'Data loaded!'
        except:
            #Compute H arrays
            print 'No savefile. Need to compute'
            num_snaps = len(glob.glob(filepath+runfolders[runid]+'snap*'))
            hist_array = np.zeros((num_snaps,5+num_abins,1+num_Rbins))

            for i in range(num_snaps):
                print 'Iteration: ', i
                snapid = str(i).zfill(3)

                #==== Load sph data ====#
                load_dict    = Q.load_Gsnap(filepath,runfolders[runid],snapid,'gas',bonus_arg='u')
                headertime   = load_dict['headertime']
                N_sph        = load_dict['N_sph']       
                M_sph        = load_dict['M_sph']      
                M_star       = load_dict['M_star']
                sph_pos      = load_dict['sph_pos']
                sph_vel      = load_dict['sph_vel']
                sph_u        = load_dict['sph_A']

                try:
                    #==== Load dust data ====#
                    load_dust    = Q.load_Gsnap(filepath,runfolders[runid],snapid,'disk')
                    N_dust       = load_dust['N_sph']       
                    M_dust       = load_dust['M_sph']      
                    dust_pos     = load_dust['sph_pos']
                    dust_a       = load_dust['dust_a']
                    print dust_a
                    print len(dust_a)
                except:
                    N_dust, M_dust, dust_a = 1,0,0
                    dust_pos = np.array([[0,0,0]])

                #==== Count and bin gas and dust radially ====#
                r_gas  = np.sqrt(sph_pos[:,0]**2+sph_pos[:,1]**2)
                r_dust = np.sqrt(dust_pos[:,0]**2+dust_pos[:,1]**2)
                gas_count = np.histogram(r_gas,Rbins)[0]
                gas_Sig   = gas_count*M_sph/Rbin_areas
                gas_u     = b.calc_binned_data(sph_u,r_gas,Rbins)[0]
                a_ids = np.digitize(dust_a,abins)-1

                for j in range(num_abins):
                    #Bin radially and calculate Sigma
                    dust_count = np.histogram(r_dust[a_ids==j],Rbins)[0]
                    Sigs     = dust_count*M_dust/Rbin_areas
                    hist_array[i,5+j,0]  = abin_mids[j]
                    hist_array[i,5+j,1:] = Sigs

                if vr_mode == True:
                    v_r, v_azi = b.v_r_azi(sph_pos,sph_vel)
                    gas_vr     = b.calc_binned_data(v_r,r_gas,Rbins)[0]
                    hist_array[i,4,1:] = gas_vr

                hist_array[i,0,0]  = headertime*code_time/c.sec_per_year #Years
                hist_array[i,0,1]  = M_sph
                hist_array[i,0,2]  = M_dust
                hist_array[i,0,3]  = M_star
                hist_array[i,1,1:] = Rbin_mids
                hist_array[i,2,1:] = gas_Sig
                hist_array[i,3,1:] = gas_u
                
            np.save(savedir+runfolders[runid].rstrip('//')+'_dust_Sigmas',hist_array)
            anim_dict[str(runid)] = hist_array
    
    #=== Plotting ===#
    fig20 = plt.figure(20,facecolor='white',figsize=(6,10))
    ax1  = fig20.add_axes([0.12,0.48,0.83,0.5])
    ax2  = fig20.add_axes([0.12,0.12,0.83,0.3],sharex=ax1)
    #ax1.set_ylim(0,1500)
    ax1.set_ylabel(r'$\Sigma [gcm^{-2}]$')
    ax2.set_ylabel('T [K]')
    if vr_mode == True:
        ax2.set_ylabel('v_r [cm/s]')
    ax1.semilogy()
    ax2.semilogy()
    plt.yscale('symlog')
    ax1.semilogx()
    plot_dict = {}
    dust_scale = 1e2
    num_snaps = len(anim_dict['0'][:,0,0])
    M_dust = anim_dict['0'][0,0,2]
    timetext = ax1.text(0.06,0.91,'Time: {:.2f}'.format(anim_dict['0'][0,0,0])
                        + ' Years',transform=ax1.transAxes)

    for runid in range(len(runfolders)):
        R_bin_mids = anim_dict[str(runid)][0,1,1:]
        gas_Sig = anim_dict[str(runid)][0,2,1:]
        gas_u   = anim_dict[str(runid)][0,3,1:]*code_L**2/code_time**2
        gas_T   = gas_u*(c.gamma_mono-1)*c.mu*c.mp/c.kb
        plot_gas,  = ax1.plot(Rbin_mids,gas_Sig*Rbin_mids,
                              color=run_cols[runid],label=str(runfolders[runid]))
        plot_gas2, = ax2.plot(Rbin_mids,gas_T,color=run_cols[runid])
        if vr_mode == True:
            gas_vr = anim_dict[str(runid)][0,4,1:]*code_L/code_time
            plot_gas2, = ax2.plot(Rbin_mids,gas_vr,color=run_cols[runid])     
        plot_dict[str(runid)+'gas'] = plot_gas
        plot_dict[str(runid)+'gas2'] = plot_gas2

        #=== Set up dust lines ===#
        if M_dust != 0:
            for j in range(num_abins):
                dust_Sig = anim_dict[str(runid)][0,5+j,1:]/anim_dict[str(runid)][0,5+j,1:]
                line,  = ax1.plot(Rbin_mids,dust_Sig,#*Rbin_mids*dust_scale,
                              label='a = {:.3f}'.format(abin_mids[j])+'cm',color=run_cols[j])
                plot_dict[str(runid)+'dust'+str(j)] = line 


    def animate(i):
        output = []
        for runid in range(len(runfolders)):
            gas_Sig = anim_dict[str(runid)][i,2,1:]
            gas_u = anim_dict[str(runid)][i,3,1:] *code_L**2/code_time**2
            gas_T = gas_u*(c.gamma_mono-1)*c.mu*c.mp/c.kb
            plot_dict[str(runid)+'gas'].set_ydata(gas_Sig*Rbin_mids)
            plot_dict[str(runid)+'gas2'].set_ydata(gas_T)
            if vr_mode == True:
                gas_vr = anim_dict[str(runid)][i,4,1:]*code_L/code_time
                plot_dict[str(runid)+'gas2'].set_ydata(gas_vr)
            if M_dust != 0:
                for j in range(num_abins):
                    dust_Sig = anim_dict[str(runid)][i,5+j,1:]/anim_dict[str(runid)][0,5+j,1:]
                    plot_dict[str(runid)+'dust'+str(j)].set_ydata(dust_Sig)#*Rbin_mids*dust_scale)
                    
        timetext.set_text('Time: {:.2f}'.format(anim_dict['0'][i,0,0]) + ' Years')
        output.append(timetext)
        output.append(plot_dict)
        return output
    
    ax1.legend(frameon=False)
    ani = animation.FuncAnimation(fig20, animate, interval=80, frames=num_snaps, blit=False, repeat=True)
    plt.show()
    
    return



def dust_distribution():
    #N ~ a^-3.5
    rho_dust = 3 #g/cm^-3
    K   = 1
    a   = np.logspace(-4,0,100)
    N   = K*a**-3.5

    M = N * 4*np.pi/3*a**3*rho_dust
    Mcum = np.cumsum(M)
    plt.figure(1,facecolor='w')
    plt.plot(a,M)
    plt.plot(a,Mcum/np.max(Mcum),c=run_cols[0])
    plt.semilogx()
    #plt.semilogy()


def dust_size():
    a_s    = [1600,70,540,11100,11200,3,93,1621,31110,2000]
    mdisc  = [0.009,0.015,0.15,0.03,0.012,0.006,0,0.018,0.114,0.03]
    plt.scatter(mdisc,a_s)
    plt.semilogx()
    plt.semilogy()

    return



if __name__ == "__main__":

    b.animate_1d(filepath,runfolders,Rin=0.1,Rout=2.5,rerun=True,norm_y=True)
    #dust_distribution()





    #Old requests..?
    
    #vertical_profile()
    #alpha_SS(snapid=70,runid=6)
    #alpha_SS(snapid=200,runid=3)
    #alpha_SS(snapid=0,runid=2)
    #alpha_SS(snapid=200,runid=2)

    #weidenschilling_r_mig(snapid=snapid,runid=0,it=0,bonus_plots=True,Rin=Rin,Rout=Rout)
    #weidenschilling_r_mig(snapid=snapid,runid=1,it=8,bonus_plots=True,Rin=Rin,Rout=Rout)

    #cross_section(runfolders=runfolders[:],anim_mode=True,rerun=False,vr_mode=False)
    
    #vertical_profile(anim_mode=True,anim_ind=0)
    #settling_test(runfolders,10,mean_bins=True,rerun=False,Rin=0.5,Rout=0.51)
    #settling_test(runfolders,0,anim_mode=True,mean_bins=True,Nframes=44,rerun=False,Rin=0.5,Rout=0.51)
    
    #Sergei Runs
    #settling_test(runfolders,10,anim_mode=False,mean_bins=True,Rin=0.5,Rout=0.501)
    #settling_test(runfolders,0,anim_mode=True,mean_bins=True,Nframes=70,rerun=False,Rin=0.5,Rout=0.501)
    #settling_test(runfolders,10,mean_bins=True)
    #dust_distribution()

    #Gio disc NSG disc vset
    #settling_test(runfolders,0,anim_mode=True,mean_bins=True,Nframes=14,rerun=False,Rin=0.5,Rout=0.51)
    #settling_test(runfolders,0,anim_mode=True,Nframes=14,rerun=False,Rin=0.5,Rout=0.51)

    #Gio disc NSG disc
    #settling_test(runfolders,0,anim_mode=True,mean_bins=True,Nframes=14,rerun=False,Rin=0.5,Rout=0.51,write=True)
    #settling_test(runfolders,0,anim_mode=True,Nframes=25,rerun=False,Rin=0.5,Rout=0.51,write=True)
    #settling_test(runfolders,0,paper_plots=True,Nframes=14,rerun=False,Rin=0.5,Rout=0.51,ppi=0)
    #settling_test(runfolders,0,paper_plots=True,Nframes=14,rerun=False,Rin=0.5,Rout=0.51,ppi=1)
    #settling_test(runfolders,0,paper_plots=True,Nframes=14,rerun=False,Rin=0.5,Rout=0.51,ppi=2)
    #settling_test(runfolders,0,paper_plots=True,Nframes=14,rerun=False,Rin=0.5,Rout=0.51,ppi=3)


    #ring_test(runfolders[0],it=0)
    #ring_test(runfolders[1],it=1)
    #ring_test(runfolders[2],it=2)
    #ring_test(runfolders[3],it=3)


    plt.show()
    
