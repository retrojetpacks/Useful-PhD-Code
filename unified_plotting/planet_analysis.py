'''Functions to analyse Gadget Planet Migration Runs'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
from matplotlib import animation
import os
import h5py
import glob
from pygadgetreader import *
import basic_astro_functions as b
import astrophysical_constants_cgs as c
import Quick_Render as Q



#Code units
code_M     = c.Msol                  #Msol in g
code_L     = 100*c.AU                #100 AU in cm
code_V     = 297837.66               #v_K @ 100 AU #in cm/s
code_time  = code_L/code_V           #seconds #~160 Yrs
code_rho   = code_M/code_L**3
code_E     = code_M*code_V**2
code_G     = c.G /code_L**3*code_M*code_time**2
AU_scale   = 100


#Font styles, sizes and weights
def set_rcparams(fsize=12):
    font = {'weight' : 'normal',
            'size'   : fsize,
            'serif' : 'Minion Pro'}
    plt.rc('font', **font)
    plt.rc('legend', fontsize=fsize)
    return
set_rcparams()

newblackholes = False
#filepath = '/rfs/TAG/rjh73/dust_accretion_runs/corrections/'
#runfolders = ['Disc_N100000_M001_relaxed_dust/']
#runfolders = ['O_M2_beta01_a1_n2e6/','O_M2_beta10_a1_n2e6/']

#runmode = 'Gio_disc'
#runmode = 'L_tests'
#runmode= 'R_sink'
runmode = 'poster_2'
#runmode = 'MP_poster'
#runmode = 'Gio_N1e6_planets'
runmode = 'Frag_sinks'

betaval='nan'

if runmode == 'N_res_beta10':
    runfolders = [#'O_M2_beta10_a1_n2e6/',
              'O_M2_beta10_a1_n1e6/',
              #'O_M2_beta10_a1/',
              #'O_M2_beta10_a1_n2e5/'
    ]
    labels = ['N1e6']#['N2e6','N1e6','N5e5','N2e5']
    betaval=10
    Pnums = [1e6]#[2e6,1e6,5e5,2e5]



elif runmode == 'N_res_beta01':
    filepath = '/rfs/TAG/rjh73/dust_accretion_runs/corrections/'
    runfolders = ['Disc_N125e5_R0130_M02_beta01_dust_MP2/',
              'Disc_N25e5_R0130_M02_beta01_dust_MP2/',
              'Disc_N5e5_R0130_M02_beta01_dust_MP2/',
              'Disc_N1e6_R0130_M02_beta01_dust_MP2/',
              'Disc_N2e6_R0130_M02_beta01_dust_MP2/'
    ]
    labels = ['N125e5','N25e5','N5e5','N1e6','N2e6']
    betaval = 0.1
    Pnums = [1.25e5,2.5e5,5e5,1e6,2e6]
    newblackholes = True


elif runmode == 'Z_tests':
    runfolders = ['O_M2_beta10_a1_n1e6_z3e4/','O_M2_beta10_a1_n1e6_z1e3/','O_M2_beta10_a1_n1e6_z3e3/']
    labels = [r'Z = 3$\times10^{-4}$',r'Z = 1$\times10^{-3}$',r'Z = 3$\times10^{-3}$']
    f_pebs = [0.03,0.1,0.3]
    betaval = 10


elif runmode == 'Gio_disc':
    filepath = '/rfs/TAG/rjh73/Gio_disc/'
    runfolders = ['Gio_N1e6_MP01/','Gio_N1e6_MP05/','Gio_N1e6_MP1/','Gio_N1e6_MP2/','Gio_N1e6_MP4/']
    labels = ['MP01','MP05','MP1','MP2','MP4']
    betaval=5


elif runmode == 'L_tests':
    filepath = '/rfs/TAG/rjh73/dust_accretion_runs/corrections/'
    runfolders = ['Disc_N1e6_R0130_M02_beta10_dust_MP2/',
                  'Disc_N1e6_R0130_M02_beta10_dust_MP2_Accdtdiv2/',
                  'Disc_N1e6_R0130_M02_beta10_dust_MP2_Coudiv2/',
                  'Disc_N1e6_R0130_M02_beta10_dust_MP2_rs05/',
                  'Disc_N1e6_R0130_M02_beta10_dust_MP2_rs2/',
                  'Disc_N1e6_R0130_M02_beta10_MP2/'
                  #'O_M2_beta10_a1_n1e6/',

    ]
    labels = ['N1e6','Accdt/2','Cou/2','rs05','rs2','nodust','O_M2',]
    betaval=10

elif runmode == 'R_sink':
    filepath = '/rfs/TAG/rjh73/dust_accretion_runs/'
    runfolders = ['O_M2_beta10_a1_r05/','O_M2_beta10_a1_r2/']
    labels = ['rs05','rs2']
    betaval=10

elif runmode == 'poster_2':
    filepath = '/rfs/TAG/rjh73/dust_accretion_runs/'
    runfolders = ['O_M2_beta01_a1_n1e6/',
                  'O_M2_beta1_a1_n1e6_IRR0/',
                  'O_M2_beta1_a1_n1e6_IRR3e4/',
                  'O_M2_beta10_a1_n1e6/']
    labels = [r'$\beta$=0.1',r'$\beta$=1',r'$\beta$=1; Irr=3$\times10^{-3}L_{\odot}$',r'$\beta$=10']
    

elif runmode == 'MP_poster':
    filepath = '/rfs/TAG/rjh73/dust_accretion_runs/'
    runfolders = ['O_M05_beta10_a1/','O_M1_beta10_a1/','O_M2_beta10_a1_n1e6/','O_M4_beta10_a1/','O_M8_beta10_a1/']
    labels = [r'0.5$M_J$',r'$M_J$',r'2$M_J$',r'4$M_J$',r'8M$_J$']
    betaval = 10
    f_pebs = [0.1,0.1,0.1,0.1,0.1]

    
elif runmode == 'Gio_N1e6_planets':
    filepath = '/rfs/TAG/rjh73/Gio_disc/'
    runfolders = [#'Gio_N1e6_aav01_R00120_MD001_Z10/',
                  'Gio_N1e6_aav01_R00120_MD001_Z10_MP03/',
                  'Gio_N1e6_aav01_R00120_MD001_Z10_MP1/',
                  'Gio_N1e6_aav01_R00120_MD001_Z10_MP3/',
                  #'Gio_N1e6_aav01_R00120_MD01_Z10/',
                  'Gio_N1e6_aav01_R00120_MD01_Z10_MP03/',
                  'Gio_N1e6_aav01_R00120_MD01_Z10_MP1/',
                  'Gio_N1e6_aav01_R00120_MD01_Z10_MP3/',
    ]
    
elif runmode == 'Frag_sinks':
    filepath = '/rfs/TAG/rjh73/Frag_in_disc/'
    runfolders = ['Polydisc_N16e5_M5_R3_r50_b5_MD001_Z10_a1_sink1e10/',
    'Polydisc_N16e5_M5_R3_r50_b5_MD001_Z10_a1_sink1e11/',
    ]
try:
    labels = labels
except:
    labels = runfolders

    
snapprefix = 'snapshot_'
savedir = '/rfs/TAG/rjh73/save_data/'
blackholes = 'blackholes.txt'
Z_disc = 0.01

#run_cols = ['#2EC4C6', '#CD6622', '#440044', '#FFC82E', '#ff4400', '#8800bb']
run_cols = ['#CD6622', '#FFC82E',   '#440044',  '#2EC4C6','#ff4400', '#8800bb']
run_colsZ = plt.cm.magma(np.linspace(0,1,len(runfolders)+2)[1:-1])


def calc_binned_data(data,rs,bins):
    '''Bin data to produce annuli counts and determine Sigma [g/cm^2]s'''
    
    #Find bin number index for each value
    r_inds = np.digitize(rs,bins)-1
    binned_stat = np.array([np.mean(data[r_inds==i]) for i in range(len(bins)-1)])
    
    return binned_stat


def calc_H68(zs,rs,bins):
    '''Calculate height containing 68% of disc particles above midplane'''
    
    #Find bin number index for each value
    r_inds = np.digitize(rs,bins)-1
    data = np.sqrt(zs**2)
    H68 = np.zeros(len(bins)-1)
    for i in range(len(H68)):
        try:
            Hs = np.sort(zs[r_inds==i])
            H68[i] = Hs[int(0.68*len(Hs))]
        except:
            H68[i] = 0
    return H68


def read_blackholes(runfolder):
    '''Read blackholes.txt'''
    a = np.genfromtxt(filepath+runfolder+blackholes)

    time        = a[:,0] * code_time / c.sec_per_year #Years
    sep         = a[:,1]
    macc_tot    = a[:,2]
    macc_dust   = a[:,3]
    Planet_mass = a[:,4]
    '''
    plt.figure(14)
    for i in range(7):
        plt.plot(a[:,0],a[:,i],label=i)
    plt.semilogy()
    plt.legend()
    '''
    return time,sep,macc_tot,macc_dust,Planet_mass


def resample(t_short,t_long,y_long):
    '''sub sample y_long based on t_short. Ideal for non uniform sampling frequency'''
    y_short = np.zeros(len(t_short))
    for i in range(len(t_short)):
        ind = np.argmin((t_long-t_short[i])**2)
        y_short[i] = y_long[ind]
    return y_short


def iterative_RH(planet_pos,sph_pos,dust_pos,M_star,MP,M_sph,M_dust):
    '''Iteratively calculate the Hill Radius'''
    sph_pos = sph_pos-planet_pos
    dust_pos = dust_pos-planet_pos
    planet_r = np.sqrt(planet_pos[0]**2+planet_pos[1]**2)
    sph_rel_P  = np.sort(np.sqrt(sph_pos[:,0]**2+sph_pos[:,1]**2+sph_pos[:,2]**2))
    dust_rel_P = np.sort(np.sqrt(dust_pos[:,0]**2+dust_pos[:,1]**2+dust_pos[:,2]**2))
    print 'MP', MP
    mass_temp = MP.copy()
    i = 0
    ri = sph_rel_P[i]
    RH_temp = b.hill_radius(planet_r,mass_temp,M_star)
    bRH = RH_temp.copy()
    
    while ri < RH_temp:
        RH_temp = b.hill_radius(planet_r,mass_temp,M_star)
        mass_temp += M_sph
        i += 1
        ri = sph_rel_P[i]
    RH = RH_temp
    MH = mass_temp

    ind  = np.argmin((sph_rel_P-RH/2)**2)
    indd = np.argmin((dust_rel_P-RH/2)**2)
    MH_2 = ind*M_sph
    MH_2_dust = indd*M_dust

    return RH,MH,MH_2,MH_2_dust


def gen_save_file(runfolder,bin_data=False):
    '''Loop over all gadget snapshots and blackholes.txt and store useful data'''

    num_snaps = len(glob.glob(filepath+runfolder+'snap*'))
    print 'Num Snaps: ',  num_snaps

    #Binning parameters
    num_bins = 200
    Rin=0.1
    Rout=1.5
    r_bins = np.linspace(Rin,Rout,num_bins+1)
    r_bin_mids = (r_bins[1:]+r_bins[:-1])/2
    dr = r_bins[1]-r_bins[0]

    #Save array
    output = np.zeros((num_snaps,20))
    bin_output = np.zeros((num_snaps,6,num_bins))


    for i in range(num_snaps):
        print 'Snap: ', i
        snapid = str(i).zfill(3)

        #==== Load body data ====#      
        headertime   = readheader(filepath+runfolder+snapprefix+snapid,'time')
        M_bodies     = readsnap(filepath+runfolder+snapprefix+snapid,'mass','bndry')
        pos_bodies   = readsnap(filepath+runfolder+snapprefix+snapid,'pos','bndry')
        vel_bodies   = readsnap(filepath+runfolder+snapprefix+snapid,'vel','bndry')
        M_star       = M_bodies[np.argmax(M_bodies)] #code_M
        M_planets    = M_bodies[np.arange(len(M_bodies))!=np.argmax(M_bodies)]
        pos_star     = pos_bodies[np.argmax(M_bodies)]
        vel_star     = vel_bodies[np.argmax(M_bodies)]
        pos_planets  = pos_bodies[np.arange(len(pos_bodies))!=np.argmax(M_bodies)] - pos_star
        vel_planets  = vel_bodies[np.arange(len(pos_bodies))!=np.argmax(M_bodies)] - vel_star
        r_planets    = np.sqrt(pos_planets[:,0]**2+pos_planets[:,1]**2)
        vphi_plan    = np.sqrt(vel_planets[:,0]**2+vel_planets[:,1]**2)
        planet_L     = M_planets * r_planets * vphi_plan
        
        #==== Load SPH data ==#
        N_sph        = readheader(filepath+runfolder+snapprefix+snapid,'gascount')
        M_sph        = readsnap(filepath+runfolder+snapprefix+snapid,'mass','gas')[0]
        sph_pos      = readsnap(filepath+runfolder+snapprefix+snapid,'pos','gas') - pos_star
        sph_vel      = readsnap(filepath+runfolder+snapprefix+snapid,'vel','gas') - vel_star
        sph_r        = np.sqrt(sph_pos[:,0]**2+sph_pos[:,1]**2)
        sph_vphi     = np.sqrt(sph_vel[:,0]**2+sph_vel[:,1]**2)
        sph_L        = np.sum(sph_r*sph_vphi)*M_sph

        #==== Load dust data ====#
        try:
            N_dust   = readheader(filepath+runfolder+snapprefix+snapid,'diskcount')
            M_dust   = readsnap(filepath+runfolder+snapprefix+snapid,'mass','disk')[0]
            dust_pos = readsnap(filepath+runfolder+snapprefix+snapid,'pos','disk') - pos_star
            dust_vel = readsnap(filepath+runfolder+snapprefix+snapid,'vel','disk') - vel_star
            dust_r        = np.sqrt(dust_pos[:,0]**2+dust_pos[:,1]**2)
            dust_vphi     = np.sqrt(dust_vel[:,0]**2+dust_vel[:,1]**2)
            dust_L        = np.sum(dust_r*dust_vphi)*M_dust
        except:
            N_dust = 0
            M_dust = 0
            dust_pos = np.array([[0,0,0]])
            dust_vel = np.array([[0,0,0]])
            dust_L = 0

        #==== Fill in save array ====#
        output[i,0] = headertime *code_time/c.sec_per_year #Years
        output[i,1] = M_star
        
        if bin_data == True:
            #=== Load data for binning ===#
            sph_u    = readsnap(filepath+runfolder+snapprefix+snapid,'u','gas')

            #=== Calculate bin outputs ===#
            sig_gas  = np.histogram(sph_r,r_bins)[0]*M_sph/(2*np.pi*r_bin_mids*dr)
            H68_gas  = calc_H68(sph_pos[:,2],sph_r,r_bins)
            u_gas    = calc_binned_data(sph_u,sph_r,r_bins)

            dust_r   = np.sqrt(dust_pos[:,0]**2+dust_pos[:,1]**2)
            sig_dust = np.histogram(dust_r,r_bins)[0]*M_dust/(2*np.pi*r_bin_mids*dr)
            H68_dust = calc_H68(dust_pos[:,2],dust_r,r_bins)

            #=== Save bin output ===#
            
            bin_output[i,0,:] = r_bin_mids *100 #AU
            bin_output[i,1,:] = sig_gas *code_M/code_L**2
            bin_output[i,2,:] = H68_gas *100 #AU
            bin_output[i,3,:] = u_gas  *code_L**2/code_time**2
            bin_output[i,4,:] = sig_dust *code_M/code_L**2
            bin_output[i,5,:] = H68_dust *100 #AU
            
        #=== Load Planet data ===#
        RH,MH,MH_2,MH_2_dust = iterative_RH(pos_planets[0],sph_pos,dust_pos,M_star,M_planets,M_sph,M_dust)
        output[i,2] = M_planets
        output[i,3] = MH
        output[i,4] = MH_2
        output[i,5] = RH *100 #AU
        output[i,6] = np.sqrt(pos_planets[0][0]**2+pos_planets[0][1]**2) *100 #AU
        output[i,8] = MH_2_dust
        output[i,9] = M_sph
        output[i,10] = M_dust
        output[i,11] = sph_L
        output[i,12] = planet_L
        output[i,12] = dust_L
            

    #Mdot blackholes.txt
    time = output[:,0] #years
    bhtime,bhsep,bhmacc_tot,bhmacc_dust,bhMP = read_blackholes(runfolder)
    Macc_dust = resample(time,bhtime[bhMP<1],bhmacc_dust[bhMP<1])
    #print time
    #print bhtime
    #print bhmacc_dust
    #Macc_dust = resample(time,bhtime,bhmacc_dust)
    output[:,7] = Macc_dust

    #Save Arrays
    np.save(savedir+runfolder.rstrip('//')+'_header',output)
    if bin_data == True:
        np.save(savedir+runfolder.rstrip('//')+'_binned',bin_output)
        return output,bin_output
    else:
        return output





def planet_diagnostic(runfolders,Z_frac=False,rerun=False):
    '''Plot of sep, Mgas, Mdust, Z?'''

    fig1 = plt.figure(1,facecolor='white',figsize=(3.5,8))
    ax1 = fig1.add_axes([0.2,0.66,0.75,0.29])
    ax2 = fig1.add_axes([0.2,0.37,0.75,0.29],sharex=ax1)
    ax3 = fig1.add_axes([0.2,0.08,0.75,0.29],sharex=ax1)
    top, mid, bot = [ax1], [ax2], [ax3]

    for i in range(len(runfolders)):
        print runfolders[i]
        try:
            if rerun == True:
                1/0
            load = np.load(savedir+runfolders[i].rstrip('//')+'_header.npy')
        except:
            load = gen_save_file(runfolders[i])

        #=== Read Gadget output from saved np array ===#
        time      = load[:,0] #Years
        M_star    = load[:,1] #Msol
        MP        = load[:,2]*c.Msol/c.MJ #MJ
        MH        = load[:,3]*c.Msol/c.MJ #MJ
        MH_2      = load[:,4]*c.Msol/c.MJ #MJ
        RH        = load[:,5] #AU
        rP        = load[:,6] #AU
        Macc_dust = load[:,7]*c.Msol/c.MJ #MJ
        MH_2_dust = load[:,8]*c.Msol/c.MJ #MJ
        L_gas     = load[:,11]

        ax1.plot(time,rP,color=run_cols[i],label=labels[i])
        ax2.plot(time,MP,color=run_cols[i],label='M$_{SINK}$')
        ax2.plot(time,MH_2+MP,ls='--',color=run_cols[i],label='M$_{RH/2}$')

        if Z_frac == True:
            M_frac = 100* Macc_dust/c.Msol*c.MJ / (Z_disc*f_pebs[i])
            ax3.plot(time,M_frac,color=run_cols[i])
            ax3.set_ylabel(r'% of Total Dust Accreted')
        else:
            ax3.plot(time,Macc_dust*100,color=run_cols[i])
            ax3.plot(time,(MH_2_dust+Macc_dust)*100,ls='--',color=run_cols[i])
            ax3.set_ylabel(r'Dust Mass [M$_J$/100]')

        if i == 0:
            ax2.legend(frameon=False,loc=2)
    ax1.set_ylabel(r'Orbital Sep [AU]')
    ax2.set_ylabel(r'Planet Mass [M$_J$]')
    ax3.set_xlabel('Time [Years]')
    if betaval != 'nan':
        ax1.text(0.75,0.9,r'$\beta$ = '+str(betaval), transform=ax1.transAxes)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax1.legend(frameon=False,loc=3)   
    return




def planet_Z(runfolders):
    '''Make metallicity predictions'''
    
    fig3 = plt.figure(3,facecolor='white',figsize=(3.5,4))
    ax1 = fig3.add_axes([0.25,0.2,0.7,0.75])
    ax1.set_ylabel('Z Estimate')
    #ax2 = fig3.add_axes([0.25,0.1,0.7,0.4],sharex=ax1)
    #ax2.set_ylabel('Metal Mass [M$_J$/100]')
    ax1.set_xlabel('Time [Years]')
    
    for i in range(len(runfolders)):
        print runfolders[i]
        try:
            #1/0
            load = np.load(savedir+runfolders[i].rstrip('//')+'_header.npy')
        except:
            load = gen_save_file(runfolders[i])

        #=== Read Gadget output from saved np array ===#
        time      = load[:,0] #Years
        M_star    = load[:,1] #Msol
        MP        = load[:,2]*c.Msol/c.MJ #MJ
        MH        = load[:,3]*c.Msol/c.MJ #MJ
        MH_2      = load[:,4]*c.Msol/c.MJ #MJ
        RH        = load[:,5] #AU
        rP        = load[:,6] #AU
        Macc_dust = load[:,7]*c.Msol/c.MJ #MJ
        MH_2_dust = load[:,8]*c.Msol/c.MJ #MJ

        Z0 = 0.02

        Z = (MP[0]*Z0 + 2*Macc_dust + (MP-MP[0])*Z0*(1-f_pebs[i]) ) / MP
        M_peb = Macc_dust
        M_gasZ = (MP-MP[0])*Z0*(1-f_pebs[i])
        
        ax1.plot(time,Z,color=run_colsZ[i],label=labels[i])
        ax1.legend(frameon=False,loc=2)
        #ax1.plot(time,M_gasZ,color=run_cols[i],ls='--')
        #ax2.plot(time,M_peb*100,color=run_cols[i],label='Pebbles')
        #ax2.plot(time,M_gasZ*100,color=run_cols[i],ls='--',label='Coupled Metals')
        #if i ==0:
        #    ax2.legend(frameon=False,loc=2)
        
    

def disc_animation(runfolder):
    '''Animations of quantities accross the disc'''
    
    try:
        #1/0
        load = np.load(savedir+runfolder.rstrip('//')+'_header.npy')
        load_bins = np.load(savedir+runfolder.rstrip('//')+'_binned.npy')
        print 'Data loaded successfully!'
    except:
        print 'Must run data gen routine'
        load, load_bins = gen_save_file(runfolder,bin_data=True)
    N_frames = np.shape(load)[0]

    fig2 = plt.figure(2,facecolor='white',figsize=(7,8))
    ax1 = fig2.add_axes([0.12,0.12,0.8,0.8])
    ax1.semilogy()
    #ax1.semilogx()
    ax1.set_ylabel(r'$\Sigma$xR [AU gcm$^{-2}$]')
    ax1.set_xlim(10,150)
    ax1.set_xlabel('Disk Radius [AU]')
    
    #Load data
    time      = load[:,0] #Years
    rP        = load[:,6] #AU
    R_bins    = load_bins[0,0,:]
    Sig_gas   = load_bins[:,1,:]
    Sig_dust  = load_bins[:,4,:]
    dust_scale = 1000
    
    plot_gas,   = ax1.plot(R_bins,Sig_gas[0]*R_bins,label=r'$\Sigma_{GAS}$',color=run_cols[0])
    plot_dust,  = ax1.plot(R_bins,Sig_dust[0]*dust_scale*R_bins,label=r'$\Sigma_{DUST} \times$'+str(dust_scale),color=run_cols[1])
    planet_mark = ax1.axvline(rP[0],c='k',ls='--')
    time_text   = ax1.text(0.72,0.95,'Time: {:.0f} '.format(int(time[0])) + ' Yrs',transform=ax1.transAxes)

    def animate(i):
        plot_gas.set_ydata(Sig_gas[i]*R_bins)
        plot_dust.set_ydata(Sig_dust[i]*dust_scale*R_bins)

        planet_mark.set_xdata(rP[i])
        time_text.set_text('Time: {:.0f} '.format(int(time[i])) + ' Yrs')
        return plot_gas, plot_dust, planet_mark,time_text,

    ani = animation.FuncAnimation(fig2, animate, interval=80, frames=N_frames, blit=True, repeat=True)
    ax1.legend(frameon=False,loc=2)
    plt.show()
    
    return





def Epstein_regime_test(runfolder,bin_data=True):
    '''Check that grains in disc midplane are Epstein valid'''    

    #==== Load Data ====#
    try:
        load = np.load(savedir+runfolder.rstrip('//')+'_header.npy')
        load_bins = np.load(savedir+runfolder.rstrip('//')+'_binned.npy')
        print 'Data loaded successfully!'
    except:
        print 'Must run data gen routine'
        load, load_bins = gen_save_file(runfolder,bin_data=True)

    snaps = [0,167]
    rP        = load[:,6] #AU
    R_bins    = load_bins[0,0,:]
    plt.figure(7,facecolor='w',figsize=(6,4))
    
    for i in range(len(snaps)):
        time     = load[snaps[i],0] #Years
        Sig_gas  = load_bins[snaps[i],1,:]
        H68_gas  = load_bins[snaps[i],2,:]
        rho0_gas = Sig_gas/(np.sqrt(2*np.pi)*H68_gas) *code_M/code_L**3
        rho0_13  = rho0_gas*1e13
        a_Ep     = 10**5/rho0_13

        plt.plot(R_bins,a_Ep,label='t = {:.0f}'.format(time)+' Yrs',color=run_cols[i])

    plt.xlabel('Disc Radius [AU]')
    plt.ylabel(r'Upper Epstein grain size [cm]')
    plt.legend(frameon=False,loc=4)
    plt.semilogy()
    return


def L_plot(it):
    '''Total ang mom plots'''
    runfolder = runfolders[it]
    try:
        #1/0
        load = np.load(savedir+runfolder.rstrip('//')+'_header.npy')
    except:
        load = gen_save_file(runfolder)

    time      = load[:,0] #Years
    M_star    = load[:,1] #Msol
    MP        = load[:,2]*c.Msol/c.MJ #MJ
    MH        = load[:,3]*c.Msol/c.MJ #MJ
    MH_2      = load[:,4]*c.Msol/c.MJ #MJ
    RH        = load[:,5] #AU
    rP        = load[:,6] #AU
    Macc_dust = load[:,7]*c.Msol/c.MJ #MJ
    MH_2_dust = load[:,8]*c.Msol/c.MJ #MJ
    L_gas     = load[:,11]
    L_planet  = load[:,12]
    L_dust    = load[:,13]
    Star_sink = 0.03
    L_acc     = (M_star-M_star[0])*Star_sink*np.sqrt(code_G*M_star/Star_sink)

    #R_sink = [0.005,0.02]
    #L_acc_corr = (MP-MP[0])*R_sink[it]*np.sqrt(code_G*MP/R_sink[it])
    
    L_tot = L_gas+L_planet+L_dust
    #L_tot_corr = L_tot + L_acc_corr
    #L_tot_c   = L_tot + L_acc

    fig9 = plt.figure(9,facecolor='w',figsize=(7,3))
    ax1 = fig9.add_axes([0.15,0.18,0.8,0.75])

    ax1.plot(time,L_tot/L_tot[0],color=run_cols[it],label=labels[it])
    #ax1.plot(time,L_tot_corr/L_tot_corr[0],color=run_cols[it],ls='--')
    #plt.ylim(0,1)
    plt.legend(frameon=False)
    plt.xlabel('Time [Years]')
    plt.ylabel(r'$\vec{L}/\vec{L_0}$')
    return




def convergence_check(runfolders):
    ''''''
    MRH_2s = []
    
    for i in range(len(runfolders)):
        print runfolders[i]
        try:
            #1/0
            load = np.load(savedir+runfolders[i].rstrip('//')+'_header.npy')
        except:
            load = gen_save_file(runfolders[i])

        #=== Read Gadget output from saved np array ===#
        time      = load[:,0] #Years
        M_star    = load[:,1] #Msol
        MP        = load[:,2]*c.Msol/c.MJ #MJ
        MH        = load[:,3]*c.Msol/c.MJ #MJ
        MH_2      = load[:,4]*c.Msol/c.MJ #MJ
        RH        = load[:,5] #AU
        rP        = load[:,6] #AU
        Macc_dust = load[:,7]*c.Msol/c.MJ #MJ
        MH_2_dust = load[:,8]*c.Msol/c.MJ #MJ

        #MRH_2s.append(MH_2_dust+Macc_dust)
        #MRH_2s.append(MH+MH_2)
        MRH_2s.append(Macc_dust)
        ##print time[150]

    plt.figure(10)
    #plt.scatter(Pnums,MRH_2s)
    ind = 150
    q = np.log((MRH_2s[0][:ind]-MRH_2s[1][:ind])/(MRH_2s[1][:ind]-MRH_2s[2][:ind])) /  np.log((MRH_2s[1][:ind]-MRH_2s[2][:ind])/(MRH_2s[2][:ind]-MRH_2s[3][:ind]))

    plt.plot(time[:ind],q,label='q')
    for i in range(len(Pnums)-1):
        plt.plot(time[:ind],abs(MRH_2s[i][:ind]-MRH_2s[i+1][:ind]),label=labels[i]+'-'+labels[i+1])
    #plt.semilogy()
    plt.legend()
    #plt.semilogx()


    plt.figure(11)
    print MRH_2s[:][:]
    for i in range(len(runfolders)-1):
        plt.scatter((Pnums[i+1]+Pnums[i])/2,np.sum(abs(MRH_2s[i+1][:ind]-MRH_2s[i][:ind])),label=str(Pnums[i+1])+'-'+str(Pnums[i]),color=run_cols[i])
    plt.semilogy()
    plt.semilogx()
    plt.legend()
    return







if __name__ == "__main__":

    #b.animate_1d(filepath,runfolders,var2='dust_gas',Rin=0.001,Rout=0.1,zoom='ZP',rerun=False,norm_y=True)
    #b.animate_1d(filepath,runfolders,var2='dust_gas')

    planet_diagnostic(runfolders,Z_frac=False)
    #planet_diagnostic(runfolders,rerun=False)
    #planet_Z(runfolders)
    #Epstein_regime_test(runfolders[1],bin_data=True)
    #disc_animation(runfolders[1])

    #for i in range(len(runfolders)):
    #    L_plot(it=i)
    

    #convergence_check(runfolders)
    plt.show()
