#!/usr/bin/env python

'''Basic functions for astrophysics'''
from __future__ import division
import numpy as np
import astrophysical_constants_cgs as c
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from scipy.stats import norm
from scipy.stats import lognorm
from scipy import spatial
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
import glob
from pygadgetreader import *
import h5py
import os
import player
from astropy.io import fits
from astropy.wcs import WCS

#==== Code units ====#
code_M     = c.Msol                  #Msol in g
code_L     = 100*c.AU                #100 AU in cm
code_V     = 297837.66               #v_K @ 100 AU #in cm/s
code_time  = code_L/code_V           #seconds #~160 Yrs
code_rho   = code_M/code_L**3
code_E     = code_M*code_V**2
code_G     = c.G /code_L**3*code_M*code_time**2
AU_scale   = 100


#==== Font styles, sizes and weights ====#
def set_rcparams(fsize=12):
    font = {'weight' : 'normal',
            'size'   : fsize,
            'serif' : 'Minion Pro'}
    plt.rc('font', **font)
    plt.rc('legend', fontsize=fsize)
    plt.rc('lines',linewidth = 2)
    
    return
set_rcparams()

#==== File directory paths ====#
savedir = '/rfs/TAG/rjh73/save_data/'
run_cols = ['#2EC4C6','#CD6622','#440044','#FFC82E','#FF1493','#6a5acd']
plot_cols = ['#2EC4C6','#FFC82E','#CD6622','#FF1493','#440044','#6a5acd','#33cc33','#0055ff','#dd44cc','#cd6622','#A8CFFF','#62D0FF']

linestyles = [':','-.','--','-',':','--']
linewidths = [1,1,1,1,2,2]




def func_powerlaw(x, m, c, c0):
    return c0 + x**m * c


def mag(a):
    '''Find magnitude of a three component vector'''
    mag_a = np.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
    return mag_a
    
def calc_binned_data(data,bin_dim_coord,bins,H68_mode=False):
    '''Bin data according to some coordinate. 
    Calculate the mean and std of a different stat based on those binnings'''
    
    #Find bin number index for each value
    bin_inds = np.digitize(bin_dim_coord,bins)-1
    
    #Calculate 68% value for each bin
    if H68_mode == True:
        H68s = np.zeros(len(bins)-1)
        for i in range(len(bins)-1):
            H_sort  = np.sort(abs(data[bin_inds==i]))
            try:
                H68s[i] = H_sort[int(0.68*len(H_sort))]
            except:
                print 'Bin empty'
                H68s[i] = 0
        return H68s

    #Calculate means and stdevs for each bin
    else:
        means = np.zeros(len(bins)-1)
        stds  = np.zeros(len(bins)-1)
        for i in range(len(bins)-1):
            means[i],stds[i] = norm.fit(data[bin_inds==i])
        return means,stds



def radial_profile(xs,ys,data, center,two_sided):
    '''Calculate a radially averaged profile from a 2d grid'''
    X,Y = np.meshgrid(xs,ys)
    Rs = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    r_bins = np.linspace(0,np.max(Rs),len(xs))
    r_bin_mids = (r_bins[1:]+r_bins[:-1])/2
    
    R_profile, R_profile_stds = calc_binned_data(data,Rs,r_bins)

    if two_sided == True:
        mid = int(len(data)/2)
        R_profileA, R_profileA_stds = calc_binned_data(data[:,:mid],Rs[:,:mid],r_bins)
        R_profileB, R_profileB_stds = calc_binned_data(data[:,mid:],Rs[:,mid:],r_bins)

    if two_sided == False:
        return r_bin_mids, R_profile, R_profile_stds
    else:
        return r_bin_mids, R_profileA, R_profileA_stds, R_profileB, R_profileB_stds



def star_planet_angle(pos):
    '''Calculate the angle of an object at pos wrt the star, star must be at (0,0). Returns in degrees'''
    try:
        theta = np.arctan2(pos[:,1],pos[:,0])
    except:
        theta = np.arctan2(pos[1],pos[0])
    return theta*180/np.pi



def v_r_azi(pos,vel):
    '''relative to 0,0,0,0,0,0'''
    '''
    v_dot_r   = np.zeros(len(pos))
    v_cross_r = np.zeros(len(pos))
    for i in range(len(pos)):
        v_dot_r[i]   = np.dot(pos[i,:],vel[i,:])
        v_c          = np.cross(pos[i,:],vel[i,:])
        v_cross_r[i] = np.sqrt(np.sum(v_c**2))
    
    r = np.sqrt(pos[:,0]**2+pos[:,1]**2+pos[:,2]**2)
 
    v_azi = v_cross_r/r
    v_r = v_dot_r /r
    '''
    rad = np.sqrt(pos[:,0]**2+pos[:,1]**2+pos[:,2]**2)
    the = np.arccos(pos[:,2]/rad)
    phi = np.arctan2(pos[:,1],pos[:,0])
    st,sp = np.sin(the), np.sin(phi)
    ct,cp = np.cos(the), np.cos(phi)
    
    svel = np.zeros(np.shape(vel))
    for i in range(len(pos[:,0])):
        vmat = np.array([[-sp[i],cp[i],0],
                        [-st[i]*cp[i],-st[i]*sp[i],ct[i]],
                         [ct[i]*cp[i],ct[i]*sp[i],st[i]]])
        svel[i,:] = np.matmul(vmat,vel[i])# np.dot(vmat,vel)
    
    
    return svel[:,2], svel[:,0]




def hill_radius(R,Mp,M_star):
    '''Hill radius: R,Mp,M_star'''
    RH = R * (Mp / (3*M_star))**(1/3)
    return RH
    
def v_kepler(M0,R):
    '''Keplerian velocity: M0,R'''
    v_kep = np.sqrt(M0*c.G/R)
    return v_kep
    

def disk_profile(R,M_disk,R_out,dH_dR):
    '''Calculate a simple disk profile'''
    Sigma_0 = M_disk / (2*np.pi*R_out**2)
    
    '''
    Sigma = Sigma_0 * R_out / R
    H = R * dH_dR
    rho_gas = Sigma/H
    T = 10*np.sqrt(100*c.AU) * R**(-0.5)
    '''
    T = 20 * np.sqrt(100*c.AU/R)
    H = np.sqrt(c.kb*T/(c.mu*c.mp)) * R / v_kepler(c.Msol,R)
    #Sigma = Sigma_0 * R_out / R
    Sigma = 10 * (1 + 1/R)
    rho_gas = Sigma/H
    
    return Sigma, H, rho_gas, T

def T_profile(R,T0=20,R0=1,power=-0.5):
    T = T0 * (R/R0)**power
    return T
    
def mean_free_path(rho_gas):
    mfp = 40 / (10**10*rho_gas)
    return mfp  
          
          
def reynolds(s,v,mfp,cs):
    Re = s*v/mfp/cs
    return Re
    
def thermal_vel(T0):
    vth0 = np.sqrt(8*c.kb*T0/(np.pi*c.mu*c.mp))   
    return vth0
    
def sound_speed(T0):
    cs = np.sqrt(c.gamma_dia*c.kb*T0/c.mp/c.mu)
    return cs
    


    
#Migration Regimes
def type_one(R,Mp,M_star,Sigma,H,f_acc):
    '''Migrate inside disk. RH<H  -  (Bate 2003)'''
    om_kep = np.sqrt(c.G*M_star/R**3)
    v_r = -f_acc * Mp / M_star**2 * Sigma / H**2 * R**5 * om_kep
    return v_r
    
    
def dimensionalise_mass(M_star,mu_c,R0,H):
    '''For a given dimensionless mass and radius, outputs a dimensional mass. See Lambrechts & Johansen 2012'''
    om_K = v_kepler(M_star,R0)/R0
    M_c = om_K**2 * H**3 * mu_c / c.G
    return M_c 

def normal_dist(x, mu, sig):
    '''Gaussian dist'''
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def lognorm_dist(x, mu, sigma):
    ''''''
    #mu = np.log(mu)
    #sigma=np.log(sigma)
    print x,mu,sigma
    dist = lognorm.pdf(x,s=sigma*mu,scale=mu,loc=mu)
    print dist

    return dist

#Polytropes
def polytrope(M,R=0,Tc=0,n=1.5):
    '''Return polytrope. Numerically calculated so indeterminant number of output
    M, R, Tc, n. PolyK returned in cgs'''
    if R==0:
        mode = 'Tc'
    elif Tc==0:
        mode = 'R'
    
    gamma = (1+n)/n
    M     = M*code_M #g
    R     = R*code_L #cm
    
    #Boundary Conditions at epsilon = 0
    theta    = 1
    dth_dep  = 0
    epsilon  = 0.00001
    thetas   = [theta]

    dth_deps = [dth_dep]
    epsilons = [epsilon]
    dep      = 0.001
    
    #Solve Lane-Emden equation numerically
    while theta > 0:
        dth_dep = dth_dep - (2/epsilon*dth_dep + theta**n)*dep
        theta   = theta   + dth_dep*dep
        thetas.append(theta)
        dth_deps.append(dth_dep)
        epsilons.append(epsilon)
        epsilon = epsilon + dep

    print 'Epsilon -1', epsilons[-1]
    
    #Undo parameterisation
    if mode == 'R':
        alpha = R/epsilons[-1]
    elif mode == 'Tc':        
        alpha = -1*M * c.mu*c.mp*c.G/(c.kb*Tc *(n+1)) /epsilons[-1]**2 /dth_deps[-1]
        R = alpha*epsilons[-1]

    rho_c = -1*M / (4*np.pi*alpha**3) / (epsilons[-1]**2 * dth_deps[-1])
    #polyK_cgs = 4*np.pi*c.G*alpha**2 / (n+1) * rho_c**((n-1)/n)
    temp      = c.G * (4*np.pi)**(1/n)/(n+1) * epsilons[-1]**((n-3)/n) * (-1*epsilons[-1]**2*dth_deps[-1])**((1-n)/n)
    

    polyK_cgs = R**((3-n)/n)* M**((n-1)/n) *temp
    print 'PolyK: ', '{0:.10f}'.format(polyK_cgs), ' [cgs]'
    
    P_c       = polyK_cgs * rho_c**((n+1)/n)
    T = c.mp*c.mu*P_c /c.kb / rho_c
    
    print 'R, M, rho_c, T', R,M, rho_c, T
    
    thetas    = np.asarray(thetas)
    epsilons  = np.asarray(epsilons)
    

    #Convert to code units. PolyK returned in cgs
    Rs    = alpha * epsilons  /code_L
    rhos  = rho_c * thetas**n /code_M*code_L**3
    rho_c = rho_c /code_M*code_L**3
    #polyK_code = polyK_cgs *code_M**(-1*n) *code_L**(n/(2*n+3)) * code_time**(-0.5)
    polyK_code = polyK_cgs *code_M**(1/n) *code_L**(-2-3/n) * code_time**(2)

    #plt.figure(0)
    #plt.scatter(epsilons,dth_deps)
    #plt.figure(1)
    #plt.scatter(Rs,rhos,color='g')
    
    return Rs, rhos, rho_c, polyK_cgs, polyK_code




def make_plots(figi,N,sharey=False,figsize=(6,8)):
    '''General figure axes making code'''
    
    fig = plt.figure(figi,facecolor='w',figsize=figsize)
    axes = []
    if N == 1:
        dy = 0.75/N
        fig_top = 0.9
    else:
        dy = 0.85/N
        fig_top=0.95
        
    for i in range(N):
        if i == 0:
            axes.append(fig.add_axes([0.15,fig_top-(i+1)*dy,0.8,dy]))
        else:
            if sharey == False:
                axes.append(fig.add_axes([0.15,fig_top-(i+1)*dy,0.8,dy],sharex=axes[0]))
            else:
                axes.append(fig.add_axes([0.15,fig_top-(i+1)*dy,0.8,dy],sharex=axes[0],sharey=axes[0]))

        if i != N-1:
            plt.setp(axes[i].get_xticklabels(), visible=False)
            
    return fig, axes


def Gadget_kernel(r_hs,hs):
    '''Evaluate Gadget kernel Wk'''
    r_hsA = r_hs[r_hs<0.5]
    r_hsB = r_hs[r_hs>0.5]
    
    Wks = np.zeros(np.shape(r_hs))
    Wks[r_hs<0.5] = 1 - 6*r_hsA**2 + 6*r_hsA**3
    Wks[r_hs>0.5] = 2 * (1 - r_hsB)**3
    Wks *= 8/(np.pi*hs[:,None]**3)

    return Wks
    
    
def Gadget_smooth(xyz_SPH, A_SPH=1, M_SPH=1, xyz_output=[0,0,0], rho_output=False,compute_tree=True,tree=1,h_output=False,dup=0.1):
    '''Using sph info, calculate the kernel weighted quantity A at the coordinate xyz_output.
    Has an option to import a tree.'''

    Nngb = 40
    if compute_tree == True:
        tree = spatial.cKDTree(xyz_SPH)
        print 'Tree is built'
    
    print 'Querying Tree for density'
    #Compute kernel SPH value at particle coords
    dists, inds = tree.query(xyz_output,k=Nngb,distance_upper_bound=dup)
    hs = dists[:,-1]
    r_hs = dists/hs[:,None]
    print 'Calculating Gadget kernel'
    Wks = Gadget_kernel(r_hs,hs)
    
    if rho_output == True:
        rhos = M_SPH * np.sum(Wks,axis=1)
        print 'Mass render particle', M_SPH
        if h_output == True:
            return rhos, hs
        else:
            return rhos

    else:
        #==== Find SPH rho for normalising ====#
        print 'Querying SPH tree at grid points'
        SPH_dists,SPH_inds = tree.query(xyz_SPH,k=Nngb)
        print 'Finished SPH query'
        SPH_hs = SPH_dists[:,-1]
        SPH_r_hs = SPH_dists/ SPH_hs[:,None]
        Wks_SPH = Gadget_kernel(SPH_r_hs,SPH_hs)
        rho_SPH = M_SPH * np.sum(Wks_SPH,axis=1)
        A_SPH = np.hstack((A_SPH,[0]))
        rho_SPH = np.hstack((rho_SPH,[0]))
        A_ngbs =  A_SPH[inds]
        A_output = M_SPH * np.sum(A_ngbs * Wks /rho_SPH[inds],axis=1)
        A_output = np.nan_to_num(A_output)

        return A_output


def cart_spherical_posvel(pos,vel):
    '''Convert cartesian positions and velocities to spherical'''
    x,y,z    = pos[:,0],pos[:,1],pos[:,2]
    vx,vy,vz = vel[:,0],vel[:,1],vel[:,2]
    
    r      = np.sqrt(x**2+y**2+z**2)
    theta  = np.arctan2(np.sqrt(x**2+y**2),z)
    phi    = np.arctan2(y,x)
    vr     = (x*vx + y*vy + z*vz) / np.sqrt(x**2+y**2+z**2)
    vtheta = (vx*y - x*vy)/ (x**2+y**2)
    vphi   = (z*(x*vx+y*vy)-vz*(x**2+y**2))/ (x**2+y**2+z**2)/np.sqrt(x**2+y**2)

    spos    = np.dstack((phi,theta,r))[0]
    svel    = np.dstack((vphi,vtheta,vr))[0]
    
    return spos,svel

    
    
    

def Zhu_op(Top, rho_op):
    """opacity from Zhu et al 2009"""
    Pop = rho_op * c.kb * Top/c.mu #gas pressure
    kap_z = np.zeros(len(Top))
    
    for i in range(len(Top)):
        xlp = np.log10(Pop[i])
        xlt = np.log10(Top[i])
        xlop = - 30.
        if (xlt < 3.+0.03*(xlp+4.)): # metal grain
            xlop=-1.27692+0.73846*xlt
        elif (xlt < 3.08+0.028084*(xlp+4)): # metal grain evap
            xlop=129.88071-42.98075*xlt+(142.996475-129.88071)*0.1*(xlp+4)
        elif (xlt < 3.28+xlp/4.*0.12): #water
            xlop=-15.0125+4.0625*xlt
        elif (xlt < 3.41+0.03328*xlp/4.):  # water evap
            xlop=58.9294-18.4808*xlt+(61.6346-58.9294)*xlp/4.
        elif (xlt < 3.76+(xlp-4)/2.*0.03):  #molecular
            xlop=-12.002+2.90477*xlt+(xlp-4)/4.*(13.9953-12.002)
        elif (xlt < 4.07+(xlp-4)/2.*0.08):  #bound free,free free
            xlop=-39.4077+10.1935*xlt+(xlp-4)/2.*(40.1719-39.4077)
        elif (xlt < 5.3715+(xlp-6)/2.*0.5594):
            xlop=17.5935-3.3647*xlt+(xlp-6)/2.*(17.5935-15.7376)
        else:
            xlop = -0.48
        if ((xlop < 3.586*xlt-16.85) and (xlt < 4.)): xlop = 3.586*xlt - 16.85
        if (xlt < 2.9): xlop = -1.27692 + 0.73846*xlt

        kap_z[i] = 10.**xlop
    return kap_z



def find_RH_2(S,a,M0=0,beta=0):
   '''Find the half Hill radius and enclosed gas and dust mass for a clump
   Needs S relative to fragment centre'''
   

   gas_R     = np.sqrt(S.gas_pos[:,0]**2+S.gas_pos[:,1]**2+S.gas_pos[:,2]**2)
   dust_R    = np.sqrt(S.dust_pos[:,0]**2+S.dust_pos[:,1]**2+S.dust_pos[:,2]**2)
   planets_R = np.sqrt(S.planets_pos[:,0]**2+S.planets_pos[:,1]**2+S.planets_pos[:,2]**2)
   Rs        = np.hstack((np.hstack((gas_R,dust_R)),planets_R))
   Rind      = np.argsort(Rs)
   Rs        = Rs[Rind]
    
   #==== Sort mass and u arrays ====#
   Ms = np.hstack((np.hstack((np.ones(len(gas_R))*S.M_gas,np.ones(len(dust_R))*S.M_dust)),S.M_planets))
   Ms              = Ms[Rind]
   Ms_enc          = np.cumsum(Ms)
   us              = np.zeros(len(gas_R)+len(dust_R)+len(planets_R))
   us[:len(gas_R)] = S.gas_u
   us              = us[Rind]
   rhos            = np.zeros(len(gas_R)+len(dust_R)+len(planets_R))
   rhos[:len(gas_R)] = S.gas_rho
   rhos            = rhos[Rind]
   i = 1000

   #==== Calculate Gpot in sinks ====#
   rho_core = 5/code_M*code_L**3
   G_pot_sinks = np.sum(3/5*code_G * (4*np.pi*rho_core/3)**(1/3) * S.M_planets**(5/3))

   #==== Iteratively find Rh_2 ====#
   Rh_2 = a * (Ms_enc[i]/(3*S.M_star))**(1/3)
   Ri   = 0
   while Ri < Rh_2*1.1:
       Rh_2 = a * (Ms_enc[i]/(3*S.M_star))**(1/3)
       Ri = Rs[i]
       i += 10

   M_frag = Ms_enc[i]
   U     = np.sum(us[:i]*Ms[:i])
   G_pot = code_G * np.sum(Ms_enc[1:i]*Ms[1:i]/Rs[1:i])

   
   E_frag = U - G_pot #- G_pot_sinks + KE

   print 'U', U
   print 'G_pot', G_pot
   print 'G pot sinks', G_pot_sinks
   print 'E_frag', E_frag

   #Find protoplanet luminosity
   #Find gas inside PP
   T_eq = 20*(1/a)**0.5
   u_eq = c.kb *T_eq / c.mu / (c.gamma_dia-1)
   rho_crit = 2e-11 /code_M*code_L**3 #code_units
   t_cool = beta * (a/(S.M_star*code_G))**0.5 *(1+(rhos[:i]/rho_crit))**5
   du_dt = (us[:i]-u_eq)/t_cool
   L_PP = abs(np.sum(du_dt)*S.M_gas)
   
   Mean_dust_R = np.mean(dust_R[dust_R<Rh_2])
   print 'Mean Accreted Dust R', Mean_dust_R
   Macc_dust = S.M_dust* len(dust_R[dust_R<Rh_2])

   #==== Find envelope metallicity ====#
   R_M0 = np.sort(gas_R)[int(M0/S.M_gas)]
   MD_env = S.M_dust*len(dust_R[(dust_R>R_M0) & (dust_R<Rh_2)])
   
   print 'R_M0', R_M0*AU_scale, ' [AU]'
   print 'MD_env', MD_env 
   print 'Macc_dust', Macc_dust

   return Rh_2, M_frag, Macc_dust, Mean_dust_R, MD_env, R_M0,E_frag,L_PP







    
class Load_Snap:
    '''New snap reader. Uses classes and methods. All positions normalised to star'''

    def __init__(self,filepath,runfolder='',snapid='000',snapprefix='snapshot_',slen=3):

        snapid          = str(snapid).zfill(slen)        
        fname = filepath+runfolder+snapprefix+snapid+'.hdf5'
        if snapprefix == '':
            fname = filepath
        try:
            print 'fname', fname, filepath
            print os.path.isfile(fname) 
        except:
            print 'No file available!'
        
        load1 = h5py.File(fname)
        keys1 = load1.keys()
        print load1, keys1
        header = load1[keys1[0]]
        N_parts = header.attrs['NumPart_ThisFile']
        self.headertime = header.attrs['Time']*code_time/c.sec_per_year #Years
        print 'Time: ', self.headertime, ' Years'
        print 'Num part loaded', N_parts
        
        #==== Gas particles ====#
        self.N_gas = N_parts[0]
        self.M_gas = load1['PartType0']['Masses'][0]
        self.gas_pos  = load1['PartType0']['Coordinates'][:]
        self.gas_vel  = load1['PartType0']['Velocities'][:]
        self.gas_ID   = load1['PartType0']['ParticleIDs'][:]
        self.gas_u    = load1['PartType0']['InternalEnergy'][:]
        try:
            self.gas_rho  = load1['PartType0']['Density'][:]
        except:
            print 'No density data'
        try:
            self.gas_h = load1['PartType0']['SmoothingLength'][:]
        except:
            print 'No smoothing length data. Is this Phantom? :)'
            self.gas_h = np.zeros(self.N_gas) 

            
        #==== Dust particles ====#        
        self.N_dust = N_parts[2]
        if self.N_dust != 0:
            self.M_dust   = load1['PartType2']['Masses'][0]
            self.dust_pos = load1['PartType2']['Coordinates'][:]
            self.dust_vel = load1['PartType2']['Velocities'][:]
            try:
                self.dust_a = load1['PartType2']['DustRadius'][:]
                if len(self.dust_a) != np.shape(self.dust_pos[:,0])[0]:
                    1/0
            except:
                print 'Error loading dust size. Radius set to 1 cm'
                self.dust_a = np.ones(np.shape(self.dust_pos[:,0]))
        else:
            print 'No dust!'
            self.M_dust = 0
            self.dust_pos = np.array([]).reshape(0,3)
            self.dust_vel = np.array([]).reshape(0,3)
            self.dust_a   = np.array([]).reshape(0)
        print 'ff', np.shape(self.dust_pos)

        #==== Stars and Planets ====#
        N_sinks = N_parts[5]
        if N_sinks != 0:
            body_type        = 'bndry' #'disk'#
            M_bodies         = load1['PartType5']['Masses'][:]
            bodies_pos       = load1['PartType5']['Coordinates'][:]
            bodies_vel       = load1['PartType5']['Velocities'][:]
            self.M_star      = M_bodies[np.argmax(M_bodies)] #code_M
            self.M_planets   = M_bodies[np.arange(len(M_bodies))!=np.argmax(M_bodies)]
            self.N_planets   = len(self.M_planets)
            self.N_star      = 1
            self.star_pos    = bodies_pos[np.argmax(M_bodies)]
            self.star_vel    = bodies_vel[np.argmax(M_bodies)]
            self.planets_pos = bodies_pos[np.arange(len(bodies_pos))!=np.argmax(M_bodies)] - self.star_pos
            self.planets_vel = bodies_vel[np.arange(len(bodies_vel))!=np.argmax(M_bodies)] - self.star_vel
            print 'N_planets: ', self.N_planets, 'M [ME]: ',  np.flipud(np.sort(self.M_planets))*c.Msol/c.ME

            #Correct for star position
            self.gas_pos = self.gas_pos - self.star_pos
            self.gas_vel = self.gas_vel - self.star_vel
            try:
                self.dust_pos = self.dust_pos - self.star_pos
                self.dust_vel = self.dust_vel - self.star_vel

            except:
                print 'Still no dust!'
            print 'aaaa', self.planets_pos

        else:
            print 'No Star or planets present in simulation'
            self.N_planets   = 0
            self.N_star      = 0
            self.M_star      = np.array([]).reshape(0)
            self.M_planets   = np.array([]).reshape(0)
            self.star_pos = np.array([]).reshape(0,3)
            self.star_vel = np.array([]).reshape(0,3)
            self.planets_pos = np.array([]).reshape(0,3)
            self.planets_vel = np.array([]).reshape(0,3)

        #Finally shift star to reference location
        self.star_pos    -= self.star_pos
        self.star_vel    -= self.star_vel
        
        load1.close()

        
    def zoom(self,zoom_pos,zoom_vel):
        '''Correct positions and velocities for a given zoom coordinate'''
        
        #==== Correct positions ====#
        self.gas_pos     = self.gas_pos - zoom_pos
        self.dust_pos    = self.dust_pos - zoom_pos
        self.planets_pos = self.planets_pos - zoom_pos
        self.star_pos    = self.star_pos - zoom_pos


        #==== Correct velocities ====#
        self.gas_vel     = self.gas_vel - zoom_vel
        self.dust_vel    = self.dust_vel - zoom_vel
        self.planets_vel = self.planets_vel - zoom_vel
        self.star_vel    = self.star_vel - zoom_vel

        return
    

    def rotate(self,inc=0,azi=0,inc2=0):
        '''Rotate 3d coordinate matrix. array=[:,3],inc,azi,inc2'''
        inc,azi,inc2 = np.radians(inc),np.radians(azi),np.radians(inc2)
    
        c_inc, s_inc = np.cos(inc), np.sin(inc)
        c_azi, s_azi = np.cos(azi), np.sin(azi)
        c_inc2, s_inc2 = np.cos(inc2), np.sin(inc2)
        R_inc = np.matrix('{} {} {}; {} {} {}; {} {} {}'
                      .format(1,0,0,0,c_inc,-s_inc,0,s_inc,c_inc))
        R_azi = np.matrix('{} {} {}; {} {} {}; {} {} {}'
                      .format(c_azi,-s_azi,0,s_azi,c_azi,0,0,0,1))
        R_inc2 = np.matrix('{} {} {}; {} {} {}; {} {} {}'
                       .format(1,0,0,0,c_inc2,-s_inc2,0,s_inc2,c_inc2))

        self.gas_pos = np.asarray(np.dot(np.dot(np.dot(self.gas_pos,R_inc),R_azi),R_inc2))
        self.gas_vel = np.asarray(np.dot(np.dot(np.dot(self.gas_vel,R_inc),R_azi),R_inc2))

        try:
            self.dust_pos = np.asarray(np.dot(np.dot(np.dot(self.dust_pos,R_inc),R_azi),R_inc2))
            self.dust_vel = np.asarray(np.dot(np.dot(np.dot(self.dust_vel,R_inc),R_azi),R_inc2))
        except:
            pass
        try:
            self.planets_pos = np.asarray(np.dot(np.dot(np.dot(self.planets_pos,R_inc),R_azi),R_inc2))
            self.planets_vel = np.asarray(np.dot(np.dot(np.dot(self.planets_vel,R_inc),R_azi),R_inc2))
        except:
            pass
        try:
            self.star_pos = np.asarray(np.dot(np.dot(np.dot(self.star_pos,R_inc),R_azi),R_inc2))
            self.star_vel = np.asarray(np.dot(np.dot(np.dot(self.star_vel,R_inc),R_azi),R_inc2))
        except:
            pass
        return

    
    def subsample(self,box_lim):
        '''Remove all particles from beyond the box_lim'''
        sub = 1.1 * box_lim

        #==== Subsample Gas ====#
        gas_r2 = self.gas_pos[:,0]**2+self.gas_pos[:,1]**2+self.gas_pos[:,2]**2
        inds   = np.where(gas_r2<2*sub**2)[0] #2 due to circle/square lims
        
        self.gas_pos = self.gas_pos[inds,:]
        self.gas_vel = self.gas_vel[inds,:]
        self.gas_ID  = self.gas_ID[inds]
        self.gas_h   = self.gas_h[inds]
        self.gas_u   = self.gas_u[inds]
        self.gas_rho = self.gas_rho[inds]

        if self.N_dust != 0:
            #==== Subsample dust ====#
            dust_r2 = self.dust_pos[:,0]**2+self.dust_pos[:,1]**2+self.dust_pos[:,2]**2
            inds   = np.where(dust_r2<2*sub**2)[0]
            self.dust_pos = self.dust_pos[inds,:]
            self.dust_vel = self.dust_vel[inds,:]
            print self.dust_a
            print self.dust_a[:]

            print np.shape(self.dust_a)
            self.dust_a   = self.dust_a[inds]
        else:
            print 'No dust to subsample'
            
        return


    def max_rho(self):
        '''Find the position and velocity of the maximum gas density. Useful for polytrope locating'''
        Nc = 5000
        rho_sort = np.argsort(self.gas_rho)
        zoom_pos = np.mean(self.gas_pos[rho_sort[-Nc:],:],axis=0)
        zoom_vel = np.mean(self.gas_vel[rho_sort[-Nc:],:],axis=0)
        return zoom_pos,zoom_vel
    

    def spherical_coords(self):
        '''Transform into spherical coordinates'''
        self.gas_pos,self.gas_vel         = cart_spherical_posvel(self.gas_pos,self.gas_vel)
        self.dust_pos,self.dust_vel       = cart_spherical_posvel(self.dust_pos,self.dust_vel)
        self.planets_pos,self.planets_vel = cart_spherical_posvel(self.planets_pos,self.planets_vel)
        self.star_pos,self.star_vel       = cart_spherical_posvel(self.star_pos,self.star_vel)
        return

    
    def radial_offset(self,roffset):
        '''Radially offset spherical coord data by offset'''
        self.gas_pos[:,2]     -= roffset
        self.dust_pos[:,2]    -= roffset
        self.planets_pos[:,2] -= roffset
        self.star_pos[:,2]    -= roffset
        return

    def radial_slice(self,roffset,dr):
        '''Slice out a shell from spherical data'''
        inds = np.where((self.gas_pos[:,2]-roffset)**2<dr**2)[0]
        self.gas_pos = self.gas_pos[inds,:]
        self.gas_vel = self.gas_vel[inds,:]
        self.gas_ID  = self.gas_ID[inds]
        self.gas_h   = self.gas_h[inds]
        self.gas_u   = self.gas_u[inds]
        self.gas_rho = self.gas_rho[inds]

        try:
            #==== Subsample dust ====#
            inds   = np.where((self.dust_pos[:,2]-roffset)**2<dr**2)[0]
            self.dust_pos = self.dust_pos[inds,:]
            self.dust_vel = self.dust_vel[inds,:]
            self.dust_a   = self.dust_a[inds]
        except:
            print 'No dust!'
        return

    
    def save_as_gadget(self,filename):
        '''Convert the S class format into a Gadget snapshot format'''
        #+==== Establish new hdf5 file ====+#
        try:
            os.remove(filename)
        except OSError:
            pass
        init = h5py.File(filename)
        
        #Gas particles
        num_array = np.zeros(6)
        num_array[0] = int(np.shape(self.gas_pos)[0])
        gas_IDs = np.arange(self.N_gas)
        Type_gas = init.create_group('PartType0')
        
        Type_gas.create_dataset('Coordinates',data=self.gas_pos)
        Type_gas.create_dataset('Velocities',data=self.gas_vel)
        Type_gas.create_dataset('ParticleIDs',data=gas_IDs)
        Type_gas.create_dataset('Masses',data=np.ones(self.N_gas)*self.M_gas)
        Type_gas.create_dataset('InternalEnergy',data=self.gas_u)
        try:
            Type_gas.create_dataset('Density',data=self.gas_rho)
        except:
            print 'No density data to write'
            
        #Dust perticles
        print 'aaa', self.N_dust
        if self.N_dust != 0:
            num_array[2] = int(np.shape(self.dust_pos)[0])
            dust_IDs = np.arange(self.N_dust)+self.N_gas
            Type_dust = init.create_group('PartType2')
            Type_dust.create_dataset('Coordinates',data=self.dust_pos)
            Type_dust.create_dataset('Velocities',data=self.dust_vel)
            Type_dust.create_dataset('ParticleIDs',data=dust_IDs)
            Type_dust.create_dataset('Masses',data=np.ones(self.N_dust)*self.M_dust)
        
        #Planets and Star      
        if (self.N_planets != 0) or (self.N_star != 0): 
            num_array[5] = int(self.N_star+self.N_planets)
            if self.N_planets != 0:
                print 'h', np.shape(self.star_pos)
                print 'c', np.shape(self.planets_pos)
                sink_pos = np.vstack((self.star_pos,self.planets_pos))
                sink_vel = np.vstack((self.star_vel,self.planets_vel))
                sink_IDs = np.arange(self.N_star+self.N_planets)+self.N_gas+self.N_dust
                sink_ms  = np.hstack((np.array(self.M_star),self.M_planets))
            else:
                sink_pos = [self.star_pos]
                sink_vel = [self.star_vel]
                sink_IDs = np.arange(self.N_star)+self.N_gas+self.N_dust
                sink_ms  = np.array([self.M_star])

            print 'sink_pos', sink_pos
            Type_sink = init.create_group('PartType5')
            Type_sink.create_dataset('Coordinates',data=sink_pos)
            Type_sink.create_dataset('Velocities',data=sink_vel)
            Type_sink.create_dataset('ParticleIDs',data=sink_IDs)
            Type_sink.create_dataset('Masses',data=sink_ms)
        else:
            print 'No star or planets to save'

            
        #==== Writing Header ====#
        num_array = num_array.astype(np.int32)
        print 'Num part saved', num_array
        header = init.create_group('Header')
        header.attrs.create('NumPart_ThisFile',num_array)
        header.attrs.create('NumPart_Total',num_array.astype(np.uint32))
        header.attrs.create('NumPart_Total_HighWord',np.array([0,0,0,0,0,0]).astype(np.uint32))
        header.attrs.create('MassTable',np.array([0.,0.,0.,0.,0.,0.]))
        header.attrs.create('Time',self.headertime/ code_time *c.sec_per_year)
        header.attrs.create('Redshift',0.0)
        header.attrs.create('Boxsize',1.0)
        header.attrs.create('NumFilesPerSnapshot',1)
        header.attrs.create('Omega0',0.0)
        header.attrs.create('OmegaLambda',0.0)
        header.attrs.create('HubbleParam',1.0)
        header.attrs.create('Flag_Sfr',1)
        header.attrs.create('Flag_Cooling',1)
        header.attrs.create('Flag_StellarAge',1)
        header.attrs.create('Flag_Metals',0)
        header.attrs.create('Flag_Feedback',1)
        header.attrs.create('Flag_DoublePrecision',0)

        init.close()

        return

    def add_dust(self,tmp_M_dust,tmp_dust_pos,tmp_dust_vel):
        '''Add more dust!'''

        try:
            self.M_dust
        except:
            self.M_dust = 0
            self.N_dust = 0
            self.dust_pos = [[]]
            self.dust_vel = [[]]
        new_N_dust = len(tmp_dust_pos[:,0])

        print 'Old dust mass: ', self.N_dust*self.M_dust *c.Msol/c.ME, ' ME', self.N_dust
        print 'New dust mass: ', tmp_M_dust*new_N_dust *c.Msol/c.ME, ' ME', new_N_dust

        #Update with new dust
        self.N_dust = self.N_dust + new_N_dust
        
        if self.M_dust != tmp_M_dust:
            print 'dust masses do not match, be aware!'
            print 'Old mass', self.M_dust
            print 'New mass', tmp_M_dust
        print 'Old dust', self.dust_pos
        print 'New dust', tmp_dust_pos

        self.dust_pos = np.concatenate((self.dust_pos,tmp_dust_pos))
        self.dust_vel = np.concatenate((self.dust_vel,tmp_dust_vel))
        self.M_dust = tmp_M_dust

        print 'New total dust mass', self.N_dust*self.M_dust *c.Msol/c.ME, ' ME'
        return

    
    def move(self,offset):
        '''Move all particles to a new central coordinate'''
        self.gas_pos     -= offset
        self.dust_pos    -= offset
        self.planets_pos -= offset
        self.star_pos    -= offset
        
        return

    
def bin_data_save(filepath,runfolder,Rin,Rout,vr_mode=False,zoom='',Poly_data=False,sink_centre=False,
                  sig_plots=True,beta=0,max_dust_dens=False,ind=0):
    '''Save: 
    Time, M_star, M_gas, M_dust, N_Rbins, Rin, Rout, N_abins, amin, amax, N_planets...
    gas_count, gas_u, (gas_vr), dust_count[all_sizes]'''

    #==== Radial binning ====#
    N_Rbins  = 200
    Rbins = np.linspace(Rin,Rout,N_Rbins+1)
    Rbin_mids = (Rbins[1:]+Rbins[:-1])/2

    #==== Grain size binning ====#
    N_abins = 6
    amin,amax = -3.5,2.5
    abins     = np.logspace(amin,amax,N_abins+1)
    da_2 = (amax-amin)/2/N_abins
    abin_mids = np.logspace(amin+da_2,amax-da_2,N_abins)

    #==== Load snapshots ====#
    N_snaps = len(glob.glob(filepath+runfolder+'snapshot*'))-1
    print 'Load file: ', filepath+runfolder
    print 'N snaps: ', N_snaps
    if ind == 0:
        Snaps = np.arange(N_snaps)
    else:
        Snaps = [ind]
    save_array = np.zeros((N_snaps+1,7+N_abins+2,N_Rbins))

    #==== Find first snap ====#
    globbed = glob.glob(filepath+runfolder+'snapshot*')
    snaps = []
    try:
        slen = len(globbed[0].split('.hd')[0].split('snapshot_')[1])
        print 'slen', slen
    except:
        print 'Failed to load: ', filepath+runfolder+'snapshot'
    
    for i in range(len(globbed)):
        snaps.append(int(globbed[i][-5-slen:][:slen]))
    min_snap = min(snaps)

    #==== Fill out header info ====#
    snap0 = Load_Snap(filepath,runfolder,min_snap,slen=slen)
    save_array[0,0,0]  = snap0.headertime
    try:
        save_array[0,0,1]  = snap0.M_star
    except:
        pass
    save_array[0,0,2]  = snap0.M_gas
    save_array[0,0,3]  = snap0.M_dust
    save_array[0,0,4]  = N_Rbins
    save_array[0,0,5]  = Rin
    save_array[0,0,6]  = Rout
    save_array[0,0,7]  = N_abins
    save_array[0,0,8]  = amin
    save_array[0,0,9]  = amax
    print '\n'

    
    
    #======== Save all info for each snap =======#
    for snapid in Snaps:
        print 'Load snap', snapid
        S = Load_Snap(filepath,runfolder,snapid+min_snap,slen=slen)
        print 'star 1', S.star_pos, S.planets_pos
        save_array[snapid+1,0,0]  = S.headertime
        try:
            save_array[snapid+1,0,1]  = S.M_star
        except:
            pass
        save_array[snapid+1,0,10]  = np.mean(S.gas_pos[:,2])
        save_array[snapid+1,0,11]  = np.mean(S.dust_pos[:,2])
        print 'Snap time', S.headertime

        #==== Zoom modes ====#
        rho_sort  = np.argsort(S.gas_rho)
        clump_pos = np.mean(S.gas_pos[rho_sort[-100:],:],axis=0)
        clump_vel = np.mean(S.gas_vel[rho_sort[-100:],:],axis=0)
        save_array[snapid+1,0,12:15]  = clump_pos

        
        if zoom == 'Zrho':
            zoom_pos = clump_pos
            zoom_vel = clump_vel
        elif zoom == 'ZP':
            zoom_pos = S.planets_pos[0]
            zoom_vel = S.planets_vel[0]
        elif zoom == 'Zdust':
            #Find density of dust at all dust locations
            dust_pos_rel_clump = S.dust_pos-clump_pos
            dust_vel_rel_clump = S.dust_vel-clump_vel
            r_dust_rel_clump  = np.sqrt(dust_pos_rel_clump[:,0]**2+dust_pos_rel_clump[:,1]**2+dust_pos_rel_clump[:,2]**2)
            clump_dust_pos = S.dust_pos[np.where(r_dust_rel_clump<0.01)]
            clump_dust_vel = S.dust_vel[np.where(r_dust_rel_clump<0.01)]
            clump_dust_rho = Gadget_smooth(clump_dust_pos,S.dust_a,S.M_dust,clump_dust_pos,rho_output=True)
            dust_rho_sort  = np.argsort(clump_dust_rho)
            zoom_pos = np.mean(clump_dust_pos[dust_rho_sort[-1:],:],axis=0)
            zoom_vel = np.mean(clump_dust_vel[dust_rho_sort[-1:],:],axis=0) 
            
        if (zoom == 'Zrho') or (zoom == 'ZP') or (zoom == 'Zdust'):
            print 'Zooming on fragment!'
            S.gas_pos  = S.gas_pos - zoom_pos
            S.gas_vel  = S.gas_vel - zoom_vel
            try:
                S.dust_pos = S.dust_pos - zoom_pos
                S.dust_vel = S.dust_vel - zoom_vel
            except:
                pass
            try:
                S.planets_pos = S.planets_pos - zoom_pos
                S.planets_vel = S.planets_vel - zoom_vel
            except:
                pass

        if sig_plots == True:
            #==== Bin gas particles ====#
            r_gas  = np.sqrt(S.gas_pos[:,0]**2+S.gas_pos[:,1]**2+S.gas_pos[:,2]**2)
            gas_count = np.histogram(r_gas,Rbins)[0]
            gas_u,gas_u_sig  = calc_binned_data(S.gas_u,r_gas,Rbins)
            gas_h     = calc_binned_data(S.gas_h,r_gas,Rbins)[0]
            save_array[snapid+1,1,:] = gas_count
            save_array[snapid+1,2,:] = gas_u
            save_array[snapid+1,3,:] = gas_h
            save_array[snapid+1,5,:] = gas_u_sig

            if vr_mode == True:
                v_r, v_azi = v_r_azi(S.gas_pos,S.gas_vel)
                #spos,svel = cart_spherical_posvel(S.gas_pos,S.gas_vel)
                
                gas_vr,gas_vr_sig  = calc_binned_data(v_r,r_gas,Rbins)
                gas_vazi,gas_vazi_sig  = calc_binned_data(v_azi,r_gas,Rbins)
                r2d_gas = np.sqrt(S.gas_pos[:,0]**2+S.gas_pos[:,1]**2) 
                gas_vazi2d,gas_vazi2d_sig  = calc_binned_data(v_azi,r2d_gas,Rbins)

                save_array[snapid+1,4,:] = gas_vr
                save_array[snapid+1,6,:] = gas_vazi2d #gas_vr_sig


            #==== Bin dust species ====#
            try:
                r_dust   = np.sqrt(S.dust_pos[:,0]**2+S.dust_pos[:,1]**2+S.dust_pos[:,2]**2)
                a_ids = np.digitize(S.dust_a,abins)-1
                for j in range(N_abins):
                    dust_count = np.histogram(r_dust[a_ids==j],Rbins)[0]
                    save_array[snapid+1,7+j,:] = dust_count
            except:
                pass

            if max_dust_dens == True:
                for i in range(len(Rbins)):
                    max_rho_local = 0
                    max_dust_gas = 0

                    print r_gas
                    print np.shape(r_gas)
                    print np.shape(Rbins)
                    print np.where((r_gas>Rbins[i]))
                    Gind = np.where((r_gas>Rbins[i]) & (r_gas<Rbins[i+1]))[0]
                    print Gind
                    print Gind[0]
                    for j in range(len(Gind)):
                        rel_dust = S.dust_pos-S.gas_pos[Gind[j]]
                        r_rel_dust = np.sqrt(rel_dust[:,0]**2+rel_dust[:,1]**2+rel_dust[:,2]**2)
                        G_h = S.gas_h[Gind[j]]
                        Dind = np.where(r_rel_dust<G_h)
                        dust_mass = S.M_dust * len(Dind)
                        Vol_h = G_h**3*4/3*np.pi
                        tmp = dust_mass/Vol_h
                        if tmp > max_rho_local:
                            max_rho_local = tmp
                            max_dust_gas = max_rho_local/S.gas_rho[Gind[j]]
                    save_array[snapid+1,7+N_abins,i] = max_rho_local
                    save_array[snapid+1,7+N_abins+1,i] = max_dust_gas


                        
        #==== Save planet positions ====#
        if S.N_planets != 0:
            #Sort by planet mass. Not ideal but useful
            massinds = np.argsort(-S.M_planets)
            MP_sort = S.M_planets[massinds]
            Pos_sort = S.planets_pos[massinds,:]

            for i in range(S.N_planets):
                save_array[snapid+1,0,20] = S.N_planets
                try:
                    save_array[snapid+1,0,25+4*i] = MP_sort[i]
                    save_array[snapid+1,0,26+4*i:29+4*i] = Pos_sort[i,:]

                except:
                    print 'Too many planets for save array!'
                    save_array[snapid+1,0,20] = int((N_Rbins-25)/4)

           
        if Poly_data == True:
            rho_sort  = np.argsort(S.gas_rho)
            frag_pos  = np.mean(S.gas_pos[rho_sort[-100:],:],axis=0)
            frag_vel  = np.mean(S.gas_vel[rho_sort[-100:],:],axis=0)
            frag_uc   = np.mean(S.gas_u[rho_sort[-100:]],axis=0)
            frag_rhoc = np.mean(S.gas_rho[rho_sort[-100:]])
            frag_hc   = np.mean(S.gas_h[rho_sort[-100:]])
            frag_dr   = np.sqrt(frag_pos[0]**1+frag_pos[1]**2+frag_pos[2]**2)
            a_frag    = np.sqrt(frag_pos[0]**2+frag_pos[1]**2+frag_pos[2]**2)
            S.gas_pos     -= frag_pos
            S.gas_vel     -= frag_vel
            S.dust_pos    -= frag_pos
            S.dust_vel    -= frag_vel
            S.planets_pos -= frag_pos
            S.planets_vel -= frag_vel

            
            #==== Core offset ====#
            core_off = S.planets_pos[0]
            core_dr = mag(core_off)
            print 'Core offset', core_dr*AU_scale, ' [AU]'
            core_vtot = mag(S.planets_vel[0])

            gas_r  = np.sqrt(S.gas_pos[:,0]**2+S.gas_pos[:,1]**2+S.gas_pos[:,2]**2)
            dust_r = np.sqrt(S.dust_pos[:,0]**2+S.dust_pos[:,1]**2+S.dust_pos[:,2]**2)
            M_enc = len(gas_r[gas_r<core_dr])*S.M_gas + len(dust_r[dust_r<core_dr])*S.M_dust
            gas_rel_core = S.gas_pos-S.planets_pos[0]
            
            r_gas_rel_core = np.sqrt(gas_rel_core[:,0]**2+gas_rel_core[:,1]**2+gas_rel_core[:,2]**2)
            sort_r_gas_rel_core = np.argsort(r_gas_rel_core)
            core_uc = np.mean(S.gas_u[sort_r_gas_rel_core[:100]],axis=0)
            
            RH_2,M_frag,Macc_dust,Mean_dust_R,MD_env,R_env,E_frag,L_PP = find_RH_2(
                S,a_frag,M0=save_array[1,0,21],beta=beta)

            print 'M_enc: ', M_enc*c.Msol/c.ME, 'ME' 
            print 'L_PP', L_PP* code_M *code_L**2/code_time**3 /c.Lsol

            save_array[snapid+1,0,2]  = core_uc
            save_array[snapid+1,0,3]  = frag_rhoc
            save_array[snapid+1,0,4]  = frag_hc
            save_array[snapid+1,0,5]  = M_enc
            save_array[snapid+1,0,6]  = core_vtot
            save_array[snapid+1,0,7]  = L_PP
            save_array[snapid+1,0,15] = R_env
            save_array[snapid+1,0,16] = frag_uc
            save_array[snapid+1,0,17] = MD_env
            save_array[snapid+1,0,18] = Mean_dust_R
            save_array[snapid+1,0,19] = E_frag
            save_array[snapid+1,0,21] = M_frag
            save_array[snapid+1,0,22] = a_frag
            save_array[snapid+1,0,23] = RH_2
            save_array[snapid+1,0,24] = Macc_dust

        print '\n'            


    
    #==== Save output ====#
    print 'Saving output: '
    print savedir+runfolder.rstrip('//')+zoom+'_store'
    np.save(savedir+runfolder.rstrip('//')+zoom+'_store',save_array)
    print '#==== Binning routine complete ====#'
    
    return save_array














def animate_1d(filepath,runfolders,var1='Sigma',var2='T',rerun=False,Rin=0.1,Rout=2.0,norm_y=False,zoom='',write=False,vr_mode=False,gamma=c.gamma_dia):
    '''New function to generalise animation code
    ZP = zoom around planet
    Zrho = zoom on max rho SPH particle'''
    
    plot_dict,anim_dict,snap_list = {},{},[]
    plot_vars = [var1,var2]
    dust_scale = 1

    #======== Set up Figure ========#
    fig1 = plt.figure(1,facecolor='white',figsize=(10,6))#(6,10))
    ax1  = fig1.add_axes([0.15,0.45,0.83,0.5])
    ax2  = fig1.add_axes([0.15,0.09,0.83,0.3],sharex=ax1)
    axes = [ax1,ax2]
    ax1.semilogy()
    ax2.semilogy()
    ax2.set_xlabel('R [AU]')

    for runid in range(len(runfolders)):
        runfolder = runfolders[runid]
        try:
            print 'Try: ', savedir+runfolder.rstrip('//')+zoom+'_store.npy'
            if rerun == True:
                1/0
            save_array = np.load(savedir+runfolder.rstrip('//')+zoom+'_store.npy')
        except:
            print '#==== Need to run data binning routine! ====#'
            save_array = bin_data_save(filepath,runfolders[runid],Rin,Rout,zoom=zoom,vr_mode=vr_mode)
            
        #==== Load bin information ====#
        snap_list.append(len(save_array[1:,0,0]))
        plot_dict[str(runid)+'time'] = save_array[1:,0,0]
        N_Rbins,Rin,Rout = save_array[0,0,4],save_array[0,0,5],save_array[0,0,6]
        Rbins = np.linspace(Rin,Rout,N_Rbins+1)
        Rbin_mids = (Rbins[1:]+Rbins[:-1])/2
        dRbin = Rbins[1]-Rbins[0]
        Rbin_areas = Rbin_mids*2*np.pi*dRbin
        Rbin_volumes = 4*np.pi/3 * (Rbins[1:]**3-Rbins[:-1]**3)
        N_abins,amin,amax = int(save_array[0,0,7]),save_array[0,0,8],save_array[0,0,9]
        abins     = np.logspace(amin,amax,N_abins+1)
        M_gas,M_dust = save_array[0,0,2],save_array[0,0,3]
        
        #============ Construct plotting dictionaries for appropriate variables ============#
        #===================================================================================#
        for i in range(len(plot_vars)):
            
            #============= Plot Sigma =============#
            if plot_vars[i] == 'Sigma':
                gas_sig = save_array[1:,1,:]*M_gas/Rbin_areas*code_M/code_L**2
                scale_fac = Rbin_mids*code_L/c.AU
                
                plot_dict[str(runid)+'_'+str(i)] = gas_sig*scale_fac
                axes[i].set_ylabel(r'$\Sigma$ R [gcm$^{-2}$ AU]')
                if norm_y == True:
                    plot_dict[str(runid)+'_'+str(i)] = gas_sig/gas_sig[0,:]
                    axes[i].set_ylabel(r'Normalised $\Sigma$')

                #-------- Dust Sigma --------#
                scale_fac *= dust_scale
                for j in range(N_abins):
                    dust_sig = save_array[1:,7+j,:]*M_dust/Rbin_areas*code_M/code_L**2
                    plot_dict[str(runid)+'_'+str(i)+'_'+str(j)] = dust_sig*scale_fac
                    if norm_y == True:
                        plot_dict[str(runid)+'_'+str(i)+'_'+str(j)] = dust_sig/dust_sig[0,:]


                        
            #============= Plot Rho =============#
            if plot_vars[i] == 'rho':
                gas_rho = save_array[1:,1,:]*M_gas/Rbin_volumes*code_M/code_L**3
                
                plot_dict[str(runid)+'_'+str(i)] = gas_rho
                axes[i].set_ylabel(r'$\rho$ [gcm$^{-3}$]')
                axes[i].set_ylim(np.min(gas_rho)/1.5,np.max(gas_rho)*1.5)
                if norm_y == True:
                    plot_dict[str(runid)+'_'+str(i)] = gas_rho/gas_rho[0,:]
                    axes[i].set_ylabel(r'Normalised $\rho$')

                #-------- Dust rho --------#
                scale_fac = dust_scale
                for j in range(N_abins):
                    dust_rho = save_array[1:,7+j,:]*M_dust/Rbin_volumes*code_M/code_L**3
                    plot_dict[str(runid)+'_'+str(i)+'_'+str(j)] = dust_rho*scale_fac
                    if norm_y == True:
                        plot_dict[str(runid)+'_'+str(i)+'_'+str(j)] = dust_rho/dust_rho[0,:]

                        
            #========= Plot Temperature =========#
            if plot_vars[i] == 'T':
                gas_T = save_array[1:,2,:]*(gamma-1)*c.mu*c.mp/c.kb*code_L**2/code_time**2
                gas_T_sig = save_array[1:,5,:]*(gamma-1)*c.mu*c.mp/c.kb*code_L**2/code_time**2
                plot_dict[str(runid)+'_'+str(i)] = gas_T
                plot_dict[str(runid)+'_'+str(i)+'_sig-'] = gas_T - gas_T_sig
                plot_dict[str(runid)+'_'+str(i)+'_sig+'] = gas_T + gas_T_sig
                axes[i].set_ylabel(r'Temperature [K]')

            #======== Plot smoothing length ========#
            if plot_vars[i] == 'h':
                gas_h = save_array[1:,3,:]*AU_scale #[AU]
                plot_dict[str(runid)+'_'+str(i)] = gas_h
                axes[i].set_ylabel(r'Smoothing Length [AU]')
                
            #========= Plot Radial Velocity ===========#
            if plot_vars[i] == 'vr':
                gas_vr = save_array[1:,3,:]*code_L/code_time
                gas_vr_sig = save_array[1:,6,:]*code_L/code_time
                plot_dict[str(runid)+'_'+str(i)] = gas_vr
                plot_dict[str(runid)+'_'+str(i)+'_sig-'] = gas_vr - gas_vr_sig
                plot_dict[str(runid)+'_'+str(i)+'_sig+'] = gas_vr + gas_vr_sig
                axes[i].set_ylabel(r'$Gas V_R [cms^{-1}]$')

            #========= Plot Dust to gas ratio =========#
            if plot_vars[i] == 'dust_gas':
                dust_gas = np.sum(save_array[1:,4:4+N_abins,:],axis=1) /save_array[1:,1,:] *M_dust/M_gas
                plot_dict[str(runid)+'_'+str(i)] = dust_gas 
                axes[i].set_ylabel('Dust to Gas Ratio')
                dg_0 = axes[i].axhline(M_dust/M_gas,color=run_cols[2],ls='--')
                if norm_y == True:
                    plot_dict[str(runid)+'_'+str(i)] *= M_gas/M_dust 
                    axes[i].set_ylabel('Normalised Dust to Gas Ratio')
                    dg_0.set_ydata(1)

            #========= Plot enclosed mass =========#
            if plot_vars[i] == 'M_enc':
                M_enc = np.cumsum(save_array[1:,1,:],axis=1)*M_gas
                plot_dict[str(runid)+'_'+str(i)] = M_enc
                axes[i].set_ylabel(r'Enclosed Mass [$M_{\odot}$]')
                if norm_y == True:
                    plot_dict[str(runid)+'_'+str(i)] = M_enc/M_enc[:,-1][:,None]
                    axes[i].set_ylabel(r'Norm Enclosed Mass')

                for j in range(N_abins):
                    M_enc_dust = np.cumsum(save_array[1:,4+j,:],axis=1)*M_dust
                    plot_dict[str(runid)+'_'+str(i)+'_'+str(j)] = M_enc_dust
                    if norm_y == True:
                        plot_dict[str(runid)+'_'+str(i)+'_'+str(j)] = M_enc_dust/M_enc_dust[:,-1][:,None]

                    
            #======== Plot Collison Velocity =======#
            if plot_vars[i] == 'Vcoll':
                Vcoll = save_array[1:,11,:]*code_L/code_time
                plot_dict[str(runid)+'_'+str(i)] = Vcoll
                axes[i].set_ylabel(r'$Dust Collision Velocity [cms^{-1}]$')

            #======== Plot Convective Stability A (A<1=stable, A>1=convection) ====#
            if plot_vars[i] == 'Con_stab':
                gas_rho   = save_array[1:,1,:]*M_gas/Rbin_volumes*code_M/code_L**3
                gas_u     = save_array[1:,2,:]*code_L**2/code_time**2
                gas_T     = gas_u * (gamma-1)*c.mu*c.mp/c.kb
                drho_dr   = (gas_rho[:,1:]-gas_rho[:,:-1])/dRbin/code_L
                dT_dr     = (gas_T[:,1:]-gas_T[:,:-1])/dRbin/code_L
                rho_mids  = (gas_rho[:,1:]+gas_rho[:,:-1])/2
                T_mids    = (gas_T[:,1:]+gas_T[:,:-1])/2

                A = 1/gamma * (dT_dr/T_mids - (gamma-1)*drho_dr/rho_mids)
                tmp = np.zeros(np.shape(gas_u))
                tmp[:,1:] = A
                plot_dict[str(runid)+'_'+str(i)] = tmp
                axes[i].set_ylabel(r'Convective Stability A')
                axes[i].set_yscale('symlog',linthreshy=1e-16)

                
            #===------------= Establish animation objects =------------===#
            anim_dict[str(runid)+'_'+str(i)] = axes[i].plot(Rbin_mids*AU_scale,
                    plot_dict[str(runid)+'_'+str(i)][0],color=run_cols[runid],
                    label=str(runfolders[runid]))
            
            #-------- Dust lines --------#
            if (plot_vars[i]=='Sigma') or (plot_vars[i]=='M_enc') or (plot_vars[i]=='rho'):
                for j in range(N_abins):
                    anim_dict[str(runid)+'_'+str(i)+'_'+str(j)] = axes[i].plot(
                        Rbin_mids*AU_scale,plot_dict[str(runid)+'_'+str(i)+'_'+str(j)][0],
                        color=run_cols[runid],ls=linestyles[j],lw=linewidths[j])
                    if runid == len(runfolders)-1:
                        anim_dict[str(runid)+'_'+str(i)+'_'+str(j)][0].set_label(
                            'a = {:.3f}'.format(abins[j])+'-{:.3f}'.format(abins[j+1])+'cm')

            #==== Temp and vr variance ====#
            if (plot_vars[i] == 'T') or (plot_vars[i] == 'vr'):
                anim_dict[str(runid)+'_'+str(i)+'_sig-'] = axes[i].plot(
                    Rbin_mids*AU_scale,plot_dict[str(runid)+'_'+str(i)+'_sig-'][0],color=run_cols[runid],ls='--')
                anim_dict[str(runid)+'_'+str(i)+'_sig+'] = axes[i].plot(
                    Rbin_mids*AU_scale,plot_dict[str(runid)+'_'+str(i)+'_sig+'][0],color=run_cols[runid],ls='--')

            #====== Planet lines ======#
            N_planets = int(np.max(save_array[1:,0,20]))
            print 'N_planets ani', N_planets
            plot_dict[str(runid)+'NP'] = N_planets
            for p in range(N_planets):
                MPs = save_array[1:,0,25+2*p]*code_M/c.MJ
                RPs = save_array[1:,0,26+2*p]*AU_scale
                plot_dict[str(runid)+'p'+str(p)+'_MPs'] = MPs
                plot_dict[str(runid)+'p'+str(p)+'_RPs'] = RPs
                anim_dict[str(runid)+'_RP_'+str(i)+'_'+str(p)] = axes[i].axvline(
                    RPs[0],lw=1,color=run_cols[runid])
                if i == 1:
                    anim_dict[str(runid)+'_MPtext_'+str(i)+'_'+str(p)] = ax2.text(
                        0.8,0.8+runid*0.05,r'M$_P$ {:.2f} M$_J$'.format(MPs[1]),transform=ax2.transAxes)

                    
    #======= Check that time arrays are self-consistent =======#
    print 'Need to write time array self-consistent check'


    #=========================== Plotting code ===========================#
    #=====================================================================#
    timetext = ax1.text(0.06,0.91,'Time: {:.2f}'.format(plot_dict[str(runid)+'time'][0])
                        + ' Years',transform=ax1.transAxes)
    N_frames = np.min(snap_list)
    print 'N frames', N_frames
    ax1.legend(frameon=False)

    def animate(anim_i):
        output = []
        for runid in range(len(runfolders)):
            for i in range(len(plot_vars)):
                anim_dict[str(runid)+'_'+str(i)][0].set_ydata(plot_dict[str(runid)+'_'+str(i)][anim_i])
                N_planets = plot_dict[str(runid)+'NP']
                for p in range(N_planets):
                    anim_dict[str(runid)+'_RP_'+str(i)+'_'+str(p)].set_xdata(
                        plot_dict[str(runid)+'p'+str(p)+'_RPs'][anim_i])
                    if i == 1:
                        anim_dict[str(runid)+'_MPtext_'+str(i)+'_'+str(p)].set_text(
                            r'M$_P$ {:.2f} M$_J$'.format(plot_dict[str(runid)+'p'+str(p)+'_MPs'][anim_i]))

                for j in range(N_abins):
                    try:
                        anim_dict[str(runid)+'_'+str(i)+'_'+str(j)][0].set_ydata(
                            plot_dict[str(runid)+'_'+str(i)+'_'+str(j)][anim_i])
                    except:
                        pass
                if (plot_vars[i] == 'T') or (plot_vars[i] == 'vr'):
                    anim_dict[str(runid)+'_'+str(i)+'_sig-'][0].set_ydata(
                        plot_dict[str(runid)+'_'+str(i)+'_sig-'][anim_i])
                    anim_dict[str(runid)+'_'+str(i)+'_sig+'][0].set_ydata(
                        plot_dict[str(runid)+'_'+str(i)+'_sig+'][anim_i])

        
        timetext.set_text('Time: {:.2f}'.format(plot_dict[str(runid)+'time'][anim_i]) + ' Years')
        output.append(timetext)
        output.append(anim_dict)
        return output

    #ani = animation.FuncAnimation(fig1, animate, interval=80, frames=N_frames, blit=False, repeat=True)
    ani = player.Player(fig1, animate, interval=80, maxi=N_frames, blit=False, repeat=True,save_count=N_frames-1)
    plt.show()

    if write == True:
            print 'Writing savefile'
            writer = animation.writers['ffmpeg'](fps=5)
            print runfolder.strip('//')+'_'+var1+'_'+var2+'.mp4'
            ani.save(runfolder.strip('//')+'_'+var1+'_'+var2+'.mp4',writer=writer)
    
    return





    
def subsample_dust_size(pos,a,amin,amax):
    '''Subsamples dust data to select only grains of a certain grain size a'''
    inds = []
    tol = 1.00001
    for i in range(len(a)):
        if (a[i]>amin*tol) & (a[i]<=amax*tol):
            inds.append(i)
    print len(inds), 'dust particles in this size bin'
    pos = pos[inds,:]
    a   = a[inds]
    
    return pos,a

    
def plot_thermalvel():
    Ts = np.logspace(1,5,100)
    T_vels = thermal_vel(Ts)
    plt.plot(Ts,T_vels)
    plt.semilogx()
    plt.semilogy()
    plt.show()


def temp_floors():
    Rs = np.arange(200)
    
    Gadget_T = 20*(Rs/100)**-0.5
    Seren_T  = 250*Rs**-0.75

    plt.figure(1,facecolor='w')
    plt.plot(Rs,Gadget_T,label='Gadget 3 (Humphries 2018)')
    plt.plot(Rs,Seren_T,label = 'Seren (Stamatellos 2018)')
    plt.legend(frameon=False)
    return



def find_f_peb(M_largedust,amin,amax,apow,amid=0,Z0=0.01,M_gas=0,N_gas=0):
    '''For coupled and uncoupled dust mix, find f_peb
    amin < coupled dust < amid < uncoupled dust < amax

    If no large dust, use Z0 and M_gas'''

    #==== Backup in case no large dust ====#
    if (M_largedust == 0) or (amid == amax):
        Mtot_dust = Z0 * M_gas * N_gas
        f_peb = 0
        return Mtot_dust, f_peb
    
    #Find normalisation factor from large dust mass (known)
    if apow == 4:
        fac = np.log10(amax)-np.log10(amid)
    else:
        fac = (amax**(4-apow)-amid**(4-apow)) / (4-apow)
    A = M_largedust / fac

    if apow == 4:
        facB = np.log10(amax)-np.log10(amin)
    else:
        facB = (amax**(4-apow)-amin**(4-apow)) / (4-apow)
    Mtot_dust = A * facB

    f_peb = M_largedust/Mtot_dust

    return Mtot_dust, f_peb



def plot_dust_mass_frac(Mtot_dust,amin,amax,apow):
    '''Plot the fraction of mass in each grain size'''
    a = np.logspace(np.log10(amin),np.log10(amax),100)
    a_mids = 10**((np.log10(a[1:])+np.log10(a[:-1]))/2)
    
    if apow == 4:
        fac = np.log10(amax)-np.log10(amin)
        A = Mtot_dust/fac
        M_cum = A * (np.log10(a)-np.log10(amin))
        M_bin = A * (np.log10(a[1:])-np.log10(a[:-1]))
        
    else:
        fac = (amax**(4-apow)-amin**(4-apow)) / (4-apow)
        A = Mtot_dust/fac
        M_cum = A * (a**(4-apow)-amin**(4-apow)) / (4-apow)
        M_bin = A * (a[1:]**(4-apow)-a[:-1]**(4-apow)) / (4-apow)
        
    fig0 = plt.figure(0,facecolor='w')
    ax1 = fig0.add_axes([0.1,0.55,0.8,0.4])
    ax2 = fig0.add_axes([0.1,0.1,0.8,0.4],sharex=ax1)

    ax1.set_ylabel('M_frac')
    ax1.plot(a,M_cum)
    ax1.semilogy()
    ax1.semilogx()

    ax2.bar(a[:-1],M_bin,log=True)
    ax2.set_xlabel('Grain size [cm]')
    ax2.set_ylabel('Mass per bin')
    return


def Tsukagoshi_data():
    '''Add Tsukagoshi TW Hya intensity profile to a rad av intensity profile'''

    TWHya_file = '/home/r/rjh73/code/Obs_data/TW_Hya_Tsukagoshi_2019.csv'
    TWHya_dat = np.genfromtxt(TWHya_file,delimiter=',')

    return TWHya_dat



def rad_av_plot(figi,im,bmean,xextent,yextent,two_sided=True,TWHya=False,two_panel=True):
    '''Make radially averaged flux plot. For analysing ALMA things
    Beamsize in AU'''
    
    if two_panel == True:
        fig = plt.figure(figi+10,facecolor='w',figsize=(9,4))
        ax2 = fig.add_axes([0.15,0.15,0.35,0.8])
        ax3 = fig.add_axes([0.6,0.15,0.35,0.8],sharex=ax2)

    else:
        fig = plt.figure(figi+10,facecolor='w',figsize=(6,5))
        ax2 = fig.add_axes([0.2,0.15,0.75,0.8])
        ax3 = fig.add_axes([-10,-10,0.8,0.8])
        
    npixx = np.shape(im)[0]
    npixy = np.shape(im)[1]
    pixx,pixy = 2*xextent/npixx,2*yextent/npixy
    xs = np.linspace(-xextent,xextent,npixx+1)
    ys = np.linspace(-yextent,yextent,npixy+1)
    x_mids = (xs[1:]+xs[:-1])/2
    y_mids = (ys[1:]+ys[:-1])/2

    #im_beam  = im /  (pixx*pixy/c.AU**2) * (bmean**2*np.pi)
    #im_beam  = im /  (pixx*pixy) * (bmean**2*np.pi)
    im_beam = im

    if two_sided == True:
        r, im_rA, im_rA_stds, im_rB, im_rB_stds = radial_profile(x_mids,y_mids,im_beam,(0,0),two_sided=True)
        r_full = np.linspace(-r[-1],r[-1],2*len(r)-1)
        im_r_full = np.concatenate((im_rA[::-1],im_rB[1:]))
        ax2.set_xlim(-100,100)
        ax3.set_xlim(-100,100)
    else:
        r_full, im_r_full, im_rA_stds = radial_profile(x_mids,y_mids,im_beam,(0,0),two_sided=False)
        ax2.set_xlim(0,100)
        ax3.set_xlim(0,100)

    im_r_full = np.ma.array(im_r_full, mask=np.isnan(im_r_full))
    ax2.plot(r_full,im_r_full,label='Model dust 52 AU')
    ax3.plot(r_full,im_r_full)
        
    ax2.set_xlabel('R [AU]')        
    ax3.set_xlabel('R [AU]')        
    ax2.set_ylabel('Intensity [mJy/beam]')
    ax2.set_ylim(np.max(im_r_full)/1000,np.max(im_r_full)*2)
    ax3.set_ylim(np.min(im_r_full),np.max(im_r_full)*1.2)
    ax2.semilogy()    

    #plt.yscale('symlog',linthreshy=np.max(im_r_full)/100)

    if TWHya == True:
        print 'Adding Tsukagoshi 2019 TW Hya data'
        TWHya_dat = Tsukagoshi_data()
    
        print np.shape(TWHya_dat)
        print TWHya_dat[:,0]
        print TWHya_dat[:,1]
        ax2.scatter(TWHya_dat[:,0],TWHya_dat[:,1],color='red',s=2,label='TW Hya (Tsukagoshi Et al. 2019)')
        ax3.scatter(TWHya_dat[:,0],TWHya_dat[:,1],color='red',s=2)
        ax2.legend(frameon=False,loc=1)

    return 





def beam_flux(image_data,xextent,yextent,npixx,npixy,pos_beam,bmaj,bmin):
    '''Calculate flux inside some beam'''

    xs = np.linspace(-xextent,xextent,npixx+1)
    ys = np.linspace(-yextent,yextent,npixy+1)
    x_mids = (xs[1:]+xs[:-1])/2
    y_mids = (ys[1:]+ys[:-1])/2
    X_mids,Y_mids = np.meshgrid(x_mids,y_mids,indexing='xy')
    Xrel_beam,Yrel_beam = X_mids-pos_beam[0],Y_mids-pos_beam[1]
    Rrel_beam = np.sqrt(Xrel_beam**2+Yrel_beam**2)
    
    bmean = (bmaj+bmin)/2
    inds = np.where(np.ravel(Rrel_beam)<bmean/2)

    Fluxes      = np.ravel(image_data)[inds]
    Beam_mean   = np.mean(Fluxes)
    Beamsq_mean = np.mean(Fluxes**2)
    Beam_std    = np.sqrt(Beamsq_mean-Beam_mean**2)
    
    #Beam_flux = np.sqrt(np.mean(np.ravel(image_data)[inds]**2))
    #return Beam_flux

    return Beam_mean, Beamsq_mean, Beam_std



    

def fits_image(froot,fol_name,fname,figi=0,d=140,contrasts=False,
               cycle='',int_time='',cycle2='',int_time2='',
               show_fig=True,rad_av=False,cbar=True,log=False,beam_convolve=False,
               two_sided=False,TWHya=False):
    '''Function to make an image from an astronomical fits file
    d [pc] '''

    mJy_scale = 1000

    print 'PATH', froot+fol_name+fname
    hdul = fits.open(froot+fol_name+fname)
    wcs = WCS(hdul[0].header).celestial

    #Dimensions
    npixx,npixy = hdul[0].header['NAXIS1'],hdul[0].header['NAXIS2']
    pixx = abs(hdul[0].header['CDELT1']/180*np.pi*(d*c.pc/c.AU)) #AU
    pixy = abs(hdul[0].header['CDELT2']/180*np.pi*(d*c.pc/c.AU)) #AU
    xextent,yextent = npixx*pixx/2,npixy*pixy/2
    print 'Pixels', pixx,pixy, npixx,npixy
    print 'extents', xextent,yextent

    try:
        bmaj_as = hdul[0].header['BMAJ']*3600 #arcsec
        bmin_as = hdul[0].header['BMIN']*3600 #arcsec
        bmaj = bmaj_as * np.pi/180/3600 *d*c.pc/c.AU #AU
        bmin = bmin_as * np.pi/180/3600 *d*c.pc/c.AU #AU
        bmean = (bmaj+bmin)/2
        bpa = hdul[0].header['BPA'] #degrees
        print 'BMAJ,BMIN,BPA', bmaj,bmin,bmaj_as,bmin_as,bpa
        Beamsize = bmaj*bmin/4*np.pi
        BS = patches.Ellipse((-0.8*xextent,-0.8*yextent), width=bmaj,height=bmin,angle=bpa,color='w')
    except:
        bmean = 3
        Beamsize = bmean**2*np.pi
        pass

        
    image_data = fits.getdata(froot+fol_name+fname,ext=0)[0] #Jy
    hdul.close()
    if len(np.shape(image_data)) == 3:
           image_data = image_data[0]

    if beam_convolve == True:
        print 'Beam convolved manually'
        image_data = image_data /pixx/pixy*(bmean**2*np.pi)

    image_data *= 1000 #mJy
    immin = 0#5e-5*mJy_scale#np.min(image_data)
    immax = np.max(image_data)
    #immin = 0.001
    print 'immax', immax
    norm_image_data = image_data /immax #Normalised

    print 'FITS dims:', np.shape(image_data)
    


    if show_fig == True:
        fig = plt.figure(figi,facecolor='w',figsize=(4.5,4))
        #ax1 = plt.subplot()#projection=wcs)
        ax1 = fig.add_axes([0.12,0.1,0.755,0.85])

        if log == False:
            im = ax1.imshow(norm_image_data,interpolation='none',cmap='magma',
                            vmin=0,vmax=1,
               extent=[-xextent,xextent,-yextent,yextent])
            ticks = np.linspace(0,1,6)
        elif log == True:
            im = ax1.imshow(abs(norm_image_data),interpolation='none',cmap='magma',
            #norm=SymLogNorm(1e-2),
            norm=LogNorm(vmin=immax/100,vmax=immax),
            extent=[-xextent,xextent,-yextent,yextent])
            ticks = np.logspace(-3,0,4)
        if cbar == True:
            cax  = fig.add_axes([0.875,0.1,0.03,0.85])
            cb = plt.colorbar(im,cax=cax,ticks=ticks)
            
        try:
            ax1.add_artist(BS)
        except:
            pass
        
        ax1.text(0.55,0.95,r'Beam: '+'{:.0f}'.format(bmean)+' AU',transform=ax1.transAxes,color='w')
        ax1.text(0.55,0.88,r'{:.2g}'.format(immax)+' mJy/beam',transform=ax1.transAxes,color='w')
        if cycle != '':
            ctxt = r'Config: '+'{:.1f}'.format(cycle)
            if cycle2 != '':
                ctxt += '+{:.1f}'.format(cycle2)
            ax1.text(0.05,0.95,ctxt,transform=ax1.transAxes,color='w')
            if int_time != '':
                ttxt = '{:.0f} min'.format(int_time)
                if int_time2 != '':
                    ttxt = ttxt[:-4] + '+{:.0f} min'.format(int_time2)
                ax1.text(0.05,0.88,ttxt,transform=ax1.transAxes,color='w')
        
    if rad_av == True:
        #Make radially averaged plots
        fig2 = rad_av_plot(figi,image_data,bmean,xextent,yextent,two_sided=two_sided,TWHya=TWHya,two_panel=False)
        
    if contrasts == True:
        r_PP = int(fol_name.split('r')[1].split('_')[0])
        pos_PP = [r_PP,0]
        print 'r_PP', r_PP
        PP_mean,PPsq_mean,PP_std = beam_flux(image_data,xextent,yextent,npixx,npixy,pos_PP,bmaj,bmin)
        PP_RMS = np.sqrt(PPsq_mean)
        
        #BG flux
        BG_fluxes = np.ravel(image_data[:200,:])
        BG_mean = np.mean(BG_fluxes)
        BG_std = np.std(BG_fluxes)
        BG_RMS = np.sqrt(np.mean(BG_fluxes**2))
        
        #Mean disc flux
        angles = np.linspace(0,1,10)*np.pi*2
        rin=10
        rout=30
        radii = np.random.rand(len(angles))*(rout-rin)+rin
        posx_disc = radii*np.cos(angles)
        posy_disc = radii*np.sin(angles)
        disc_fluxes = []
        for i in range(len(angles)):
            disc_fluxes.append(beam_flux(image_data,xextent,yextent,npixx,npixy,
                                         [posx_disc[i],posy_disc[i]],bmaj,bmin))
        disc_flux = np.mean(disc_fluxes)

        PP_disc_SNR = PP_mean/disc_flux
        
        PP_SNR = (PP_mean)/BG_std 
        print 'SNR', PP_SNR
        print 'PP_mean flux', PP_mean
        print 'BG_std', BG_std

        #plt.hist(np.ravel(image_data[:200,:]))
        #plt.show()
        
        if show_fig == True:
            ax1.text(0.55,0.81,r'$PP_{SNR}$: '+'{:.2f}'.format(PP_SNR),
                     transform=ax1.transAxes,color='w')
            

        return PP_SNR,PP_mean,PP_std,BG_mean,BG_std, PP_disc_SNR

    else:
        return 1,1,1,1,1,1



def dust_mass_flux(lam,T,d,Fb=0,M_dust=0):
    '''Observation function: calculate dust mass from integrated flux in Jy
    lam [mm], d [pc], Fb [Jy/beam], M_dust [ME]'''

    if Fb != 0:
        F = 10**(-23) * Fb  #erg/s/cm^2 #integrated over dv
    elif M_dust != 0:
        M_dust *= c.ME #g
        
    lam /= 10 #cm
    v = c.c / lam #/s
    d *= c.pc #pc
    Kv = 2.3 #g/cm^2 at lam = 1.3mm
    #Kv = 2.8 g/cm^2 at lam = 0.87mm #Andrews 2012? Tsukagoshi 2019
    Lv = 2 * c.h * v**3 /c.c**2  * 1/(np.exp(c.h*v/(c.kb*T))-1 )#Spectral radiance erg/s/cm^2 /Sr/ Hz 

    if Fb != 0:
        M_dust = F * d**2 /Kv /(Lv) #g   
        print 'M_dust: ', M_dust /c.ME
    elif M_dust != 0:
        F = M_dust / (d**2 /Kv /(Lv)) *1000
        print 'F: ', F /10**(-23), 'mJy'
        
    return

    

    
if __name__ == "__main__":
    filepath='/rfs/TAG/rjh73/Gio_disc/'
    filepath = '/rfs/TAG/rjh73/Clump_project/'
    runfolders = ['Gio_N1e6_aav01_R00120_MD01_Z10_MP03/','Gio_N1e6_aav01_R00120_MD01_Z10_MP3/']
    runfolders = ['P1e5_M5_R3_b5_rho2e11_r60_T30_Ti34/','P1e5_M5_R3_b5_rho2e11_r60_T30_Ti12/']
    #temp_floors()
    #animate_1d(filepath=filepath,runfolders=runfolders,rerun=False,var2='M_enc',zoom='Zrho',Rin=0.001,Rout=0.04)
    #plot_dust_mass_frac(1,1e-6,1,3.5)

    
    #dust_mass_flux(lam=1.3,T=30,d=140,Fb=6.5e-4*0.48,) #PP project
    #dust_mass_flux(lam=1.3,T=18,d=59.5Fb=2.5e-4,) #Tsukagishi
    #dust_mass_flux(lam=1.3,T=18,d=59.5,Fb=6.5e-4*0.48,) #PP proj as Tsuka
    dust_mass_flux(lam=1.3,T=18,d=59.5,M_dust=0.028,) #PP proj as Tsuka

    plt.show()
