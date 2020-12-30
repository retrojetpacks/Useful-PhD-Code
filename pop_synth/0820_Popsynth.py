'''Functions to run population sysnthesis models'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import basic_astro_functions as b
import astrophysical_constants_cgs as c
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from datetime import datetime
import ndtest as KS

#Font styles, sizes and weights
def set_rcparams(fsize=12):
    font = {'weight' : 'normal',
            'size'   : fsize,
            'serif' : 'Minion Pro'}
    plt.rc('font', **font)
    plt.rc('legend', fontsize=fsize)
    plt.rc('lines',linewidth = 1.5)
    
    return
set_rcparams()



Q_crit = 1.4 #Lodato & Rice 2005

t_GImax = 1e5*c.sec_per_year
t_max  = 1e7*c.sec_per_year  #End point
t_infall = 1e4*c.sec_per_year
#t_acc  = 1e50*c.sec_per_year  #planet accretion time
eta = 100.#For the Malik+15 opening gap condition
RH_frac = 2/3
Q_thresh = 1.4#2

#Gas accretion tests
#Gas_pow = -1.44
#Gas_c   = 6e5

plot_cols = ['#2EC4C6','#FFC82E','#CD6622','#FF1493','#440044','#6a5acd','#33cc33','#0055ff','#dd44cc','#cd6622','#A8CFFF','#62D0FF',
            '#2EC4C6','#FFC82E','#CD6622','#FF1493','#440044','#6a5acd','#33cc33','#0055ff','#dd44cc','#cd6622','#A8CFFF','#62D0FF'
]
lss  = ['-','--',':','-.','-','-','-','-']
syms = ['o','v','s','D','p','+','2','*']

#fol = 'C:\Users\Colin\Documents\PhD\Pop Synth Project\\'
fol = '/home/r/rjh73/code/pop_synth/'
obs_fol = '/home/r/rjh73/code/Obs_data/'




#============ Define analytic disc properties ============#

def calc_analytic_disc(t,A,M_star,M_disc,args0):
    '''Calculate analytic disc properties
    A[cm], M[g]'''

    Rin      = args0['Rin']
    Rout     = args0['Rout']
    T100     = args0['T100']
    t_disp   = args0['t_disp'] 
    alpha0   = args0['alpha0']
    try:
        SG_corr = args0['SG_corr']
    except:
        SG_corr = False
    
    #M_disc = M_disc0 * np.exp(-t/t_disp)    #
    #M_star = M_star0 + M_disc0*(1 - np.exp(-t/t_disp))
    
    T    = T100 * (A/(100*c.AU))**-0.5
    cs   = np.sqrt(c.kb*T/c.mu/c.mp)
    HtoR = np.sqrt(c.kb*T*A / (c.mu*c.mp*c.G*M_star))
    H    = A*HtoR
    Om_K = np.sqrt(c.G*M_star/A**3)
    
    Sig0 = M_disc / (2*np.pi*(Rout-Rin))
    Sig  = Sig0 * (1/A)
    Q    = cs*Om_K / (np.pi*c.G*Sig)
    Qout = Q * (Rout/A)**(-3/4)

    #Self-grav correction
    if SG_corr == True:
        #print cs,Sig,c.G,A
        HtoR_SG = cs**2/(np.pi*Sig*c.G)/A
        HtoR_NSG = HtoR
        HtoR = min(HtoR_SG,HtoR_NSG)
        #print HtoR,HtoR_SG,HtoR_NSG
        if HtoR == HtoR_SG:
            print 'SG at', t/c.sec_per_year
        H = A*HtoR
    
    return T,cs,HtoR,H,Om_K,Sig,Q,Qout



def load_allona_data(file1):
    '''Load data from Allona's clump evolution simulations'''
    #folder = 'Allona_Jupiters/'
    folder = '181104_Allona_Jupiters/'
    data = np.genfromtxt(fol+folder+file1,skip_header=3)
    
    Time  = data[:,1]*c.sec_per_year #Years
    Tc    = data[:,2]                #K
    Rho_c = data[:,6]                #g/cm^3 
    R_P   = data[:,7]*c.Rsol         #cm
    T_eff = data[:,11]               #K
    L     = data[:,12]               #Solar L
    
    return Time,Tc,Rho_c,R_P,T_eff,L


def post_disrupt_mass(a,R_P,M_P,M_star,Sig,H,tau,t):
    '''Calculate the post disruption mass for a planet by recalculating RH
    Power law - rho = AR^m
    '''

    #time, layer number, x, mass, radius, P, T, L, rho, internal energy in this layer, entropy, opacity, x, ...
    mj3 = 'out3D_3Mj_clmpev_isol'
    data = np.genfromtxt(fol+mj3,skip_header=0)
    data = np.reshape(data,(-1,150,19))
    times = data[:,0,0]
    tind = np.argmin((times-tau/c.sec_per_year)**2)
    
    snapdata = data[tind]
    Rs    = snapdata[:,4]*c.Rsol
    rho_R = snapdata[:,8]
    Ts    = snapdata[:,6]
    Vols = np.zeros(len(Rs)+1)
    Vols[1:] = 4*np.pi/3*Rs**3
    M_encs = np.cumsum((Vols[1:]-Vols[:-1])*rho_R)
    RHs    =  a * (M_encs/3/M_star)**(1/3)

    diff_R = Rs - RHs*RH_frac

    ind = np.where(diff_R>0)[0][0] #Choose first positive value
    M_post = M_encs[ind] #Sum up enclosed mass inside ind..
    R_post = Rs[ind]
    
    print t/c.sec_per_year, M_post/c.MJ
    #print '\n'
    return M_post, R_post
    
    
    

def migrate(t,A,M_P,M_star,M_disc,M_infall,args0,Model):
    '''Perform one migration time step'''

    T,cs,HtoR,H,Om_K,Sig,Q,Qout = calc_analytic_disc(t,A,M_star,M_disc,args0)
    RH = A * (M_P/(3*M_star))**(1/3)
    alpha0 = args0['alpha0']
    alpha_sg = args0['alpha_sg']
    t_disp = args0['t_disp']
    typeI = True
    Md = np.pi * Sig * A**2  #'local disc mass'
    mu = M_disc/(M_disc+M_star) #For Kratter08
    
    if Model == 'NF15':
        alpha = alpha0 + alpha_sg * Q_crit**2/(Q_crit**2 + Q**2)
        Crida_p = 0.75*H/RH + 50.*alpha*HtoR**2 * M_star/M_P
        if Crida_p > 1:
            t_migr = 0.8/Om_K * M_star**2/(M_P * Md) * HtoR**2 #type I
        else:
            typeI = False
            t_migr = 1./(alpha*Om_K*HtoR**2) * (1. + M_P/Md) #type II

    elif Model == 'M+18':
        #This describes alpha_self_grav after Kratter,Matzner,Krumholz+08
        #They use Q_out at disc outer boundary
        '''
        if Qout < 2.:
            alpha_sh = 0.14*(1.3**2/Qout**2 - 1.)*(1. - mu)**1.15
            if (alpha_sh < 0.):
                alpha_sh = 0.
            alpha_long = 1.4e-3*(2. - Qout)/(mu)**1.25/Qout**0.5
            if (alpha_long < 0.):
                alpha_long = 0.
            alpha = alpha0 + (alpha_sh**2 + alpha_long**2)**0.5
        else:
            alpha = alpha0
        '''
        alpha = alpha0
        
        #Baruteau+11 migration
        #t_migr = 5.6/(2.8)*2.*np.pi/Om_K*(5/3)*Q*HtoR**3*(M_star/M_P)*(0.1/HtoR)**2 #Type I
        t_migr = 0.8/Om_K * M_star**2/(M_P * Md) * HtoR**2 #type I
        Crida_p = 0.75*H/RH + 50.*alpha*HtoR**2 * M_star/M_P
        if (Crida_p < 1.):
            #Malik+15 gap migration
            t_cross = 2.5*RH/A*t_migr
            t_gap = 1./Om_K * (M_star/M_P)**2 * HtoR**5
            
            if (t_cross > eta * t_gap):
                typeI = False
                t_migr = 1./(alpha*Om_K*HtoR**2) * (1. + M_P/Md) #type II
               
        
    elif Model == 'FR13':
        alpha = alpha0
        if (M_P < 2.* M_star*HtoR**3):
            t_migr = 1/Om_K * M_star/(M_P) *HtoR #type I
        else:
            t_migr = 1./(alpha*Om_K*HtoR**2) * (1. + M_P/Md) #type II
            typeI = False

    
    elif Model == 'K+18':
        alpha = alpha0
        Kt = 20
        K = (M_P/M_star)**2*(HtoR)**(-5)/alpha
        t_1 = np.pi/2 * 1/Om_K * M_star**2/(M_P * Md) * HtoR**2
        t_migr = -t_1 *(1+ 0.004*K) / (-2.4 + 0.77*np.exp(-K/Kt)) 

    elif Model == 'Quenched migration':
        alpha = alpha0 
        Crida_p = 0.75*H/RH + 50.*alpha*HtoR**2 * M_star/M_P
        if Crida_p > 1:
            t_migr = 0.8/Om_K * M_star**2/(M_P * Md) * HtoR**2 #type I
        else:
            typeI = False
            t_migr = 1./(alpha*Om_K*HtoR**2) * (1. + M_P/Md) #type II

    elif Model == 'Super migration':
        alpha = alpha0 + alpha_sg * Q_crit**2/(Q_crit**2 + Q**2)
        Crida_p = 0.75*H/RH + 50.*alpha*HtoR**2 * M_star/M_P
        if Crida_p > 1:
            t_migr = 0.8/Om_K * M_star**2/(M_P * Md) * HtoR**2 #type I
        else:
            typeI = False
            t_migr = 1./(alpha*Om_K*HtoR**2) * (1. + M_P/Md) #type II

        
    #==== Set min and max migration times ====#
    t_migr_min = 2.*np.pi/Om_K
    #print 't_migr, A', t_migr/c.sec_per_year, A/c.AU
    
    if t_migr < t_migr_min:
        t_migr = t_migr_min
  
    dt = 0.01 * t_migr
    
    if dt > 0.01*t_disp:
        dt = 0.01*t_disp

    dMstar_dt = M_disc/t_disp
            
    if args0['Gasacc'] == True and M_disc != 0:
        #==== Proto gas accretion ====#
        '''
        t_KH = Gas_c * (M_P/c.MJ)**Gas_pow *c.sec_per_year
        Mdot_disc = 3*np.pi*Sig*alpha*cs*H

        Z_peb = max(1-np.log10(M_P/c.MJ),0)
        t_KH *= 10**Z_peb**0.7
        dMP_dt = M_P / t_KH
        if typeI == True:
            pass
        elif typeI == False:
            #if M_P > mass_ring:
            dMP_dt = min(Mdot_disc,M_P/t_KH)
        if M_P > M_disc:
            dMP_dt = min(dMstar_dt,M_P/t_KH)
            dMstar_dt = dMstar_dt - dMP_dt
        '''
        
        #peb_fac = 10 ** max(1.3-np.log10(M_P/c.MJ),0)
        #dMP_dt = 2 * RH**2*Om_K*Sig /peb_fac**3
        M_crit = args0['M_crit']#10#20
        M_fac = 1 / (M_crit/(M_P/c.MJ) + 1)
        Z_fac =  10**-args0['Z']

        dMP_dt = 2 * RH**2*Om_K*Sig * M_fac**3 * Z_fac

        
        #Gap_fac = 1.668*(M_P/c.MJ)**(1/3) * np.exp(-M_P/(1.5*c.MJ)) +0.04 #Veras Armitage 2004
        #Gap_fac = 0.2
        Gap_fac = min((Md/M_disc),1)
        if typeI == False:
            dMP_dt *= Gap_fac

        #BD companions starve star    
        if M_P > 0.05*M_star:
            #dMP_dt = min(dMstar_dt,dMP_dt)
            dMstar_dt = max(dMstar_dt-dMP_dt,0)
        if M_P > M_star:
            dMP_dt /= 2
            dMstar_dt = dMP_dt   
        
        if dt > 0.01*M_P / dMP_dt:
            dt = 0.01 * M_P / dMP_dt
        
        M_P    += dMP_dt*dt
        #M_disc = max(0,M_disc-dMP_dt*dt)
        M_disc = M_disc-dMP_dt*dt
            
    if t_infall != 0 and M_disc != 0:    
        dMinfall_dt = M_infall/t_infall
        M_infall -= dMinfall_dt*dt
        M_disc   += dMinfall_dt*dt

        
        
        #print 'MP', M_P/c.MJ
      
    #Update star and disc for dispersal time
    M_disc -= dMstar_dt * dt
    M_star += dMstar_dt * dt
    if M_disc < 0:
        M_disc = 0
        
    A = max(A * np.exp(-dt/t_migr),args0['Rin'])
    if np.isnan(A) == True:
        print 'dt',dt
        print 'tmigr',t_migr
        print M_P
    t = t + dt

    return t,A,M_P,RH,H,T,alpha,Sig,Om_K,typeI,M_disc,M_star,M_infall,Q,t_migr




def solve_migration(args0):
    '''Solve migration tracks'''
    
    #==== Initialise storage lists ====#
    Model  = args0['Model']
    t      = args0['t0'] #sec
    A      = args0['A0']
    M_P    = args0['M_P0']
    M_star = args0['M_star0']
    M_disc = args0['M_disc0']
    M_infall = args0['M_infall0']
    R_P    = 0
    RH     = A * (M_P/(3*M_star))**(1/3)
    alpha0 = args0['alpha0']
    M_core = 0
    E_fb = 0
    U = 0
    ts,As,M_Ps,R_Ps,RHs,Phases,alphas,M_discs = [t],[A],[M_P],[R_P],[RH],[1],[alpha0],[M_disc]
    M_cores,E_fbs,Us = [M_core],[E_fb],[U]
    tau = t
    RP_mode = args0['RP_mode']
    Z = args0['Z']
    Z_interp = args0['Z_interp']
    phase = 2 #Default 
    
    #==== Load Allona dataset ====#
    if RP_mode == 'Allona' or RP_mode == 'thermal_bath':
        #file1  = 'poutsum_'+str(int(args0['M_P0']/c.MJ))+'Mj_clmpev_isol_tmp'
        phase = 1
        MP_str = str(int(args0['M_P0']/c.MJ))
        if args0['M_P0']/c.MJ == 0.5:
            MP_str = '05'
        if Z_interp == False:
            file1  = 'poutsum_'+MP_str+'Mj_Z'+str(Z).replace('.','')
            file0  = 'poutsum_'+MP_str+'Mj_Z0'.replace('.','')
        else:
            file1  = 'poutsum_'+MP_str+'Mj_Z0'.replace('.','')


        try:
            #print file1
            eTime,eTc,eRho_c,eR_P,eT_eff,eL = load_allona_data(file1)
            if Z_interp == True:
                eTime *= (10**Z)**0.7   #Hi opac cool slower
                eL    *= (10**-Z)**0.7  #Lo opac are more luminous
            else:
                #Cap Max fragment radius at Z0 value.
                tZ0,TZ0,rZ0,R_PZ0,TeffZ0,LZ0 = load_allona_data(file1)
                Max_RP = R_PZ0[0]
                eR_P = np.clip(eR_P,0,Max_RP)
            R_Ps[0] = eR_P[0]
        except:
            RP_mode = 'None'
            print 'Clump evolution data not available for this planet mass'
            
    t_20AU = t_max
    
    #================ Run main migration loop ================#
    while (t < t_max) and (A > args0['Rin']):
        t,A,M_P,RH,H,T_mid,alpha,Sig,Om_K,typeI,M_disc,M_star,M_infall,Q,t_migr = migrate(t,A,M_P,M_star,M_disc,M_infall,args0,Model)
        if len(ts) == 1:
            Q0 = Q
            #==== Purge smol discs ====#
            if Q0 > Q_thresh:# or A < 50*c.AU:
                M_P = 0
                A = -500*c.AU
                
        dt   = t-ts[-1]
        if (A < 20*c.AU) and (t_20AU == t_max):
            t_20AU = t        

        #==== Check if phase ii collapse has occured ====#
        if RP_mode == 'Allona' or RP_mode == 'thermal_bath':
            if phase == 1:        
                ind = np.argmin((eTime-t)**2)
                R_P = eR_P[ind]
                if RP_mode == 'thermal_bath':
                    tau  = tau + dt * (1 - (4*np.pi*R_P**2*c.sigma_SB*T_mid**4)/ (eL[ind]*c.Lsol))
                    ind2 = np.argmin((eTime-tau)**2)
                    R_P = eR_P[ind2]
                if tau > eTime[-1]:
                    phase = 2
                        
            #Model has collapsed to phase ii
            else:
                R_P = 2*c.RJ

        elif RP_mode == 'Allona_fit':
            t_KH = Gas_c * (M_P/c.MJ)**Gas_pow *c.sec_per_year
            t_KH *= 10**Z**0.7

            #Z_peb = max(1-np.log10(M_P/c.MJ),0)
            #t_KH *= 10**Z_peb**0.7

            if t < t_KH:
                R_P = c.AU
            else:
                R_P = 2*c.RJ
                phase = 2
            
        #==== No radius treatment ====#
        else:
            R_P = 1 #point source


        #==== Check if tidal disruption has occured ie Rp > RH_frac*RH====#
        if R_P > RH*RH_frac:
            #print 'Clump disrupted!'
            #if phase == 1:
            #    M_P,R_P = post_disrupt_mass(A,R_P,M_P,M_star,Sig,H,tau,t)
            #    pass    
            #t_disrupt = t
            t = t_max
            A = -100*c.AU
             
            

        #==== Check if pebble accretion driven core feedback destroys fragment ====#
        if M_core < M_disc*args0['DtoG']/10: #10% of all dust
            Mdot_p = Sig*args0['DtoG'] *RH**2*Om_K /2
            M_core += Mdot_p*dt
            R_core = (3*M_core/4/np.pi/args0['rho_core'])**(1/3)
            U      = 6/5*c.G*M_P**2 /R_P #gamma 7/5 polytrope
            E_fb   += M_core*Mdot_p *c.G/ R_core *dt

            if E_fb > U:
                t = t_max
                A = -200*c.AU
                M_core = - M_core
            
        #==== Update save lists ====#
        ts.append(t)
        As.append(A)
        M_Ps.append(M_P)
        R_Ps.append(R_P)
        RHs.append(RH)
        Phases.append(phase)
        alphas.append(alpha)
        M_cores.append(M_core)
        E_fbs.append(E_fb)
        Us.append(U)
        M_discs.append(M_disc)

    #==== Export data for complete evolution track ====#
    ts      = np.asarray(ts)/c.sec_per_year #Years
    As      = np.asarray(As)/c.AU  #AU
    M_Ps    = np.asarray(M_Ps)/c.MJ
    R_Ps    = np.asarray(R_Ps)/c.AU #AU
    RHs     = np.asarray(RHs)/c.AU  #AU
    Phases  = np.asarray(Phases)
    alphas  = np.asarray(alphas)
    M_cores = np.asarray(M_cores)/c.ME #ME
    E_fbs   = np.asarray(E_fbs)
    Us      = np.asarray(Us)
    M_discs = np.asarray(M_discs)/c.MJ
    
    return ts,As,M_Ps,R_Ps,RHs,Phases,alphas,M_cores,E_fbs,Us,t_20AU,typeI,Q0,M_discs





def run_models(params,var='Model',mode='GI_pop',timescales=False):
    '''Run three comparison models'''
    #==== Initial system parameters ====#
    lss = ['-','--',':']
    args0 = {}
    #mode = 'GI_pop'#'GI_gasacc'
    
    if mode == 'GI_pop':
        q0 = 0.2
        args0['M_star0']  = 1.*c.Msol #g
        args0['M_disc0']  = q0*args0['M_star0']    #g
        args0['Rin']      = 0.01*c.AU    #cm
        args0['Rout']     = 150*c.AU#150*c.AU     #cm
        args0['T100']     = 20           #K
        args0['A0']       = 100*c.AU #100*c.AU     #cm
        args0['M_P0']     = 5*c.MJ  #10*c.ME #g
        args0['RP_mode']  = 'thermal_bath'#'None'#
        args0['alpha0']   = 0.005
        args0['t_disp']   = 3e5 *c.sec_per_year
        args0['t0']       = 1 #Years
        args0['Z']        = 0
        args0['DtoG']     = 0
        args0['rho_core'] = 5 #g/cm^3
        args0['Model']    = 'Super migration'
        args0['SG_corr']  = False
        args0['Z_interp'] = False
        args0['Gasacc']   = False
        args0['M_crit']   = 10 #MJ
        args0['alpha_sg'] = 0.12  
        M_P0s = np.array([1,3,5,7])*c.MJ

    elif mode == 'GI_gasacc':
        q0 = 1.9#0.4
        Infall_frac = 0.5
        args0['M_star0']   = 0.7*c.Msol #g
        args0['M_disc0']   = q0 * (1-Infall_frac) *args0['M_star0']    #g
        args0['M_infall0'] = q0 * Infall_frac *args0['M_star0'] #g
        args0['Rin']       = 0.05*c.AU    #cm
        args0['Rout']      = 150*c.AU#150*c.AU     #cm
        args0['T100']      = 20           #K
        args0['A0']        = 150*c.AU #100*c.AU     #cm
        args0['M_P0']      = 5*c.MJ  #10*c.ME #g
        args0['RP_mode']   = 'None'#'thermal_bath'#'None'#
        args0['alpha0']    = 0.005
        args0['t_disp']    = 1e6 *c.sec_per_year #sec
        args0['t0']        = 1 #Years
        args0['Z']         = 0
        args0['DtoG']      = 0
        args0['rho_core']  = 5 #g/cm^3
        args0['Model']     = 'Super migration'
        args0['SG_corr']   = False
        args0['Z_interp']  = False
        args0['Gasacc']    = True
        args0['M_crit']    = 12 #MJ
        args0['alpha_sg']  = 0.06#12
        M_P0s = np.array([1,2,3,4,5])*c.MJ #g

    
    Models = ['Super migration']#['Quenched migration']#['NF15','FR13','K+18']
    RP_modes = ['Allona','thermal_bath']
    SG_corr = [True,False]
    Zs = [-0.3,0,0.3]
    #M_P0s = np.array([0.5,1,2,3,4,5,6,7,8,9])*c.MJ
    #M_P0s = np.array([1,3,5,7])*c.ME
    #M_P0s = np.array([3])*c.MJ
    t_disps = np.array([1e5,1e6]) * c.sec_per_year
    
    #==== Plotting code ====#
    fig_h = 2.5*len(params)
    a_min = 0.4
    if len(params) == 1:
        fig_h *= 1.5
        a_min = 0.1
    
    fig10,axes = b.make_plots(11,len(params),figsize=(6,fig_h))

    for i in range(len(axes)-1):
        plt.setp(axes[i].get_xticklabels(), visible=False)

    if timescales == True:
        #==== Timescale plot ====#
        fig15 = plt.figure(15,facecolor='w')
        ax4 = fig15.add_axes([0.15,0.15,0.8,0.8])
        tk_MPs = [1,3,5,7]
        t_KHs_Z0  = np.array([470685,91343,32206,15946])
        t_KHs_Z03 = np.array([751039,168415,67682,31471])
        t_KHs_Z_03 = np.array([301244,49774,18656,8588])
        ax4.plot(tk_MPs,t_KHs_Z_03,color=plot_cols[0],marker='o',label=r'$t_{KH}$, Z = -0.3')
        ax4.plot(tk_MPs,t_KHs_Z0,color=plot_cols[1],marker='o',label=r'$t_{KH}$, Z = 0')
        ax4.plot(tk_MPs,t_KHs_Z03,color=plot_cols[2],marker='o',label=r'$t_{KH}$, Z = 0.3')
        ax4.semilogy()
        ax4.set_xlim(M_P0s[0]*0.8/c.MJ,M_P0s[-1]*1.1/c.MJ)
        ax4.set_ylabel('Time [Years]')
        ax4.set_xlabel(r'Mass [$M_J$]')
        plt.legend(frameon=False,loc=2)
        
    #Set differentiating variable
    if var == 'Model':
        Mvar = Models
    elif var == 'RP_mode':
        Mvar = RP_modes
    elif var == 'SG_corr':
        Mvar = SG_corr
    elif var == 'Z':
        Mvar = Zs
    elif var == 'M_P0':
        Mvar = M_P0s

    label_var = Mvar
    if var == 'M_P0':
        label_var = Mvar/c.MJ
        label_var = label_var.astype(str)
        print 'Label', label_var
        for i in range(len(label_var)):
            label_var[i] = label_var[i] + r' $M_J$'

    var2 = ' '#'Model'#'t_disp'
    Mvar2 = [0]#Models#[0]#t_disps
    #var2 = 'Model'
    #Mvar2 = Models #[]
        
    t_20AUs = np.zeros((len(Mvar),len(Mvar2)))

    if len(Mvar2) > 1:
        for k in range(len(Mvar2)):
                axes[0].plot([],[],color=plot_cols[0],ls=lss[k],label=str(Mvar2[k]))
        
    for i in range(len(Mvar)):
        for k in range(len(Mvar2)):
            args0[var] = Mvar[i]
            args0[var2] = Mvar2[k]

            ts,As,M_Ps,R_Ps,RHs,Phases,alphas,M_cores,E_fbs,Us,t_20AU,typeI,Q0,M_discs = solve_migration(args0)
            print 'N steps', len(ts)
            inds1 = np.where(Phases==1)
            inds2 = np.where(Phases==2)
            if args0['RP_mode'] == 'None':
                inds1 = np.arange(len(Phases))
                inds2 = []
            if k>0:
                label_var[i] = ''
                
            for j in range(len(axes)):
                if params[j] == 'A':
                    axes[j].plot(ts[inds1],As[inds1],label=str(label_var[i]),color=plot_cols[i],ls=lss[k])
                    axes[j].plot(ts[inds2],As[inds2],ls=':',color=plot_cols[i])
                    axes[j].set_ylabel('R [AU]')
                    if i == len(Mvar)-1:
                        axes[j].semilogy()
                        #axes[j].set_yscale('symlog')
                        if args0['RP_mode'] != 'None':
                            axes[j].plot([],[],ls=':',color=plot_cols[i],label='Phase 2')
                    axes[j].set_ylim(a_min,400)
                    axes[j].set_xlim(101,t_max*1.1/c.sec_per_year)
                    axes[j].semilogx()
                    axes[j].legend(frameon=False)

                elif params[j] == 'R_P':
                    axes[j].plot(ts[:-1],R_Ps[:-1],color=plot_cols[i],label=r'$R_P$')
                    axes[j].plot(ts[:-1],RH_frac*RHs[:-1],ls='--',color=plot_cols[i],label=r'$R_{disr}$')
                    if i == 0:
                        axes[j].legend(frameon=False,loc=1)
                    axes[j].set_ylabel(r'R$_P$ [AU]')
                    axes[j].set_ylim(-0.2,9)

                elif params[j] == 'M_P':
                    axes[j].plot(ts,M_Ps,color=plot_cols[i])
                    axes[j].set_ylabel('M [MJ]')
                    axes[j].semilogy()

                elif params[j] == 'alpha':
                    axes[j].plot(ts[:-1],alphas[:-1],color=plot_cols[i])
                    axes[j].set_ylabel(r'$\alpha_P$')
                    if i == 0:
                        axes[j].semilogy()
                    axes[j].set_ylim(4e-3,0.09)

                elif params[j] == 'M_core':
                    axes[j].plot(ts,M_cores,color=plot_cols[i])
                    axes[j].set_ylabel(r'$M_{core}$ [ME]')

                elif params[j] == 'E':
                    axes[j].plot(ts,E_fbs,color=plot_cols[i])
                    axes[j].plot(ts,Us,color=plot_cols[i],ls='--')
                    axes[j].set_ylabel(r'Energy [erg]')
                    if i == 0:
                        axes[j].semilogy()

                elif params[j] == 'M_disc':
                    axes[j].plot(ts,M_discs,color=plot_cols[i])
                    axes[j].set_ylabel(r'$M_{disc}$ [M$_{J}$]')


        #==== Extra values for timescale loop ====#
        if timescales == True:
            for k in range(len(Mvar2)):
                args0[var2] = Mvar2[k]
                ts,As,M_Ps,R_Ps,RHs,Phases,alphas,M_cores,E_fbs,Us,t_20AU,typeI,Q0,M_discs = solve_migration(args0)
                t_20AUs[i,k] = t_20AU
       

    axes[-1].set_xlabel('Time [Years]')
    
    text = ''
    if var != 'M_P0':
        text += r'M$_P$ = ' + str(args0['M_P0']/c.MJ)+r' M$_J$  '
    #text += r'$t_{disp}$ = ' + str(args0['t_disp']) + ' Years'
    if var != 'Model':
        text += args0['Model']
    text += r', M$_{tot}$ = ' + str(args0['M_star0']*(1+q0)/c.Msol) + r' M$_{\odot}$'
    axes[0].text(0,1.03,text,transform=axes[0].transAxes)

    if timescales == True:
        for k in range(len(Mvar2)):
            print M_P0s
            print t_20AUs[:,k]
            ax4.plot(M_P0s/c.MJ,t_20AUs[:,k]/c.sec_per_year,color=plot_cols[k])
    return



def resample_L(L1,t1,t2):
    '''Project L1 into t2 space'''
    #Rescale to dimensionless time
    t1 /= t1[-1]
    t2 /= t2[-1]

    L2 = np.zeros(len(t2))
    for i in range(len(t2)):
        ind = np.argmin((t2[i]-t1)**2)
        L2[i] = L1[ind]

    return L2
    


def plot_allona_data():

    #MPs = [0.5,1,2,3,4,5,6,7,8,9,10,11,12]
    #MPs = [2,3,4,5,6,]#[6,7,8,9,10,11,12]
    #MPs = [2,4,6,8,10,12]
    MPs = [1,2,3]
    #MPs = [5,6,7,8,9,10,11,12]
    Zs  = [-0.3,0,0.3]
    #Zs = [0]
    lss = ['-','--',':']
        
    fig1 = plt.figure(11,facecolor='white',figsize=(6,7))
    ax1 = fig1.add_axes([0.2,0.74,0.75,0.22])
    ax2 = fig1.add_axes([0.2,0.52,0.75,0.22],sharex=ax1)
    ax3 = fig1.add_axes([0.2,0.3,0.75,0.22],sharex=ax1)
    ax4 = fig1.add_axes([0.2,0.08,0.75,0.22],sharex=ax1)
    axes = [ax1,ax2,ax3,ax4]

    fig2 = plt.figure(12,facecolor='w',figsize=(6,7))
    ax5 = fig2.add_axes([0.1,0.1,0.8,0.8])
    ax5.semilogx()
    ax5.set_xlabel('Collapse time [Years]')
    ax5.set_ylabel(r'Planet Mass [$M_J$]')

    fig3 = plt.figure(13,facecolor='w',figsize=(6,7))
    Laxes,L0s,time0s,R0s = [],[],[],[]
    for i in range(len(MPs)):
        dy = 0.85/len(MPs)
        if i == 0:
            Laxes.append(fig3.add_axes([0.15,0.95-(i+1)*dy,0.8,dy]))
        else:            
            Laxes.append(fig3.add_axes([0.15,0.95-(i+1)*dy,0.8,dy],sharex=Laxes[0],sharey=Laxes[0]))

    fig4 = plt.figure(14,figsize=(6,5))
    ax6 = fig4.add_axes([0.15,0.11,0.8,0.42])
    ax7 = fig4.add_axes([0.15,0.53,0.8,0.42])
    R0_store = np.zeros((len(MPs),len(Zs),2))
         
    #==== Loop over Allona's data ====#
    for i in range(len(MPs)):
        for j in range(len(Zs)):
            
            fileij = 'poutsum_'+str(MPs[i]).replace('.','')+'Mj_Z'+str(Zs[j]).replace('.','')
            r_label = str(MPs[i])+' $M_J$'
            MP      = MPs[i]
            Z       = str(Zs[j])
 
            Time,Tc,Rho_c,R_P,T_eff,L = load_allona_data(fileij)
            R0_store[i,j,0] = R_P[0]
            R0_store[i,j,1] = Time[-1]

            Time = Time/c.sec_per_year
            if j != 0:
                r_label=''
                
            ax1.plot(Time,Tc,color=plot_cols[i],ls=lss[j])
            ax1.plot(Time,T_eff,color=plot_cols[i],ls=lss[j])
            ax2.plot(Time,Rho_c,color=plot_cols[i],ls=lss[j])
            #ax2.plot(Time,T_eff,color=plot_cols[i],ls=lss[j])
            ax3.plot(Time,R_P/c.AU,color=plot_cols[i],ls=lss[j],label=r_label)
            ax4.plot(Time,L,color=plot_cols[i],ls=lss[j])
            if i == 0 and Zs != [0]:
                ax2.plot([],[],color=plot_cols[0],label='[M/H]: '+Z,ls=lss[j])

            #+ t_collapse plot +#
            ax5.scatter(Time[-1],MP,color=plot_cols[i])
            if Z == '0':
                fac = (10**0.3)**0.7
                ax5.scatter(Time[-1]*fac,MP,color=plot_cols[i],marker='x')
                ax5.scatter(Time[-1]/fac,MP,color=plot_cols[i],marker='x')
                ax5.scatter([],[],color=plot_cols[i],label=r_label)

            if i == 0 and j == 0:
                ax5.scatter([],[],color=plot_cols[0],marker='x',label=r'$t_{collapse} \propto Z^{0.7}$')

            #+== Luminosity plot ===#
            if j == 0:
                L0s.append(L)
                time0s.append(Time)
                R0s.append(R_P)
            L   = resample_L(L,Time,time0s[i])
            R_P = resample_L(R_P,Time,time0s[i])
            L_R2 = L/L0s[i] * (R0s[i]/R_P)**2
            #L_R2 = L/L0s[i]

            if Z == '0':
                fac = (10**0.3)**0.7
                Laxes[i].plot(time0s[i]/time0s[i][-1],L_R2,color=plot_cols[i],label='MP: '+str(MP)+r' $M_J$')
                Laxes[i].plot(time0s[i]/time0s[i][-1],L_R2*fac,color=plot_cols[i],ls='--')
                Laxes[i].plot(time0s[i]/time0s[i][-1],L_R2/fac,color=plot_cols[i],ls='--')
                Laxes[i].legend(frameon=False,loc=2)
            else:
                Laxes[i].plot(time0s[i]/time0s[i][-1],L_R2,color=plot_cols[i])
            Laxes[i].semilogx()
        

    ax1.set_ylabel('Tc [K]')
    ax2.set_ylabel(r'$\rho_c$ [gcm$^{-3}$]')
    #ax2.set_ylabel('T_eff [K]')
    ax3.set_ylabel(r'R$_P$ [AU]')
    ax4.set_ylabel(r'L [L$_\odot$]')
    ax4.set_xlabel('Time [Years]')
    ax1.semilogx()
    #ax1.semilogy()
    ax2.semilogy()
    #ax3.semilogy()
    ax4.semilogy()

    ax3.legend(frameon=False,loc=1)
    ax2.legend(frameon=False)
    ax5.legend(frameon=False)

    #Luminosity
    Laxes[0].set_xlim(7e-4,1.35)
    Laxes[0].set_ylim(0,2.4)
    Laxes[len(MPs)-1].set_xlabel('Dimensionless time')
    Laxes[int(len(MPs)/2)].set_ylabel('L/L0')

    for i in range(len(Zs)):
        print 'a',R0_store[:,i,1]/c.sec_per_year
        ax6.plot(MPs,R0_store[:,i,0]/c.AU,color=plot_cols[i],label='[M/H]=' + str(Zs[i]))
        ax7.plot(MPs,R0_store[:,i,1]/c.sec_per_year,color=plot_cols[i],label='[M/H]=' + str(Zs[i]))
        
    #ax6.plot(MPs,(np.array(MPs)*1e26)**0.5+0.2)

    ax6.set_ylabel('Initial radius [AU]')
    ax6.set_xlabel(r'Planet mass [$M_J$]')
    ax6.legend(frameon=False)
    ax6.set_xticks(MPs[1::2])
    ax7.set_ylabel(r'Collapse time [Years]')
    #ax7.set_xlabel(r'Planet mass [$M_J$]')
    plt.setp(ax7.get_xticklabels(), visible=False)

    #ax7.legend(frameon=False)
    ax7.set_xticks(MPs[1::2])
    ax7.semilogy()

    return





def gas_accretion_ests():
    '''Estimate functional forms for gas accretion rates.'''
    MPs = [0.5,1,2,3,4,5,6,7,8,9,10,11,12]
    t_KHs_Z_03 = np.array([845251,301366,98822,49770,26258,16668,11460,8198,6041,4696,3757,3105,2931])
    t_KHs_Z0  = np.array([1331597,473011,178805,94424,53456,32363,22940,15969,12466,9209,7353,6131,4947])
    t_KHs_Z03 = np.array([2100413,753594,282595,167634,100977,63350,45525,31655,21534,15635,12226,11205,8579])
    fMPs = np.logspace(0,3,100)
    
    #Fit power laws to gas accretion rates..
    #Ask Allona to run some high mass.. 20,30,40,50MJ etc
    
    popt_Z0, pcov_Z0 = curve_fit(b.func_powerlaw, MPs,t_KHs_Z03,p0=np.asarray([-1.8,2e6,0]))

    popt_hack = np.asarray([-1.44,6e5,0])#-1.2e4])
    print popt_Z0
    print popt_hack 
    t_KH_fit = b.func_powerlaw(MPs,*popt_Z0)
    t_hack = b.func_powerlaw(fMPs,*popt_hack)
    plt.plot(MPs,t_KH_fit,'--',label='Fit')
    plt.plot(fMPs,t_hack,'--',label='hack')
    
    plt.plot(MPs,t_KHs_Z0,label='Z0 data',color=plot_cols[0])
    plt.plot(MPs,t_KHs_Z03,label='Z0 data',color=plot_cols[1])
    plt.plot(MPs,t_KHs_Z_03,label='Z0 data',color=plot_cols[2])

    plt.legend(frameon=False)
    plt.semilogy()
    plt.semilogx()
    
    return


def troup_comps(Mstar_u=1.4,Mstar_d=0.7):
    '''Load companion data from Troup 16 RV companions'''
    fname = 'troup16_companions.csv'
    data = np.genfromtxt(obs_fol+fname,skip_header=1,delimiter=',')

    A      = data[:,6]
    M_P    = data[:,5] *c.Msol/c.MJ #MJ
    M_star = data[:,3]  #Msol
    Z_star = data[:,2]
    
    inds = np.where((M_star>=Mstar_d) & (M_star<Mstar_u))[0]
    
    return A[inds], M_P[inds]

def ZS09_comps(Mstar_u=1.4,Mstar_d=0.7):
    '''Load companion data from Zuckerman Song 09 IR BD companions'''
    fname = 'ZuckermanSong_BDcompanions.csv'
    data = np.genfromtxt(obs_fol+fname,skip_header=1,delimiter=',')

    A      = data[:,6] #AU
    M_P    = data[:,5] #MJ
    M_star = data[:,4]  #Msol
    
    inds = np.where((M_star>=Mstar_d) & (M_star<Mstar_u))[0]
    return A[inds], M_P[inds]

def Raghavan_comps():
    '''Ragahavan companions around 0.67-1.22 K3-F6 solar Mass stars'''
    fname = 'Raghavan_2010_q-P_dataextraction.csv'
    data = np.genfromtxt(obs_fol+fname,delimiter=',')

    A = (data[:,0] /365.25)**(2/3) #AU
    q = data[:,1]

    #Assume Msol = 1
    M_P = q * c.Msol/c.MJ #MJ

    return A, M_P
    

def cumming_powerlaw(MP0,MP1,A0=0.031,A1=5):
    '''Population fits based on Cumming 2008
    C calculated based on 10.5% of stars hosting a planet in mass range 0.3-10 and P 2-2000'''
    
    alpha = -0.31
    alpha_err = 0.2
    beta  = 0.26
    beta_err = 0.1
    C = 1.46e-3
    A0 = min(A0,0.031)
    
    P0 = A0**(3/2)*365.25
    P1 = A1**(3/2)*365.25
    percent = C * (MP1**alpha - MP0**alpha)/alpha * (P1**beta - P0**beta)/beta

    return percent


def exoplanets_org(split='MP',Mstar_u=1.4,Mstar_d=0.7,plot_data=False):
    '''Construct a histogram of exoplanets.org Juipters'''
    #fname = 'Exoplanet Primary.csv'
    fname = 'exoplanets.csv'
    data = np.genfromtxt(obs_fol+fname,skip_header=1,delimiter=',', invalid_raise = False)
    
    #M_P    = np.nan_to_num(data[:,19])
    #A      = np.nan_to_num(data[:,20])
    #M_star = np.nan_to_num(data[:,50])
    #R_star = np.nan_to_num(data[:,51])
    #Z_star = np.nan_to_num(data[:,52])

    A      = np.nan_to_num(data[:,0])
    M_P    = np.nan_to_num(data[:,139])
    M_star = np.nan_to_num(data[:,145])
    R_star = np.nan_to_num(data[:,197])
    Z_star = np.nan_to_num(data[:,75])

    scale = 1
    if split == 'MP':
        #Edges = [0.5,5,13]
        Edges = [5*c.ME/c.MJ,10*c.ME/c.MJ,20*c.ME/c.MJ]
        scale = c.MJ/c.ME
        Var   = M_P
    elif split == 'Mstar':
        #Edges = [0.7,1.1,1.5]#[0.7,0.9,1.1,1.3,1.5]
        Edges = [Mstar_d,Mstar_u]
        Var = M_star
    elif split == 'Rstar':
        Edges = [0,1,10]
        Var = R_star
    elif split == 'Zstar':
        Edges = [-0.4,-0.2,0,0.2,0.4]#[-1,0,1]
        Var = Z_star

    Ns = len(Var[np.where((Var>=Edges[0]) & (Var<Edges[-1]))])
    
    plt.figure(1)
    abins     = np.logspace(-2,2.3,50) #Radial bins
    As = []
    MPs = []
    
    for i in range(len(Edges)-1):
        inds = np.where((Var>=Edges[i]) & (Var<Edges[i+1]))[0]
        weights = np.ones(len(inds))*100/Ns
        label = 'Exoplanets.org: '+split+' {:.0f}'.format(Edges[i]*scale)+'-{:.0f}'.format(Edges[i+1]*scale)+r'$M_{\oplus}$,'+' {:.1f}'.format(len(inds)/Ns*100)+'%'

        plt.hist(A[inds],bins=abins,alpha=0.2,label=label,
                 weights=weights,color=plot_cols[2*i+1])
        As.append(A[inds])
        MPs.append(M_P[inds])
        
        
    plt.semilogx()
    plt.legend(frameon=False)
    inds2 = np.where((Var>=Edges[0]) & (Var<Edges[1]))[0]
    
        
    if plot_data == True:
        fig21 = plt.figure(2)
        axe  = fig21.add_axes([0.15,0.1,0.71,0.56])
        axeb = fig21.add_axes([0.15,0.81,0.71,0.15],sharex=axe)
        axeb2 = fig21.add_axes([0.15,0.66,0.71,0.15],sharex=axe)

        axec = fig21.add_axes([0.86,0.1,0.1,0.56],sharey=axe)
        MP_cut   = 4
        MP_floor = 0.1
        abins2  = np.logspace(-2,1.1,30) #Radial bins
        MPbins = np.logspace(np.log10(MP_floor),1.3,30)

        for i in range(len(Edges)-1):
            plot_A  = As[i][MPs[i]>MP_floor]
            plot_MP = MPs[i][MPs[i]>MP_floor]


            plot_Ab   = plot_A[plot_MP>MP_cut]
            plot_Ab2  = plot_A[plot_MP<MP_cut]

            axe.scatter(plot_A,plot_MP,color=plot_cols[i],label=split+': '+str(Edges[i])+'-'+str(Edges[i+1]),s=10,alpha=0.5)
            axeb.hist(plot_Ab,bins=abins2,color=plot_cols[i],label=str(Edges[i]),alpha=0.5,histtype='step')
            axeb2.hist(plot_Ab2,bins=abins2,color=plot_cols[i],label=str(Edges[i]),alpha=0.5,histtype='step')
            axec.hist(plot_MP,bins=MPbins,color=plot_cols[i],label=str(Edges[i]),alpha=0.5,orientation='horizontal',histtype='step')
            axec.text(0.6,0.85+i*0.03,str(len(plot_Ab)),transform=axec.transAxes,color=plot_cols[i])
            axec.text(0.6,0.05+i*0.03,str(len(plot_Ab2)),transform=axec.transAxes,color=plot_cols[i])

        axe.axhline(MP_cut,color=plot_cols[len(Edges)])
        axec.axhline(MP_cut,color=plot_cols[len(Edges)])
        axeb.text(0.05,0.7,r'$M_P$ > '+str(MP_cut)+r'$M_J$',transform=axeb.transAxes)
        axeb2.text(0.05,0.7,r'$M_P$ < '+str(MP_cut)+r'$M_J$',transform=axeb2.transAxes)
        axe.legend(loc=4)
        axe.semilogy()
        axe.semilogx()
        axe.set_xlabel('a [AU]')
        axe.set_ylabel('Planet mass [MJ]')
    
    
    return A[inds2],M_P[inds2]


def N_KS_test(f1,f2,extra=True):
    '''Run a 2d 2 sample Kolmogorov-Smirnov test to check that the two a-MP samples are 'identical'.
    Uses scripts from https://stackoverflow.com/questions/29544277/python-2-d-kolmogorov-smirnov-with-2-samples '''

    data1 = np.load(f1)
    data2 = np.load(f2)
    As1 = np.ravel(data1[:,:,1,:])
    As2 = np.ravel(data2[:,:,1,:])
    MP1 = np.ravel(data1[:,:,4,:])
    MP2 = np.ravel(data1[:,:,4,:])

    print 'KS p, D values: ', KS.ks2d2s(As1,MP1,As2,MP2,extra=extra)
    

    
def histogram(params,N_combi=1000,init=False,d_hists=False,col=plot_cols[0],lw=1,ls='-',label=''):
    '''Construct and plot one population histogram'''

    #==== Generate population parameters from input commands ====#
    global t_GImax
    sweep_key = ' '
    
    for key in params:
        #print key
        #print params[str(key)]
        
        if params[str(key)][0] == 'lin':
            pk = params[str(key)]
            lin = (pk[2]-pk[1])*np.random.rand(N_combi) + pk[1]
            params[str(key)] = lin
            
        elif params[str(key)][0] == 'log':
            pk = params[str(key)]
            log = pk[1] * (pk[2]/pk[1])**np.random.rand(N_combi)
            params[str(key)] = log
        
        elif params[str(key)][0] == 'sweep':
            pk = params[str(key)]
            params['NPs'] = [len(pk)-1]
            t_GImax = 1 #Launch all planets at same time
            sweep = np.tile(pk[1:],(N_combi,1))
            params[str(key)] = sweep
            sweep_key = str(key)
            
        else:
            params[str(key)] = np.asarray(params[str(key)])
            inds = (np.random.rand(N_combi)*len(params[str(key)])).astype(int)
            params[str(key)] = params[str(key)][inds]
    
 
    Nruns = np.sum(params['NPs'])
    Ais = np.zeros(Nruns)        #Store for initial separations
    Afs = np.zeros(Nruns)        #Store for final separations
    Ads = -1*np.ones(Nruns)      #Store for destruction separations
    Mfs = np.zeros(Nruns)        #Store for final masses
    Survive = np.ones(Nruns)     #Store for survival info
    t_20AUs = np.zeros(Nruns) #Store for disruption time (phase i)
    typeIs = np.zeros(Nruns)     #Store for typeI true at disruption..
    M_corefs = np.zeros(Nruns)   #Store for final core masses
    Zs = np.zeros(Nruns)
    Q0s = np.zeros(Nruns)
    Frags = np.zeros(Nruns)
    M_disc0s = np.zeros(Nruns) 
    M_infall0s = np.zeros(Nruns) 
    M_star0s = np.zeros(Nruns)
    
    #==== Build init condition dictionary ====#
    args0 = {}
    args0['T100']    = 20           #K
    ind = 0
    if sweep_key == ' ':
        params[sweep_key] = np.zeros((N_combi,params['NPs'][0]))
    sweep_store = params[sweep_key]
    MP0s = np.ravel(params['M_P0']*c.MJ)

    for i in range(N_combi):
        if ind%1000 == 0:
            print 'Run: ',ind
            
        for j in range(params['NPs'][i]):
            t0s = np.arange(0,t_GImax,t_GImax/params['NPs'][i]) #sec
            params[sweep_key] = sweep_store[:,j]

            args0['Model']       = params['Model'][i]
            args0['M_star0']     = params['M_stars'][i]*c.Msol #g stellar mass
            args0['M_disc0']     = params['qs'][i]*args0['M_star0']*(1-params['Infall_frac'][i]) #g
            args0['M_infall0']   = params['qs'][i]*args0['M_star0']*params['Infall_frac'][i] #g infall mass
            MR_fac = ((params['qs'][i]*args0['M_star0']/c.Msol)/0.05)**params['MR_pow'][i]
            args0['Rin']         = params['Rin'][i]*c.AU     #cm
            args0['Rout']        = params['Rout'][i]*c.AU  *MR_fac          #cm
            args0['A0']          = params['A0'][i]*args0['Rout'] #cm
            args0['M_P0']        = params['M_P0'][i]*c.MJ                    #g
            args0['RP_mode']     = params['RP_mode'][i]
            args0['t_disp']      = params['t_disp'][i]*c.sec_per_year
            args0['alpha0']      = params['alpha0'][i]
            args0['t0']          = t0s[j]
            args0['DtoG']        = params['DtoG'][i]
            args0['rho_core']    = params['rho_core'][i]
            args0['Z_interp']    = params['Z_interp'][i]
            args0['Z']           = params['Z'][i]
            args0['Gasacc']      = params['Gasacc'][i]
            args0['M_crit']      = params['M_crit'][i]
            args0['alpha_sg']    = params['alpha_sg'][i]
            
            #==== Solve migration for a single parameter combination ====#
            ts,As,M_Ps,R_Ps,RHs,Phases,alphas,M_cores,E_fbs,Us,t_20AU,typeI,Q0,M_discs = solve_migration(args0)
            
            #==== Save separation and planets mass data ====#
            Ais[ind] = As[0]
            Afs[ind] = As[-1]
            if As[-1] == -100:
                Ads[ind] = As[-2]
            if As[-1] < args0['Rin']/c.AU:
                Survive[ind] = 0
            Mfs[ind] = M_Ps[-1]#/c.MJ
            t_20AUs[ind] = t_20AU
            typeIs[ind] = typeI
            M_corefs[ind] = M_cores[-1]
            Zs[ind] = args0['Z']
            Q0s[ind] = Q0
            M_disc0s[ind] = args0['M_disc0']
            M_infall0s[ind] = args0['M_infall0']
            M_star0s[ind] = args0['M_star0']
            ind += 1
            
    #======== Plotting code ========#
    Nbins = 40
    bins     = np.logspace(-2,2.3,Nbins+1) #Radial bins    
    
    
    #Survival weights
    Weights = (MP0s/np.max(MP0s))**params['MPf_pow'][0]
    Weighted_number = np.sum(Weights)
    Norm_weights = Weights*100/Weighted_number
    
        
    #==== Plot initial histogram ====#
    plt.figure(1)
    d_label = ''
    if init == True:
        plt.hist(Ais,bins=bins,facecolor=plot_cols[0],alpha=0.2,label='Initial distribution',weights=Norm_weights)
        d_label = 'Disruption location'

    #==== Plot output histogram ====#
    plt.figure(1)
    p_survive = np.sum(Survive*Weights)/Weighted_number*100
    label += ', P={:.2f}'.format(p_survive)+'%'
    print Afs
    plt.hist(Afs,bins=bins,label=label,histtype='step',lw=lw,ls=ls,color=col,alpha=0.8,weights=Norm_weights)

    #==== Plot destruction histogram ====#
    if d_hists == True:
        plt.hist(Ads,bins=bins,label=d_label,histtype='step',lw=lw,ls=':',color=col,alpha=0.8,weights=Norm_weights)
    '''
    plt.figure(5)
    M_cores_embed  = M_cores[M_cores>=0]
    M_cores_perish = -M_cores[M_cores<0]
    plt.hist(M_cores_embed,color=col,alpha=0.3)
    plt.hist(M_cores_perish,color=col,alpha=0.8)
    '''
    plt.figure(20)
    plt.hist(Q0s)
    return Ais,Afs,t_20AUs,typeIs,MP0s,Mfs,Zs,Weights,Ads,Q0s,M_disc0s,M_infall0s,M_star0s
            


def abundance():
    masses    = np.array([50,10,2,1.5,1,0.7,0.2])
    lifetime  = np.array([10,100,1000,3000,10000,50000,200000])
    abundance = np.array([1e-5,0.1,0.7,2,3.5,8,80])
    l_a = lifetime/abundance
    l_a = l_a / np.sum(l_a)*100
    plt.scatter(masses,l_a)
    plt.semilogx()
    plt.semilogy()

    return

def simple_Mdot():
    M0 = np.arange(10)
    tf = M0**(-1.8) * 5/9 * 5e5
    plt.figure(12)
    plt.plot(M0,tf)
    plt.semilogy()
    plt.semilogx()
    plt.xlabel('MP0 [MJ]')
    plt.ylabel('Runaway time [Years]')
    
    return


def disc_diagnostic(params):
    '''Plot disc parameters against time'''
    for key in params:
        if params[str(key)][0] == 'lin':
            a = params[str(key)]
            mid = a[1] + (a[2]-a[1])/2
            params[str(key)] = mid
        elif params[str(key)][0] == 'log':
            b = params[str(key)]
            mid = 10 ** ( (np.log10(b[2])-np.log10(b[1]))/2 + np.log10(b[1]))
            params[str(key)] = mid
        else:
            params[str(key)] = params[str(key)][0]
            
    args0 = {}
    args0['T100']    = 20           #K
    args0['t0']      = 0
    M_tot = params['M_stars']*c.Msol #g Total system mass
    args0['M_star0'] = (1-params['qs'])*M_tot           #g
    args0['M_disc0'] = params['qs']*M_tot               #g
    args0['Rin']     = params['Rin']*c.AU     #cm
    args0['Rout']    = params['Rout']*c.AU            #cm
    args0['t_disp']  = params['t_disp']
    args0['alpha0']  = params['alpha0']

    NR = 100
    Nt = 5
    Rbins = np.logspace(np.log10(args0['Rin']),np.log10(args0['Rout']),NR)
    pRbins = Rbins / c.AU
    #store = np.zeros((Nt,NR,2))
    
    t  = args0['t0'] 
    ts = np.linspace(0,t_max,Nt)
    print ts
    print args0['t_disp']
    
    for i in range(Nt):
        M_disc,M_star,T,cs,HtoR,H,Om_K,Sig,Q,Qout = calc_analytic_disc(ts[i],Rbins,args0)
        plt.figure(10)
        plt.plot(pRbins,Sig,color=plot_cols[i])
        plt.text(pRbins[-1]+5,Sig[-1],' {:.2f} Myr'.format(ts[i]/c.sec_per_year/1e6),color=plot_cols[i])
        
    plt.figure(10)
    plt.xlim(pRbins[0]/2,pRbins[-1]*5)
    plt.xlabel('R [AU]')
    plt.ylabel(r'$\Sigma$ [gcm$^{-2}$]')
    plt.semilogy()
    plt.semilogx()

    return
            

    
def hist_grid(disc_diag=False,d_hists=False,bar_pops=False,pos_spread=False,plot_timescales=False,survive_plot=False,
              a_MP_plot=False,N_mass_hist=False,M_hists=False,aM_scat=False,M_barplot=False,disc0_plots=False):
    '''Build an array of histograms for each experimental variable'''
    lws = [1,1.5,3,1,4]
    lss = ['-','--',':','-.','-']
    init = True
    N_combi = 500 #Number of random samples per histogram

    #==== Define parameter ranges ====#
     
    GI_params = {
        #[A] single value
        #[A,B,C] multivalues for experimental.
        #[min,max,'log'/'lin'] randomly distributed values
    
        'Model':    ['Super migration'],#,'Super migration'],#'Quenched migration'],#,
        #'M_P0':    ['sweep',0.5,1,2,3,4,5,6,7,8,9,10,11,12],# #MJ
        #'M_P0':     [0.5,1,2,3,4,5,6,7,8,9,10,11,12],
        'M_P0':    [1,2,3,4,5,6,7,8,9,10,11,12],
        #'M_P0':    [2,3,4,5,6,7,8,9,10,11,12],
        #'M_P0':     [1,3,5,7],#,5,7],#,5,7], #MJ
        #'M_P0':     ['lin',1,7],#MJ
        #'M_P0':     ['log',1,10],#MJ

        'RP_mode':  ['thermal_bath'],#,'None'],
        'qs':       ['lin',0.1,0.3],#['lin',0.15,0.6],##  #disc/stellar fraction
        'M_stars':  [1],#[0.7,1,1.5],#['lin',0.7,1.5],  #Msol
        'Rin':      [0.01],#[0.01],#['log',0.01,0.05],  #AU
        'Rout':     ['lin',80,300],#['lin',80,250],#['lin',80,150],# ['lin',100,200],#  #AU
        'A0':       ['lin',0.5,1.0],# ['lin',0.5,0.8],  #initial orbit as fraction of Rout
        't_disp':   ['log',3e5,3e6],   #Years
        'Z':        [0],#['lin',-0.5,0.5],#[],#,0.3],
        'Z_interp': [False],
        'alpha0':   [0.005],#[1e-4,1e-2],['log',1e-4,1e-2],
        'NPs':      [1],#,3,5], #Number of planets per system. Only works as experimental atm
        'DtoG':     [0], #Dust to gas
        'rho_core': [5], #g/cm^3
        'MPf_pow':  [-1.3],
        'Gasacc':   [False],
        'M_crit':   [10], #MJ
        'alpha_sg': [0.12],
        'Infall_frac': [0],
        'MR_pow':   [0],
        ' ':        [0],
    }


    Gassacc_params = {
        #[A] single value
        #[A,B,C] multivalues for experimental.
        #[min,max,'log'/'lin'] randomly distributed values
        'Model':    ['Super migration'],
        #'M_P0':    ['sweep',0.5,1,2,3,4,5,6,7,8,9,10,11,12],# #MJ
        #'M_P0':    [0.5,1,2,3,4,5,6,7,8,9,10,11,12],
        #'M_P0':    [1,2,3,4,5,6,7,8,9,10,11,12],
        #'M_P0':    [2,3,4,5,6,7,8,9,10,11,12],
        #'M_P0':     [1,3,5,7],#,5,7],#,5,7], #MJ
        #'M_P0':     ['lin',1,7],#MJ
        'M_P0':     ['log',1,20],#MJ
        'RP_mode':  ['None'],
        'M_stars':  [0.7],#[0.7,1,1.5],#['lin',0.7,1.5],  #Msol
        'qs':       ['log',0.1,2],#['lin',0.15,0.6],##  #disc/stellar fraction
        #'qs':       [0.3], #disc/stellar fraction
        'Rin':      ['lin',0.02,0.08],#[0.01],#['log',0.01,0.05],  #AU
        #'Rout':     ['lin',50,300],# ['lin',100,200],#  #AU
        'Rout':     ['log',10,150],# ['lin',100,200],#  #AU
        'A0':       [1],#'lin',0.5,1],   #initial orbit as fraction of Rout
        't_disp':   ['log',3e5,3e6],   #Years
        'Z':        [0],#['lin',-0.5,0.5],#[],#,0.3],
        'Z_interp': [False],
        'alpha0':   [0.005],#[1e-4,1e-2],['log',1e-4,1e-2],
        'NPs':      [1],#,3,5], #Number of planets per system. Only works as experimental atm
        'DtoG':     [0], #Dust to gas
        'rho_core': [5], #g/cm^3
        'MPf_pow':  [0],
        'Gasacc':   [True],
        'M_crit':   [12], #MJ
        'alpha_sg': [0.06],#,0.12],
        'Infall_frac': [0.5],
        'MR_pow':   [0.37], #0
        ' ':        [0],
    }
    
    Core_params = {
        #[A] single value
        #[A,B,C] multivalues for experimental.
        #[min,max,'log'/'lin'] randomly distributed values

        'Model':   ['NF15'],#,'Mueller+18'],#'FR13'],
        'M_P0':      np.array([5,10,15])*c.ME/c.MJ, #ME
        'RP_mode': ['None'],#,'Allona','thermal_bath'],#['None'],#,
        'qs':       ['lin',0.05,0.2],  #disc/stellar fraction
        'M_stars':  ['lin',0.7,1.5], #['lin',0.9,1.1], #[0.7,1.1,1.5],#  #Msol
        'Rin':    [0.01],#['log',0.01,0.05],  #AU
        'Rout':   ['lin',70,150],    #AU
        'A0':      ['lin',0.15,0.3],   #initial orbit as fraction of Rout
        't_disp':  ['log',3e5,3e6],   #Years
        'Z':        [0],
        'Z_interp': [False],
        #'alpha0':  np.logspace(-4,-2,5),#['log',2e-3,1e-2]
        'alpha0':  ['log',1e-4,1e-3],
        'DtoG':     [0], #Dust to gas
        'rho_core': [5], #g/cm^3
        'NPs':      [1],#,3,5], #Number of planets per system. Only works as experimental atm
        'DtoG':     [0], #Dust to gas
        'rho_core': [5], #g/cm^3
        'MPf_pow':  [0],
        'Gasacc':   [False],
        'Infall_frac': [0],
        'MR_pow':   [0],
        ' ':        [0]
    }

    params = Gassacc_params #
    #params = GI_params
    Mass_scale = 1#c.MJ/c.ME

    labels = {'Model':    'Model',
              'M_P0':     r'$M_P$',
              'RP_mode':  r'$R_P$',
              'qs':       r'$M_{disc}/M_*$',
              'M_stars':  r'$M_*$',
              'Rin':      r'$R_{in}$',
              'Rout':     r'$R_{out}$',
              'A0':       r'$A_0$',
              't_disp':   r'$t_{disp}$',
              'Z':        r'$[M/H]$',
              'Z_interp': 'Z interp',
              'alpha0':   r'$\alpha_0$',
              'DtoG':     'Dust to gas',
              'rho_core': r'$\rho_{core}$',
              'NPs':      r'$N_Ps$',
              'MPf_pow':  r'$N(M_P) \propto M_P$',
              ' ':        '',
    }
    
    
    #==== Set experimental variables ====#
    var1 = 'M_stars'#'alpha_sg'#'alpha0'#'Z'#'M_P0'
    var2 = ' '#'MPf_pow'#' '#'Z'#' '#M_stars'
    
    Redges = [-150,0,5,20,300]#100]
    xlabels = ['Disrupted']
    for i in range(len(Redges)-1-len(xlabels)):
        xlabels.append(str(Redges[len(xlabels)])+'-'+str(Redges[len(xlabels)+1]))
    Medges = [0,1,13,75,1e4]
    Mxlabels = ['Disrupted']
    for i in range(len(Medges)-2-len(Mxlabels)):
        Mxlabels.append(str(Medges[len(Mxlabels)])+'-'+str(Medges[len(Mxlabels)+1]))
    Mxlabels.append(str(Medges[-2])+'+')
    N_params = len(params[var1])
    bar_width = 1/(N_params+1)
        
    Vigan_2_14 = 1.15
    V_errm, V_errp = 0.3,2.5
    Cumming_1_12 = 7.28
    Cumming_2_12 = 4.66
    Cumming_4_12 = 2.55 #with NP propto MP MP^(-1.3)
    error_4_15 = 0.9 #with NP propto MP MP^(-1.3)
    MB0,MBu,MBd = 1.65,0.45,9.05
    
    #Allona data
    Q_0s    = [0.028,0.007,0.152]
    Q_13s   = [0.064,0.0016,0.34]
    S_0s    = [0.129,0.0315,0.681]
    S_13s   = [0.413,0.083,0.967]
    FP_facs = [Q_0s,S_0s,Q_13s,S_13s]
    
    #==== Establish Figures ====#

    if disc_diag == True:
        diag_params = params.copy()
        disc_diagnostic(diag_params)
    
    plt.figure(1,figsize=(10,7))
    if bar_pops == True:
        fig2 = plt.figure(2,)
        ax2 = fig2.add_axes([0.15,0.1,0.8,0.8])
        index = np.arange(len(Redges)-1)
        fig6 = plt.figure(6)
        ax6 = fig6.add_axes([0.15,0.15,0.8,0.8])
        
    if plot_timescales == True:
        fig4 = plt.figure(4)
        ax4 = fig4.add_axes([0.12,0.12,0.82,0.8])
        
    if survive_plot == True:
        fig5 = plt.figure(5)
        ax5 = fig5.add_axes([0.12,0.12,0.82,0.8])
        fig5_data = np.zeros((len(params[var1]),len(params[var2]),len(Redges)))

    if a_MP_plot == True:
        fig7 = plt.figure(7,figsize=(10,6))
        ax7  = fig7.add_axes([0.1,0.1,0.67,0.76])
        ax7b = fig7.add_axes([0.1,0.86,0.67,0.1],sharex=ax7)
        ax7c = fig7.add_axes([0.77,0.1,0.1,0.76],sharey=ax7)
        cax7  = fig7.add_axes([0.87,0.1,0.02,0.76])
    if N_mass_hist == True:
        fig8 = plt.figure(8)
        ax8 = fig8.add_axes([0.12,0.12,0.82,0.8])

    if M_hists == True:
        fig9 = plt.figure(9)
        ax9 = fig9.add_axes([0.12,0.12,0.82,0.8])

    if aM_scat == True:
        fig10 = plt.figure(10)
        ax10 = fig10.add_axes([0.12,0.12,0.82,0.8])

    if d_hists == True:
        fig11 = plt.figure(11)
        ax11 = fig11.add_axes([0.12,0.12,0.82,0.8])

    if M_barplot == True:
        fig12 = plt.figure(12)
        ax12 = fig12.add_axes([0.12,0.12,0.82,0.8])
        Mindex = np.arange(len(Medges)-1)

    if disc0_plots == True:
        fig13 = plt.figure(13,figsize=(15,4))
        ax13a = fig13.add_axes([0.1,0.15,0.15,0.8])
        ax13b = fig13.add_axes([0.32,0.15,0.15,0.8])
        ax13c = fig13.add_axes([0.54,0.15,0.15,0.8])
        ax13d = fig13.add_axes([0.76,0.15,0.15,0.8])

    #==== Main histogram generation loop ====#
    dim = N_combi
    if params['M_P0'][0] == 'sweep':
        dim = N_combi * len(params['M_P0'][1:])
    big_store = np.zeros((len(params[var1]),len(params[var2]),14,dim))
    
    for i in range(len(params[var1])):
        for j in range(len(params[var2])):
            run_params = params.copy()
            run_params[var1] = [params[var1][i]]
            run_params[var2] = [params[var2][j]]

            label = str(params[var1][i])
            label += ', ' + labels[var2]+'= ' + str(params[var2][j])
            Ais,Afs,t_20AUs,typeIs,MP0s,MPs,Zs,weights,Ads,Q0s,M_disc0s,M_infall0s,M_star0s = histogram(
                run_params,N_combi=N_combi,init=init,d_hists=d_hists,
                col=plot_cols[i],lw=lws[j],ls=lss[j],label=label)
            init = False
            
            weighted_number = np.sum(weights)
            A_bins,bins = np.histogram(Afs,bins=Redges,weights=weights)
            A_bins = A_bins/weighted_number*100

            big_store[i,j,0,:] = Ais
            big_store[i,j,1,:] = Afs
            big_store[i,j,2,:] = t_20AUs
            big_store[i,j,3,:] = typeIs
            big_store[i,j,4,:] = MPs
            big_store[i,j,5,:] = Zs
            big_store[i,j,6,:] = weights
            big_store[i,j,7,0:len(Redges)-1] = A_bins
            big_store[i,j,8,:] = MP0s/c.MJ
            big_store[i,j,9,:] = Ads
            big_store[i,j,9,:] = Q0s
            big_store[i,j,11,:] = M_disc0s
            big_store[i,j,12,:] = M_infall0s
            big_store[i,j,13,:] = M_star0s


    #Save big store for KS tests
    datestr = datetime.today().strftime('%m-%d-%H-%M')
    np.save(datestr+'bigstore'+str(params['Model'][0][0])+'_KS_N'+str(N_combi), big_store)

    
    #==== Load planet masses ====#
    MPs = np.array(params['M_P0'])
    if params['M_P0'][0] == 'sweep':
        MPs = np.array(params['M_P0'][1:])
    elif params['M_P0'][0] == 'lin':
        MPs = np.arange(params['M_P0'][1],params['M_P0'][2],1)
    elif params['M_P0'][0] == 'log':
        MPs = np.logspace(np.log10(params['M_P0'][1]),np.log10(params['M_P0'][2]),dim)

    MP_mids = (MPs[1:]+MPs[:-1])/2
    MP_edges = np.zeros(len(MPs)+1)
    MP_edges[1:-1] = MP_mids
    MP_edges[0] = MPs[0]/1.1
    MP_edges[-1] = MPs[-1]+0.5 #*1.1


    #==== Load Redge binned data ====#
    A_bins = big_store[:,:,7,:len(Redges)-1] + 1e-2
    survive_count = np.sum(A_bins[:,:,1:],axis=2)

    A_vals = np.logspace(-2,2.7,40)

    #==== Main histogram Figure N-a ====#
    plt.figure(1)
    #plt.axvline(params['R_ins'][0],ls='--')
    plt.xlabel('Orbit [AU]')
    plt.ylabel('Percentage of systems')
    plt.semilogx()
    plt.yscale('log', nonposy='clip')
    plt.xlim(0.01,params['Rout'][-1]*1.2)
    plt.legend(frameon=False,loc=9)


    #==== Plot bar charts ====#
    if bar_pops == True:
        plt.figure(2)
        for i in range(len(params[var1])):
            for j in range(len(params[var2])):
                if len(params[var1]) > 2:
                    if j != 0:
                        label = ''
                    else:
                        label = labels[var1]+ ': ' + str(params[var1][i])
                else:
                    label = str(params[var1][i]) +', '+labels[var2]+'='+str(params[var2][j])+'- {:.1f}'.format(survive_count[i,j])+'%'

                if j == 0:
                    plt.bar(index+i*bar_width,A_bins[i,j],bar_width,color=plot_cols[i],alpha=0.5,label=label)
                elif j == 1:
                    plt.bar(index+i*bar_width,A_bins[i,j],bar_width,label=label,lw=1.5,edgecolor=plot_cols[i],fill=False,alpha=0.5)
                elif j == 2:
                    plt.bar(index+i*bar_width,A_bins[i,j],bar_width,label=label,lw=1.5,ls='--',edgecolor=plot_cols[i],fill=False,alpha=0.5)
        plt.legend(frameon=False)
        plt.ylabel('Percentage count')
        plt.xlabel('Final separation [AU]')
        plt.yscale('log', nonposy='clip')
        ax2.set_xticks(index+(bar_width*(N_params-1))/2)
        ax2.set_xticklabels(xlabels)
        '''
        if len(params[var1]) > 2:
            #==== Bar type legend ====#
            for i in range(len(params[var2])):
                label = labels[var2]+': '+str(params[var2][i])
                if var2 != ' ':
                    if i == 0:
                        plt.bar(-2,[1],label=label,color=plot_cols[0],alpha=0.5)
                    elif i == 1:
                        plt.bar(-2,[1],label=label,color='#ffffff',lw=1.5,edgecolor=plot_cols[0],alpha=0.5)
                    elif i == 2:
                        plt.bar(-2,[1],label=label,color='#ffffff',lw=1.5,ls='--',edgecolor=plot_cols[0],alpha=0.5)
        '''
        ax2.set_xlim(-bar_width)
        plt.legend(frameon=True,loc=4)
        ax2.text(0,1.02,params['Model'][0],transform=ax2.transAxes)

        plt.figure(6)
        for i in range(len(params[var1])):
            for j in range(len(params[var2])):
                try:
                    if params['MPf_pow'][0] == 0:
                        FP = FP_facs[i]
                    elif params['MPf_pow'][0] == -1.3:
                        FP = FP_facs[2+i]
                except:
                    FP = [0,0,0]
                if var1 == 'Model':
                    slabel = str(params[var1][i][:-10])
                else:
                    slabel = str(params[var1][i])
                if var2 == 'MPf_pow':
                    slabel += ', '+labels[var2]+ r'$^{}$'.format('{'+str(params[var2][j])+'}')   
                else:
                    slabel += ''#', '+labels[var2]+'= '+str(params[var2][j]) 
                ax6.plot(np.arange(len(Redges)-1),A_bins[i,j],color=plot_cols[i],label=slabel,ls=lss[j])
                ax6.plot(np.arange(len(Redges)-1),A_bins[i,j]*FP[0],color=plot_cols[i],ls='--',alpha=0.6,label=slabel+': $F_{best}$ fit')
                ax6.fill_between(np.arange(len(Redges)-1),A_bins[i,j]*FP[1],A_bins[i,j]*FP[2],color=plot_cols[i],alpha=0.15)


        #ax6.errorbar(3,Vigan_2_14,yerr=[[V_errm],[V_errp]],color=plot_cols[i+1],capsize=3,label=r'V+172-14$M_J$')
        ax6.errorbar(3,MB0,yerr=[[MB0-MBd],[MBu-MB0]],color=plot_cols[i+1],capsize=3,label=r'DI: $M_P<12M_J$',marker='d')
        ax6.errorbar(1,Cumming_2_12,yerr=error_4_15,color=plot_cols[i+2],capsize=3,label=r'Cumming+08: 2-12$M_J$',marker='d')
        ax6.errorbar(1,Cumming_4_12,yerr=error_4_15,color=plot_cols[i+3],capsize=3,label=r'Cumming+08: 4-12$M_J$',marker='d')

        ax6.text(0,1.02,labels['MPf_pow']+r'$^{}$'.format('{'+str(params['MPf_pow'][0])+'}'),transform=ax6.transAxes)
        ax6.set_xticks(np.arange(len(Redges)-1))
        ax6.set_xticklabels(xlabels)
        ax6.set_ylabel('Percentage of systems')
        ax6.set_xlabel('Final separation [AU]')
        ax6.set_ylim(0.008,105)
        plt.semilogy()
        plt.legend(frameon=False,ncol=2)


        
    #==== Make timescale plots ====#
    #var1=Z, set RP_mode='None'
    if plot_timescales == True:
        plt.figure(4)
        for i in range(len(params[var1])):
            for j in range(len(params[var2])):
                t_20AUs = big_store[i,j,2]/c.sec_per_year

                if params['M_P0'][0] == 'sweep':
                    pt_20AUs = t_20AUs.reshape(N_combi,len(MPs))
                    for k in range(N_combi):
                        ax4.plot(MPs,pt_20AUs[k],color=plot_cols[i],alpha=0.05,lw=3)
                ax4.plot([],[],color=plot_cols[i],alpha=0.5,lw=3,label=r'$t_{20AU}$')
                        
        #==== KH timescales ====#
        tMPs = [0.5,1,2,3,4,5,6,7,8,9,10,11,12]
        t_KHs_Z0  = np.array([1331597,473011,178805,94424,53456,32363,22940,15969,12466,9099,7013,6131,4947])
        t_KHs_Z_03 = np.array([845251,301366,98822,49770,29337,18064,12736,8622,5312,2700,1654,1465,1372])
        t_KHs_Z03 = np.array([2100413,753594,282595,167634,100977,63350,45525,31655,22331,16064,10962,7424,4789])
        #ax4.plot(tMPs,t_KHs_Z_03,color=plot_cols[0],label=r'$t_{KH}$, $[M/H] = [M/H]_{\odot}/$ 2')
        ax4.plot(tMPs,t_KHs_Z0,color=plot_cols[1],label=r'$t_{KH}$, $[M/H] = [M/H]_{\odot}$',ls='--')#,marker='o' 
        #ax4.plot(tMPs,t_KHs_Z03,color=plot_cols[2],label=r'$t_{KH}$, $[M/H] = [M/H]_{\odot}\times 2$')

        ax4_text = params['Model'][0]
        ax4_text += r', $\alpha_0$ = '+str(params['alpha0'][0])
        ax4.text(0.02,1.02,ax4_text,transform=ax4.transAxes)

        ax4.set_xlim(np.min(MPs)*0.5,np.max(MPs)*1.05)
        ax4.set_ylabel('Time [Years]')
        ax4.set_xlabel(r'Mass [$M_J$]')
        plt.legend(frameon=False)#,loc=2)
        ax4.set_ylim(120,1.6e7)
        plt.semilogy()


        
    #==== Make survive plot as a function of MPs ====#
    if (survive_plot == True) and (var1 == 'M_P0'):
        plt.figure(5)
        
        for j in range(len(params[var2])):
            label1 = labels[str(var2)]+'= '+str(params[var2][j])
            ax5.plot(MPs,A_bins[:,j,1]/100,color=plot_cols[j],label=label1)
            ax5.plot(MPs,A_bins[:,j,2]/100,color=plot_cols[j],ls='--')
            ax5.plot(MPs,A_bins[:,j,3]/100,color=plot_cols[j],ls=':')                

        ax5.plot([],[],color=plot_cols[0],label=r'$R_{final}$: '+str(xlabels[1])+' AU')
        ax5.plot([],[],color=plot_cols[0],label=r'$R_{final}$: '+str(xlabels[2])+' AU',ls='--')
        ax5.plot([],[],color=plot_cols[0],label=r'$R_{final}$: '+str(xlabels[3])+' AU',ls=':')
        ax5.set_xlabel(r'$M_P$ [$M_J$]')
        ax5.set_ylabel(r'Survival fraction')
        plt.legend(frameon=False)
        ax5.text(0,1.02,params['Model'][0],transform=ax5.transAxes)

    #==== Make a_MP plot ====#
    if (a_MP_plot == True) and (var1 == 'M_P0'):
        a_MP_store = np.zeros((len(MPs)+1,2,len(A_vals)-1))

        Weights = (MPs/np.max(MPs))**params['MPf_pow'][0]
        NormWeights = Weights * 100 / N_combi /np.sum(Weights)
        
        for i in range(len(MPs)):
            Ais = big_store[i,0,0,:]
            Afs = big_store[i,0,1,:]
            Ai_vals = np.logspace(1,np.log10(A_vals[-1]),len(A_vals))
            valsi,bins = np.histogram(Ais,bins=Ai_vals,weights=NormWeights[i]*np.ones(len(Ais)))
            valsf,bins = np.histogram(Afs,bins=A_vals,weights=NormWeights[i]*np.ones(len(Afs)))
            a_MP_store[1+i,0,:] = valsi
            a_MP_store[1+i,1,:] = valsf
            
        #=== Plotting code ==#      
        contour_store = np.zeros(np.shape(a_MP_store[:,0,:]))
        contour_store[:-1,:] = a_MP_store[:-1,0,:]
        ax7.contour(Ai_vals[1:],MP_edges,contour_store,levels=[0,100],label='Initial population',colors=plot_cols[4],linewidth=2)

        #NaCo-LP limit contours
        NaCo = np.genfromtxt(obs_fol+'NaCo-LP_average_detmap.txt',delimiter='  ', invalid_raise = False)
        a_NaCo = NaCo[0,1:]
        MP_NaCo = NaCo[1:,0]
        NaCo_cont = ax7.contour(a_NaCo,MP_NaCo,NaCo[1:,1:]*100,levels=[0.5,5,50],colors=plot_cols[0],linestyles='dashed')
        #plt.clabel(NaCo_cont,colors=plot_cols[0],fmt='%2.1f',fontsize=10)

        #Blank 0% parts
        aMP_cmap = plt.cm.YlOrRd
        aMP_cmap.set_under('white')
        
        a_MP_imshowf = ax7.pcolormesh(A_vals,MP_edges,a_MP_store[1:,1,:],cmap=aMP_cmap,vmin=1e-2)

        #Save data
        a_MP_output = np.hstack((MPs[:,None],a_MP_store[1:,1,:]))
        a_MP_output = np.vstack((A_vals[None,:],a_MP_output))
        np.savetxt('0327_a_MP_'+str(params['Model'][0][0])+'_output_forMB.csv', a_MP_output,delimiter=',',fmt='%.3e')
        
        ax7.set_xscale('log')
        ax7.set_yscale('log')
        ax7.set_xlabel('Final separation [AU]')
        ax7.set_ylabel(r'Planet mass [$M_J$]')
        ax7.set_xlim(0.01,1.4*Redges[-1])#200)
        ax7.set_ylim(0.4,25)
        
        #-- exo.org --#
        exo_A,exo_MP = exoplanets_org(split='Mstar')
        ax7.scatter(exo_A,exo_MP,color=plot_cols[0],s=10,label=r'Exoplanet.org 0.7$M_{\odot}$<$M_*$<1.4$M_{\odot}$')
        exoMPbins = np.arange(1,20,1)

        ax7.errorbar(275,12,yerr=2,color=plot_cols[3],capsize=3,label=r'AB Pic',marker='d')

        #-- mini histogram code --#
        im_ais  = np.ravel(big_store[:,0,0,:])
        im_afs  = np.ravel(big_store[:,0,1,:])
        im_MPs  = np.ravel(big_store[:,0,4,:])
        Weightsb = np.ravel(NormWeights[:,None]*np.ones(len(Ais))[None,:])
        Weightse = np.ones(len(exo_A))*100/len(exo_A[exo_MP>1])
        ax7b.hist(im_afs,bins=A_vals,histtype='step',color=plot_cols[2],weights=Weightsb)
        ax7b.hist(im_ais,bins=A_vals,histtype='step',color=plot_cols[4],weights=Weightsb)
        ax7b.hist(exo_A[exo_MP>1],bins=A_vals,histtype='step',color=plot_cols[0],weights=Weightse[exo_MP>1])
        ax7c.hist(im_MPs[im_afs>0],bins=MP_edges,orientation='horizontal',histtype='step',color=plot_cols[2],weights=Weightsb[im_afs>0])
        ax7c.hist(im_MPs,bins=MP_edges,orientation='horizontal',histtype='step',color=plot_cols[4],weights=Weightsb)
        ax7c.hist(exo_MP,bins=exoMPbins,orientation='horizontal',histtype='step',color=plot_cols[0],weights=Weightse)
        ax7b.set_yscale('log')
        ax7c.set_xscale('log')
        plt.setp(ax7b.get_xticklabels(), visible=False)
        plt.setp(ax7c.get_yticklabels(), visible=False)
        
        #-- colorbar code --#
        plt.colorbar(a_MP_imshowf,cax=cax7)#,orientation='horizontal')
        cax7.text(5,0.7,'Frequency of appearence (%)',rotation=270)
        ax7b.text(0,1.1,params['Model'][0],transform=ax7b.transAxes)
        #ax7.axvline(20,color=plot_cols[3],ls='--',label='Vigan 2017 constraint')
        plt.legend(loc=4,frameon=False,ncol=2)


        
    #==== Make Rbinned mass spectrum histograms ====#
    if N_mass_hist == True:
        for j in range(len(params[var2])):
            Afs    = np.ravel(big_store[:,j,1,:])
            Masses = np.ravel(big_store[:,j,4,:])

            #Weights
            MPf_pow = params['MPf_pow'][0]
            if var2 == 'MPf_pow':
                MPf_pow = params['MPf_pow'][j]    
            weights =  (Masses/np.max(Masses))**MPf_pow
            weights = weights/np.sum(weights)*100

            #MB DI constraint factors
            try:
                if params['MPf_pow'][0] == 0:
                    FP = FP_facs[j]
                elif params['MPf_pow'][0] == -1.3:
                    FP = FP_facs[2+j]
            except:
                FP = [0,0,0]
            print 'FP', FP
            weights *= FP[0]
                    
            #Plot all Rbin histograms with correct weights
            for k in range(len(Redges)-2):
                bin_Masses = Masses[(Afs>Redges[k+1]) & (Afs<Redges[k+2])]
                bin_weights = weights[(Afs>Redges[k+1]) & (Afs<Redges[k+2])]
                label = ''
                if k == 0:
                    label = str(params[var2][j])[:-10]+r': $F_{best}$ fit'
                #print bin_Masses
                #print MP_edges
                ax8.hist(bin_Masses,bins=MP_edges,histtype='step',color=plot_cols[j],alpha=0.8,label=label,weights=bin_weights,ls=lss[k])
                if j == len (params[var2])-1:
                    ax8.plot([],[],color=plot_cols[0],label=r'$R_{final}$: '+str(xlabels[k+1])+' AU',ls=lss[k])

        #Cumming figures
        N_C08 = np.zeros(len(MP_edges)-1)
        for i in range(len(N_C08)):
            N_C08[i] = cumming_powerlaw(MP0=MP_edges[i],MP1=MP_edges[i+1],A0=0.031,A1=Redges[2])*100
        ax8.plot(MPs,N_C08,color=plot_cols[3],label='Cumming+08 R<5AU')       
            
        ax8.legend(frameon=False)
        ax8.set_xlabel(r'Planet Mass [$M_J$]')
        ax8.set_ylabel('Frequency of appearence (%)')
        ax8.text(0,1.02,labels['MPf_pow']+r'$^{}$'.format('{'+str(params['MPf_pow'][0])+'}'),transform=ax8.transAxes)
        ax8.semilogy()

    if M_hists == True:
        A_out = 5#1000 #5
        M_bins = np.logspace(0,4,30)
        for i in range(len(params[var1])):
            for j in range(len(params[var2])):
                As     = big_store[i,j,1,:]
                masses = big_store[i,j,4,:]
                weights = big_store[i,j,6,:]
                weights *= 100/sum(weights)
                label = str(params[var1][i])+' '+str(params[var2][j])
                #ax9.hist(masses,M_bins,label=label,weights=weights,histtype='step',ls=lss[j],color=plot_cols[i])
                ax9.hist(masses[As<A_out],M_bins,label=label,weights=weights[As<A_out],histtype='step',ls=':',color=plot_cols[i])

                #ax9.scatter(As,masses,color=plot_cols[i])
        ax9.semilogx()
        ax9.set_xlabel(r'Planet mass [$M_J$]')
        ax9.set_ylabel('Percentage of total')

        Obs_per = []
        Obs_Ms = M_bins[:10] #np.arange(1,15,1)
        Obs_Mmids = (Obs_Ms[1:]+Obs_Ms[:-1])/2
        for i in range(len(Obs_Ms)-1):
            Obs_per.append(cumming_powerlaw(Obs_Ms[i],Obs_Ms[i+1],0,5))
        ax9.scatter(Obs_Mmids,np.array(Obs_per)*1000)
        
        Tro_A,Tro_MP = troup_comps()
        Tro_weights = np.ones(len(Tro_A))/len(Tro_A)*100
        ax9.hist(Tro_MP,M_bins,label='Tr',histtype='step',color=plot_cols[10],weights=Tro_weights)
        exo_A,exo_MP = exoplanets_org(split='Mstar')
        exo_A  = exo_A[exo_MP>M_bins[0]]
        exo_MP = exo_MP[exo_MP>M_bins[0]]
        exo_weights = np.ones(len(exo_A))/len(exo_A)*100
        ax9.hist(exo_MP,M_bins,label='Exo',histtype='step',color=plot_cols[9],weights=exo_weights)
        Rag_A,Rag_MP = Raghavan_comps()
        Rag_MP = Rag_MP[Rag_A<A_out]
        Rag_A = Rag_A[Rag_A<A_out]
        Rag_weights = np.ones(len(Rag_A))/len(Rag_A)*100
        ax9.hist(Rag_MP,M_bins,label='Rag',histtype='step',color=plot_cols[11],weights=Rag_weights)
        
        
        ax9.legend(frameon=False)

    if aM_scat == True:
        for i in range(len(params[var1])):
            for j in range(len(params[var2])):
                ind = len(params[var2])*i + j
                As     = big_store[i,j,1,:]
                masses = big_store[i,j,4,:]
                A0s    = big_store[i,j,0,:]
                mass0s = big_store[i,j,8,:]
                ax10.scatter(As,masses,color=plot_cols[ind],label=str(params[var1][i]),alpha=0.5,s=10)#,marker=syms[j])
                ax10.scatter(A0s,mass0s,color=plot_cols[ind],label=str(params[var1][i]),alpha=0.1,s=5,marker=syms[1])

        ax10.semilogx()
        ax10.semilogy()
        ax10.set_xlabel('Separation [AU]')
        ax10.set_ylabel(r'Planet mass [$M_J$]')

        vig17 = patches.Rectangle((20,2),280,73,linewidth=1,edgecolor='r',facecolor='none')
        ax10.add_patch(vig17)

        Tro_A,Tro_MP = troup_comps()
        ZSo_A,ZSo_MP = ZS09_comps()
        exo_A,exo_MP = exoplanets_org(split='Mstar')
        Rag_A,Rag_MP = Raghavan_comps()

        ax10.scatter(exo_A,exo_MP,color='#ff5555',s=3,alpha=0.5,marker=syms[-1],label=r'Exoplanet.org 0.7$M_{\odot}$<$M_*$<1.4$M_{\odot}$')
        ax10.scatter(Tro_A,Tro_MP,color=plot_cols[-3],s=5,marker=syms[-2],label=r'Troup et al. 16 0.7$M_{\odot}$<$M_*$<1.4$M_{\odot}$')
        ax10.scatter(ZSo_A,ZSo_MP,color=plot_cols[-4],s=5,marker=syms[-3],label=r'ZS09 0.7$M_{\odot}$<$M_*$<1.4$M_{\odot}$')
        ax10.scatter(Rag_A,Rag_MP,color=plot_cols[-5],s=5,marker=syms[-4],label=r'Rag10 0.67$M_{\odot}$<$M_*$<1.22$M_{\odot}$')
        ax10.errorbar(275,12,yerr=2,color=plot_cols[3],capsize=3,label=r'AB Pic',marker='d')

        ax10.plot([0.3,300],[21,9],label='Brown dwarf desert',color=plot_cols[-5])

        #NaCo-LP limit contours
        NaCo = np.genfromtxt(obs_fol+'NaCo-LP_average_detmap.txt',delimiter='  ', invalid_raise = False)
        a_NaCo = NaCo[0,1:]
        MP_NaCo = NaCo[1:,0]
        NaCo_cont = ax10.contour(a_NaCo,MP_NaCo,NaCo[1:,1:]*100,levels=[0.5,5,50],colors='r',linestyles='dashed')
        
        ax10.legend(frameon=False)

    if M_barplot == True:            
        plt.figure(12)
        for i in range(len(params[var1])):
            As  = big_store[i,j,1,:]
            MPs = big_store[i,j,4,:]
            MPs[As<0] = 0
            M_bars,binsm = np.histogram(MPs,bins=Medges)
            plt.bar(Mindex+i*bar_width,M_bars,bar_width,color=plot_cols[i],alpha=0.5,label=label)
                
        plt.legend(frameon=False)
        plt.ylabel('Percentage count')
        plt.xlabel(r'Final planet mass [$M_J$]')
        #plt.yscale('log', nonposy='clip')
        ax12.set_xticks(Mindex+(bar_width*(N_params-1))/2)
        ax12.set_xticklabels(Mxlabels)
        ax12.set_xlim(-bar_width)
        plt.legend(frameon=True,loc=4)
        ax12.text(0,1.02,params['Model'][0],transform=ax12.transAxes)
        
    if d_hists == True:
        #A_diss = np.logspace(0,1.3,100)
        A_diss = np.linspace(10,25,30)
        for i in range(len(params[var1])):
            for j in range(len(params[var2])):
                Ads = big_store[i,j,9,:]
                Norm = np.ones(len(Ads))*100/len(Ads)
                label = ''
                if j == 0:
                    label = str(params[var1][i])
                ax11.hist(Ads,bins=A_diss,histtype='step',color=plot_cols[i],ls=lss[j],label=label,weights=Norm)
        for j in range(len(params[var2])):
            ax11.plot([],[],color=plot_cols[0],ls=lss[j],label=r'$M_P=$'+str(params[var2][j])+r'$M_J$')

        ax11.legend(frameon=False)
        ax11.set_xlabel('Disruption location [AU]')
        ax11.set_ylabel('Percentage of population disrupted')

    if disc0_plots == True:
        for i in range(len(params[var1])):
            for j in range(len(params[var2])):
                A0s        = big_store[i,j,0,:] 
                Q0s        = big_store[i,j,9,:]
                M_disc0s   = big_store[i,j,11,:] 
                M_infall0s = big_store[i,j,12,:]
                M_tot0s = (M_disc0s + M_infall0s) /c.Msol
                M_star0s = big_store[i,j,13,:] /c.Msol
                M_q0s = M_tot0s/M_star0s
                
                M_bins = np.logspace(-2,1.5,20)

                Qind  = np.where(Q0s<Q_thresh)# & (A0s>50))
                Qnind = np.where(Q0s>Q_thresh)# | (A0s<50))

                ax13a.hist(A0s,bins=A_vals,histtype='step',color=plot_cols[i],cumulative=True)
                ax13b.hist(M_tot0s,bins=M_bins,histtype='step',color=plot_cols[i],cumulative=True)
                ax13c.hist(M_q0s,bins=np.logspace(-1.5,1.2),histtype='step',color=plot_cols[i],cumulative=True)
                ax13d.scatter(A0s[Qind],M_tot0s[Qind],color=plot_cols[0])
                ax13d.scatter(A0s[Qnind],M_tot0s[Qnind],color=plot_cols[1])

        ax13a.set_xlabel(r'$R_{\rm out}$ [AU]')
        ax13b.set_xlabel(r'$M_{\rm disc}$ [$M_{\odot}$]')
        ax13c.set_xlabel(r'Disc/Star mass ratio')
        ax13d.set_xlabel(r'$R_{\rm out}$ [AU]')
        ax13a.set_ylabel(r'Cumulative number')
        ax13c.set_ylabel(r'Cumulative number')
        ax13b.set_ylabel(r'Cumulative number')
        ax13d.set_ylabel(r'$M_{\rm disc}$ [$M_{\odot}$]')
        ax13a.semilogx()
        ax13b.semilogx()
        ax13c.semilogx()        
        ax13d.semilogx()
        ax13d.semilogy()

    return 





if __name__ == '__main__':
    
    #run_models(params=['A'],var='M_P0',timescales=False)#var='M_P0')#'SG_corr') #'M_P',#'alpha'
    #run_models(params=['A','alpha'],var='Model',timescales=False)#var='M_P0')#'SG_corr') #'M_P',#'alpha'
    #run_models(params=['A','R_P','alpha'],var='M_P0',timescales=False)#var='M_P0')#'SG_corr') #'M_P',#'alpha'
    #run_models(params=['A','M_P'],var='M_P0',timescales=False)#var='M_P0')#'SG_corr') #'M_P',#'alpha'
    run_models(params=['A','M_P','M_disc'],var='M_P0',timescales=False,mode='GI_gasacc')#var='M_P0')#'SG_corr') #'M_P',#'alpha'

    #plot_allona_data()
    #abundance()
    
    #hist_grid(disc_diag=False,d_hists=False,bar_pops=True,pos_spread=False,plot_timescales=True,survive_plot=True,a_MP_plot=True,N_mass_hist=True,M_hists=False,aM_scat=False) 
    #hist_grid(disc_diag=False,d_hists=False,bar_pops=False,pos_spread=False,plot_timescales=False,survive_plot=False,a_MP_plot=False,N_mass_hist=False,M_hists=True,aM_scat=True,M_barplot=True) 
    #hist_grid(M_hists=True,aM_scat=True,M_barplot=True,disc0_plots=True) 

    #simple_Mdot()
    
    #exoplanets_org(split='Zstar',plot_data=True)

    #gas_accretion_ests()

    #N_KS_test(f1='04-25-12-25bigstoreQ_KS_N100.npy',f2='04-25-12-26bigstoreQ_KS_N100.npy')
    #N_KS_test(f1='04-25-12-25bigstoreQ_KS_N100.npy',f2='04-25-12-33bigstoreQ_KS_N100.npy')
    #N_KS_test(f1='04-25-12-33bigstoreQ_KS_N100.npy',f2='04-25-12-26bigstoreQ_KS_N100.npy')

    #N_KS_test(f1='04-24-20:50bigstoreS_KS_N10000.npy',f2='04-25-12-55bigstoreS_KS_N10000.npy')
    #N_KS_test(f1='04-24-20:51bigstoreQ_KS_N10000.npy',f2='04-25-12-58bigstoreQ_KS_N10000.npy')

    plt.show()
