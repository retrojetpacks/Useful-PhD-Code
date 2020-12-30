'''Functions to run population sysnthesis models'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import random

plot_cols = ['#440044','#2EC4C6','#FFC82E','#FF1493','#6a5acd','#33cc33','#0055ff',
             '#dd44cc','#cd6622','#A8CFFF','#62D0FF','#2EC4C6','#FFC82E','#CD6622','#FF1493',
             '#440044','#6a5acd','#33cc33','#0055ff','#dd44cc','#cd6622','#A8CFFF','#62D0FF']
lss  = ['-','--',':','-.','-','-','-','-']

def PP_observability(t_ALMA,N_sys=1000,GI_frac=1,PP_delay=1,young_half=False,Beam=15,Inc_cut=0):
    '''Estiamte what fraction of PPs might be observable'''
    Rin  = 20
    Rout = 100

    f_obs = np.ones(N_sys) #1=detection,0=non-detection

    #==== Spatial detection ====#
    #Beam = 4#15 #7.7=15AU 7.8=6AU at 140pc
    R_disc = np.random.normal(15,15,N_sys).clip(min=0) #Rough fit to Cieza, Ophiuchus
    Inc_max = np.pi/2 - Inc_cut/180*np.pi #radians
    mu_min = np.cos(Inc_max)
    
    mu = np.random.rand(N_sys)*(1-mu_min) + mu_min
    phi = np.random.rand(N_sys)*2*np.pi
    a_PP   = np.random.rand(N_sys)*(Rout-Rin) + Rin
    
    #d_gap = (a_PP - R_disc).clip(min=0)
    #D =  d_gap*mu # Projected star-PP separation

    R_disc_proj = abs(R_disc*mu)
    a_PP_proj = abs(a_PP * np.sqrt(np.cos(phi)**2 + (np.sin(phi)*mu)**2))
    D = (a_PP_proj - R_disc_proj).clip(min=0)

    f_obs[D<Beam] = 0 

    
    #==== Temporal detection ====#
    MPP_max = 10
    MPP_min = 1
    MPP = (MPP_max-MPP_min)*np.random.rand(N_sys) + MPP_min
    
    t_PP =  6e5 * (MPP)**-1.44 * PP_delay #Rough fit from Vazan for 2x Zsol
    #t_sys = np.random.normal(2e6,1e6,N_sys)     #Disc age (Taurus 1-3 Myr) assume normal...
    t_sys = np.random.normal(0,1,N_sys)     #Disc age (Taurus 1-3 Myr) assume normal...
    if young_half == True:
        t_sys = -abs(t_sys)
    t_sys *= 1e6
    t_sys += 2e6

    f_obs[t_sys>t_PP] = 0


    #==== ALMA Config ====#
    #Obs propto T_PP**4
    T = 20*(100/a_PP)**0.5
    T_fac = (T/30)**4
    
    #Obs propto R_PP**2
    R_PP = 1.5*np.log10(MPP)+0.8 #AU
    R_fac = (R_PP/1.5)**2

    t_canon = 10 #T=30K, R=1.5AU PP detected after 10 mins
    t_req = t_canon / T_fac/R_fac
    f_obs[t_req>t_ALMA] = 0

    
    Obs_percent = len(f_obs[f_obs==1])/len(f_obs) * GI_frac * 100
    #print mode, ' observed fraction: ', Obs_percent, ' percent'
    
    return Obs_percent, a_PP[f_obs==1], MPP[f_obs==1]




def PP_sep_plot():
    '''Lodato ALMA gap and beams..'''
    rs = np.linspace(10,150,20)
    qs = np.logspace(-4,-2,30)
    r_mids = (rs[1:]+rs[:-1])/2
    q_mids = (qs[1:]+qs[:-1])/2

    mesh_rs, mesh_qs = np.meshgrid(r_mids,q_mids,indexing='xy')
    Gap = 5.5 * mesh_rs * (mesh_qs/3)**(1/3)

    plt.figure(5,figsize=(6,4.5))
    plt.pcolormesh(rs,qs,Gap,cmap='viridis',vmin=0,vmax=100)#,norm=LogNorm(vmin=2,vmax=50))
    cbar = plt.colorbar()
    plt.semilogy()
    plt.xlabel('Protoplanet orbital separation')
    plt.ylabel('Protoplanet/star mass ratio')
    cbar.set_label('Beam size [AU]', rotation=270)

    cycle7_beams = [4,6,15,19,34,57] #AU at 140 pc
    cycle7_labs  = ['7.9','7.8','7.7','7.6','7.5','7.4']
        
    Cs = plt.contour(r_mids,q_mids,Gap,levels=cycle7_beams,colors='w',linewidth=2)
    fmt = {}
    for l, s in zip(Cs.levels, cycle7_labs):
        fmt[l] = s

    plt.clabel(Cs,Cs.levels,inline=True,fmt=fmt)
    
    return


def PP_int_probs():
    '''Plot the probability of finding a PP with ALMA for different models'''
    fig1 = plt.figure(1,facecolor='white')
    ax1 = fig1.add_axes([0.15,0.55,0.8,0.4])
    ax1b = fig1.add_axes([0.15,0.15,0.8,0.4],sharex=ax1)

    t_tot = 1200
    int_times = np.array([2,4,8,16,32,64,128]) #minutes
    N_sys = 1000000
    PP_common,PP_rare,PP_sub,PP_age,PP_inc = [],[],[],[],[]
    
    for i in int_times:
        p, a_PP, MPP = PP_observability(i,N_sys,GI_frac=1,PP_delay=5)
        PP_common.append(p)
        
        p, a_PP, MPP = PP_observability(i,N_sys,GI_frac=0.02,PP_delay=1)
        PP_rare.append(p)

        p, a_PP, MPP = PP_observability(i,N_sys,GI_frac=1,PP_delay=5,Beam=4)
        PP_sub.append(p)

        p, a_PP, MPP = PP_observability(i,N_sys,GI_frac=1,PP_delay=5,young_half=True,Beam=4)
        PP_age.append(p)

        p, a_PP, MPP = PP_observability(i,N_sys,GI_frac=1,PP_delay=5,Inc_cut=20)
        PP_inc.append(p)
        
        
    ax1.plot(int_times,PP_rare,color=plot_cols[0],label='Protoplanets are rare')
    ax1.plot(int_times,PP_common,color=plot_cols[2],label='Common: 7.7')
    #ax1.plot(int_times,PP_inc,color=plot_cols[11],label=r'Common: 7.7, $\theta < 70 \degree$')
    ax1.plot(int_times,PP_sub,color=plot_cols[3],label=r'Common: 7.9')
    ax1.plot(int_times,PP_age,color=plot_cols[9],label=r'Common: 7.9, $t_{\rm sys}$ < 2 Myr')

    N_choice = t_tot / int_times
    ax1b.plot(int_times,PP_rare*N_choice/100,color=plot_cols[0])
    ax1b.plot(int_times,PP_common*N_choice/100,color=plot_cols[2])
    #ax1b.plot(int_times,PP_inc*N_choice/100,color=plot_cols[11])
    ax1b.plot(int_times,PP_sub*N_choice/100,color=plot_cols[3])
    ax1b.plot(int_times,PP_age*N_choice/100,color=plot_cols[9])

    ax1.semilogx()
    ax1.semilogy()
    ax1b.semilogy()

    ax1b.set_xlabel('Integration time [minutes]')
    ax1.set_ylabel('Percent of systems')
    ax1b.set_ylabel(r'$N_{PP}$ observed ['+str(t_tot)+' mins]')
    ax1.legend(frameon=False)




    

def a_MP_obs():
    '''Make mass separation scatter for observed systems'''

    N_sys = 10000
    GI_fracs = [0.02,1]
    PP_delays = [1,5]
    int_times = [10,60]
    labels = ['GI rare','GI common']

    fig2 = plt.figure(2,facecolor='white')
    ax2 = fig2.add_axes([0.15,0.15,0.65,0.65])
    ax2b = fig2.add_axes([0.8,0.15,0.15,0.65])
    ax2c = fig2.add_axes([0.15,0.8,0.65,0.15])

    for i in range(len(GI_fracs)):
        for j in range(len(int_times)):
            ind = len(int_times)*i+j
            p, a_PP, MPP = PP_observability(int_times[j],N_sys,GI_fracs[i],PP_delays[i])
            weights = np.ones(len(a_PP))*GI_fracs[i]/N_sys*100
            
            ax2.scatter(a_PP,MPP,color=plot_cols[ind],alpha=0.5,label=labels[i]+r', '+str(int_times[j])+' mins',s=10)
            ax2b.hist(MPP,histtype='step',color=plot_cols[ind],orientation='horizontal',weights=weights)
            ax2c.hist(a_PP,histtype='step',color=plot_cols[ind],weights=weights)
        
    ax2.legend(ncol=2)
    ax2.set_xlabel('Radius [AU]')
    ax2.set_ylabel('Protoplanet mass [$M_J$]')
    ax2b.set_xlabel('Percent')
    ax2c.set_ylabel('Percent')

    ax2b.semilogx()
    ax2c.semilogy()

    plt.setp(ax2b.get_yticklabels(), visible=False)
    plt.setp(ax2c.get_xticklabels(), visible=False)




    
if __name__ == "__main__":
    PP_sep_plot()
    #a_MP_obs()
    #PP_int_probs()
    plt.show()
