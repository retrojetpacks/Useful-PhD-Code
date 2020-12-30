'''Write Gadget initial condition file in hdf5'''

from __future__ import division
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import basic_astro_functions as b
import astrophysical_constants_cgs as c
import os
from shutil import copyfile
import Quick_Render as Q

#Code Units
code_M     = c.Msol                  #Msol in g
code_L     = 100*c.AU                #100 AU in cm
code_V     = 297837.66               #v_K @ 100 AU #in cm/s
code_time  = code_L/code_V           #seconds #~160 Yrs
code_rho   = code_M/code_L**3
code_E     = code_M*code_V**2
code_G     = c.G /code_L**3*code_M*code_time**2
AU_scale   = 100


gas_col  = '#224499'
dust_col = '#880088'
graft_dir = 'graft_dir/'
init_setups = 'init_setups/'
snapprefix = ''
#Set up parameters


#====+ Initial Condition Dictionaries +====#
#Work in Progress

#Basic Star
Star_A_dict = {'name':'Mstar1','modes':['star'],'M_star':1}

#Relaxed Polytropes
poly_B_dict  = {'modes':['polytrope_Tc'],'N_poly':10000,'M_poly':0.001,'Tc_poly':200,'n_poly':1.5}
poly_C_dict  = {'modes':['polytrope_Tc'],'N_poly':100000,'M_poly':0.001,'Tc_poly':200,'n_poly':1.5}
#Clement polytrope tests
Poly_D_dict  = {'modes':['polytrope_R'],'N_poly':100000,'M_poly':0.005,'R_poly':0.05,'n_poly':1.5}
Poly_E_dict  = {'modes':['polytrope_R'],'N_poly':1000000,'M_poly':0.005,'R_poly':0.05,'n_poly':1.5}
Poly_F_dict  = {'modes':['polytrope_R'],'N_poly':100000,'M_poly':0.005,'R_poly':0.03,'n_poly':1.5}
Poly_G_dict  = {'modes':['polytrope_R'],'N_poly':100000,'M_poly':0.005,'R_poly':0.01,'n_poly':1.5}
#Clump in disc polytropes
Poly_H_dict  = {'modes':['polytrope_R'],'N_poly':100000,'M_poly':0.003,'R_poly':0.03,'n_poly':1.5}
Poly_I_dict  = {'modes':['polytrope_R'],'N_poly':400000,'M_poly':0.003,'R_poly':0.03,'n_poly':1.5}
Poly_J_dict  = {'modes':['polytrope_R'],'N_poly':800000,'M_poly':0.005,'R_poly':0.03,'n_poly':2.5}
Poly_K_dict  = {'modes':['polytrope_R'],'N_poly':100000,'M_poly':0.005,'R_poly':0.03,'n_poly':2.5}
Poly_L_dict  = {'modes':['polytrope_R'],'N_poly':60000,'M_poly':0.003,'R_poly':0.02,'n_poly':2.5}
Poly_M_dict  = {'modes':['polytrope_R'],'N_poly':20000,'M_poly':0.001,'R_poly':0.01,'n_poly':2.5}
Poly_N_dict  = {'modes':['polytrope_R'],'N_poly':140000,'M_poly':0.007,'R_poly':0.04,'n_poly':2.5}
#Radmc protoplanets
Poly_O_dict  = {'modes':['polytrope_R'],'N_poly':100000,'M_poly':0.001,'R_poly':0.01,'n_poly':2.5}
Poly_P_dict  = {'modes':['polytrope_R'],'N_poly':200000,'M_poly':0.002,'R_poly':0.01,'n_poly':2.5}
Poly_Q_dict  = {'modes':['polytrope_R'],'N_poly':300000,'M_poly':0.003,'R_poly':0.01,'n_poly':2.5}
Poly_R_dict  = {'modes':['polytrope_R'],'N_poly':100000,'M_poly':0.001,'R_poly':0.02,'n_poly':2.5}
Poly_S_dict  = {'modes':['polytrope_R'],'N_poly':200000,'M_poly':0.002,'R_poly':0.02,'n_poly':2.5}
Poly_T_dict  = {'modes':['polytrope_R'],'N_poly':300000,'M_poly':0.003,'R_poly':0.02,'n_poly':2.5}
Poly_U_dict  = {'modes':['polytrope_R'],'N_poly':100000,'M_poly':0.001,'R_poly':0.005,'n_poly':2.5}

#Relaxed Discs
disc_B_dict = {'modes':['gas_disc','star'],'N_gas_disc':100000,'Rin':0.1,'Rout':3.0,'M_gas_disc':0.01,'M_star':1,'grad':1,'Teq':20}
disc_C_dict = {'modes':['gas_disc','star'],'N_gas_disc':1000000,'Rin':0.1,'Rout':3.0,'M_gas_disc':0.01,'M_star':1,'grad':1,'Teq':20}
disc_D_dict = {'name':'Disc_N1e5_R00120_M0005','modes':['gas_disc','star'],'N_gas_disc':100000,'Rin':0.1,'Rout':2.0,'M_gas_disc':0.005,'M_star':1,'grad':1,'Teq':20,'Roll':True}
disc_E_dict = {'name':'Disc_N1e5_R00120_M0005_hot','modes':['gas_disc','star'],'N_gas_disc':100000,'Rin':0.1,'Rout':2.0,'M_gas_disc':0.005,'M_star':1,'grad':1,'Teq':30,'T_ind':0.75,'Roll':True}
#Clump in disc Discs
disc_F_dict = {'name':'Disc_N3e5_R00120_M0009','modes':['gas_disc','star'],'N_gas_disc':300000,'Rin':0.1,'Rout':2.0,'M_gas_disc':0.009,'M_star':1,'grad':1,'Teq':20,'T_ind':0.5,'Roll':True}
disc_G_dict = {'name':'Disc_N12e5_R00120_M0009','modes':['gas_disc','star'],'N_gas_disc':1200000,'Rin':0.1,'Rout':2.0,'M_gas_disc':0.009,'M_star':1,'grad':1,'Teq':20,'T_ind':0.5,'Roll':True}
disc_H_dict = {'name':'Disc_N15e5_R00120_M0075','modes':['gas_disc','star'],'N_gas_disc':1500000,'Rin':0.1,'Rout':2.0,'M_gas_disc':0.075,'M_star':1,'grad':1,'Teq':20,'T_ind':0.5,'Roll':True}
disc_I_dict = {'name':'Disc_N15e5_R00110_M0075','modes':['gas_disc','star'],'N_gas_disc':1500000,'Rin':0.1,'Rout':1.0,'M_gas_disc':0.075,'M_star':1,'grad':1,'Teq':20,'T_ind':0.5,'Roll':True}
disc_J_dict = {'name':'Disc_N12e6_R00110_M0075','modes':['gas_disc','star'],'N_gas_disc':12000000,'Rin':0.1,'Rout':1.0,'M_gas_disc':0.075,'M_star':1,'grad':1,'Teq':20,'T_ind':0.5,'Roll':True}
disc_K_dict = {'name':'Disc_N12e6_R00110_M0075','modes':['gas_disc','star'],'N_gas_disc':12000000,'Rin':0.1,'Rout':1.0,'M_gas_disc':0.075,'M_star':1,'grad':1,'Teq':20,'T_ind':0.5,'gamma':c.gamma_dia,'Roll':True}
disc_L_dict = {'name':'Disc_N15e5_R00110_M0075_g75','modes':['gas_disc','star'],'N_gas_disc':1500000,'Rin':0.1,'Rout':1.0,'M_gas_disc':0.075,'M_star':1,'grad':1,'Teq':20,'T_ind':0.5,'gamma':c.gamma_dia,'Roll':True}
disc_M_dict = {'name':'Disc_N2e6_R00110_M01_g75','modes':['gas_disc','star'],'N_gas_disc':2000000,'Rin':0.1,'Rout':1.0,'M_gas_disc':0.1,'M_star':1,'grad':1,'Teq':20,'T_ind':0.5,'gamma':c.gamma_dia,'Roll':True}
disc_N_dict = {'name':'Disc_N16e6_R00110_M01_g75','modes':['gas_disc','star'],'N_gas_disc':16000000,'Rin':0.1,'Rout':1.0,'M_gas_disc':0.1,'M_star':1,'grad':1,'Teq':20,'T_ind':0.5,'gamma':c.gamma_dia,'Roll':True}
disc_O_dict = {'name':'Disc_N4e6_R00120_M01_g53','modes':['gas_disc','star'],'N_gas_disc':4000000,'Rin':0.1,'Rout':2.0,'M_gas_disc':0.2,'M_star':0.8,'grad':1,'Teq':20,'T_ind':0.5,'gamma':c.gamma_mono,'Roll':True}

disc_P_dict = {'name':'Disc_N2e5_R530_M001_g53','modes':['gas_disc','star'],'N_gas_disc':200000,'Rin':0.05,'Rout':0.3,'M_gas_disc':0.01,'M_star':1.0,'grad':1,'Teq':20,'T_ind':0.5,'gamma':c.gamma_mono,'Roll':True}
disc_Q_dict = {'name':'Disc_N1e5_R530_M0001_g53','modes':['gas_disc','star'],'N_gas_disc':100000,'Rin':0.05,'Rout':0.3,'M_gas_disc':0.001,'M_star':1.0,'grad':1,'Teq':20,'T_ind':0.5,'gamma':c.gamma_mono,'Roll':True}
disc_R_dict = {'name':'Disc_N1e6_R530_M001_g53','modes':['gas_disc','star'],'N_gas_disc':1000000,'Rin':0.05,'Rout':0.3,'M_gas_disc':0.01,'M_star':1.0,'grad':1,'Teq':20,'T_ind':0.5,'gamma':c.gamma_mono,'Roll':True}
TW_Hya = {'name':'TW_Hya_','modes':['gas_disc','star'],'N_gas_disc':1000000,'Rin':0.1,'Rout':2.0,'M_gas_disc':7.5e-4,'M_star':0.8,'grad':1,'Teq':20,'T_ind':0.5,'gamma':c.gamma_mono,'Roll':True}
TW_Hya2 = {'name':'TW_Hya2','modes':['gas_disc','star'],'N_gas_disc':200000,'Rin':0.1,'Rout':2.0,'M_gas_disc':2e-3,'M_star':0.8,'grad':1,'Teq':20,'T_ind':0.5,'gamma':c.gamma_mono,'Roll':True}


#Giovanni discs
Giovanni_dict  = {'modes':['gas_disc','star'],'N_gas_disc':2000000,'Rin':0.01,'Rout':1,'M_gas_disc':0.01,'M_star':1,'grad':1,'Teq':20}
Giovanni_dictN1e6 = {'name':'Disc_N1e6_R00210','modes':['gas_disc','star'],'N_gas_disc':1000000,'Rin':0.02,'Rout':1,'M_gas_disc':0.01,'M_star':1,'grad':1,'Teq':20}
Giovanni_dictN1e6_00120 = {'name':'Disc_N1e6_R00120','modes':['gas_disc','star'],'N_gas_disc':1000000,'Rin':0.01,'Rout':2.0,'M_gas_disc':0.02,'M_star':1,'grad':1,'Teq':20,'Roll':True}
Giovanni_dictN1e5_00120 = {'name':'Disc_N1e5_R00120','modes':['gas_disc','star'],'N_gas_disc':100000,'Rin':0.01,'Rout':2.0,'M_gas_disc':0.02,'M_star':1,'grad':1,'Teq':20,'Roll':True}



#Ring test
Ring_dict  = {'name':'Ring_N5e5_R05','modes':['gas_disc','star'],'N_gas_disc':500000,'Rin':0.495,'Rout':0.505,'M_gas_disc':0.01,'M_star':1,'grad':1,'Teq':20}







#====+ Data Generation Functions +====#

def disc(Rin,Rout,M_disc,M_star,N,grad,Teq,Ptype,IDoffset,T_ind=0.5,gamma=c.gamma_mono,Roll=False):
    '''Generate disc of particles. Ptype: PartType0=gas,PartType2=dust
    Teq = equilibrium temp at 100 AU'''
    ID    = np.arange(N)+IDoffset
    q     = 2-grad


    if Ptype == 'PartType0':
        M_sph = M_disc/N
    elif Ptype == 'PartType2':
        M_sph = M_disc/N*Z_met

    Sig0  = q*M_disc/(2*np.pi) * (Rout**q - Rin**q)**(-1)

    
    #Generate Rs including mirroring
    R          = np.zeros(N)
    R[0:2]     = np.array([Rin,Rin])
    Theta      = np.zeros(N)
    rangle = np.random.rand(int(N/2))*np.pi
    Theta[::2] = rangle
    Theta[1::2] = rangle-np.pi
    
    for i in range(int(N/2-1)):
        dR = 2*M_sph / (2*np.pi*Sig0) * R[2*i]**(grad-1)
        R[2*i+2:2*i+4] = R[2*i:2*i+2]+dR

    if Roll == True:
        for i in range(len(R)):
            if R[i] < 1.2*Rin:
                R[i] = 2*Rin-R[i]
            elif R[i] > 0.8*Rout:
                R[i] = 2*Rout-R[i]
        
    X     = R*np.cos(Theta)
    Y     = R*np.sin(Theta)

    #Enclosed mass
    M_plus = np.arange(N)*M_sph
    
    #Velocities
    V_K   =  b.v_kepler((M_star+M_plus)*code_M,R*code_L)/code_L*code_time
    VX    = -V_K*np.sin(Theta)
    VY    =  V_K*np.cos(Theta)
    VZ    =  np.zeros(N)

    T     = Teq* (1/R)**T_ind
    Hs    = R/V_K * np.sqrt(c.kb*T/(c.mu*c.mp)) *code_time/code_L
    Z     = np.random.normal(0,Hs)
    
    print 'Coords',X,Y,Z
    plt.figure(0)
    plt.scatter(X,VY)

    POS   = np.stack((X,Y,Z)).T
    MASS  = np.ones(N)*M_sph
    U     = np.zeros(N)
    
    if Ptype == 'PartType0':
        #Gas pressure
        n       = 11/4
        eta     = n * (Hs/R)**2
        gas_sub = (1-eta)**0.5
        print 'gas sub', gas_sub
        V_K_gas = V_K * gas_sub
        VX, VY  = VX*gas_sub, VY*gas_sub
        U       = T*c.kb / (c.mu*c.mp*(gamma-1))/code_L**2 * code_time**2
        plt.figure(3)
        plt.hist(U*code_L**2/code_time**2,alpha=0.5)

    VEL   = np.stack((VX,VY,VZ)).T

    print 'POS', type(POS), np.shape(POS), POS

    return POS,VEL,ID,MASS,U




def star(M_star,IDoffset):
    POS  = np.array([[0.,0.,0.]])
    VEL  = np.array([[0.,0.,0.]])
    ID   = [IDoffset]
    MASS = [float(M_star)]
    U    = [0]
    return POS,VEL,ID,MASS,U




def polytrope(N,M,R=0,Tc=0,n=1.5,IDoffset=0):
    '''Solve Polytrope. Theta = polytropic temperature. Epsilon = scaled radius.
    n=1.5 -> gamma=5/3 adiabatic solution
    N = number of particles'''
    ID    = np.arange(N)+IDoffset
    M_sph = M/N
    MASS = np.ones(N)*M_sph
    gamma = (1+n)/n

    #Solve Polytrope
    if R==0:
        Rs, rhos, rho_c, polyK_cgs, polyK_code  = b.polytrope(M=M,Tc=Tc,n=n)
    elif Tc==0:
        Rs, rhos, rho_c, polyK_cgs, polyK_code  = b.polytrope(M=M,R=R,n=n)
    
    R_outs   = np.zeros(N)
    rho_outs = np.zeros(N)
    R_outs[0] = (3*M_sph / (4*np.pi*rho_c))**(1/3)
    rho_outs[0] = rho_c
    
    for i in range(N-1):
        rho_i  = np.interp(R_outs[i],Rs,rhos)
        dri            = M_sph/(4*np.pi*R_outs[i]**2*rho_i)
        if (np.isnan(dri) == True) or (np.isinf(dri) == True):
            dri = 0
            rho_i = rho_outs[i]

        R_outs[i+1]    = R_outs[i] + dri
        rho_outs[i+1]  = rho_i
    
    theta_outs = np.arccos(2*np.random.rand(N)-1)
    phi_outs   = np.random.rand(N)*2*np.pi

    plt.figure(1)
    plt.scatter(R_outs,rho_outs)
    plt.figure(2)
    plt.hist(theta_outs,bins=200)
    plt.show()
    
    #Convert to output values
    X = R_outs * np.cos(phi_outs) * np.sin(theta_outs)
    Y = R_outs * np.sin(phi_outs) * np.sin(theta_outs)
    Z = R_outs * np.cos(theta_outs)
    Vs = np.zeros(N)
    
    POS   = np.stack((X,Y,Z)).T
    VEL   = np.stack((Vs,Vs,Vs)).T
    U     = 1/(gamma-1)*polyK_code*rho_outs**(1/n)

    plt.figure(3)
    plt.hist(U*code_L**2/code_time**2,alpha=0.5)
    
    return POS,VEL,ID,MASS,U







#====+ Write Initial Condition Files +====#

def write_hdf5(init_dict):
    '''Write a hdf5 gadget initial condition file'''

    placeholder = 'placeholder'
    try:
        os.remove(placeholder)
    except OSError:
        pass
    
    init = h5py.File(placeholder)
    num_array = np.array([0,0,0,0,0,0])
    modes = init_dict['modes']

    
    #====+ Generate Input Data +====#
    #Gas=0,dust=2,sink=5. Write in order!
   
    #--- Gas Disc ---#
    if 'gas_disc' in modes:
        Ptype      = 'PartType0'
        Type       = init.create_group(Ptype)
        Rin        = init_dict['Rin']
        Rout       = init_dict['Rout']
        M_gas_disc = init_dict['M_gas_disc']
        M_star     = init_dict['M_star']
        N_gas_disc = init_dict['N_gas_disc']
        grad       = init_dict['grad']
        Teq        = init_dict['Teq']
        T_ind      = init_dict['T_ind']
        gamma      = init_dict['gamma']
        filename   = init_dict['name']
        try:
            Roll   = init_dict['Roll']
        except:
            Roll   = False

        POS,VEL,ID,MASS,U = disc(Rin=Rin,Rout=Rout,M_disc=M_gas_disc,
                                 M_star=M_star,N=N_gas_disc,grad=grad,gamma=gamma,
                                 Teq=Teq,T_ind=T_ind,Ptype=Ptype,IDoffset=np.sum(num_array))
        num_array[0] += N_gas_disc
        Type.create_dataset('Coordinates',data=POS)
        Type.create_dataset('Velocities',data=VEL)
        Type.create_dataset('ParticleIDs',data=ID)
        Type.create_dataset('Masses',data=MASS)
        Type.create_dataset('InternalEnergy',data=U)

    #--- Polytrope ---#
    if 'polytrope_Tc' in modes:
        Ptype = 'PartType0'
        Type = init.create_group(Ptype)
        N_poly  = init_dict['N_poly']
        M_poly  = init_dict['M_poly']
        Tc_poly = init_dict['Tc_poly']
        n_poly  = init_dict['n_poly']
        filename   = 'Poly_N'+str(N_poly)+'_M'+str(M_poly)+'_T'+str(Tc_poly)+'_n'+str(n_poly)

        POS,VEL,ID,MASS,U = polytrope(N=N_poly,M=M_poly,Tc=Tc_poly,n=n_poly,
                                      IDoffset=np.sum(num_array))
        num_array[0] += N_poly
        Type.create_dataset('Coordinates',data=POS)
        Type.create_dataset('Velocities',data=VEL)
        Type.create_dataset('ParticleIDs',data=ID)
        Type.create_dataset('Masses',data=MASS)
        Type.create_dataset('InternalEnergy',data=U)

    if 'polytrope_R' in modes:
        Ptype = 'PartType0'
        Type = init.create_group(Ptype)
        N_poly  = init_dict['N_poly']
        M_poly  = init_dict['M_poly']
        R_poly = init_dict['R_poly']
        n_poly  = init_dict['n_poly']
        filename   = 'Poly_N'+str(N_poly)+'_M'+str(M_poly)+'_R'+str(R_poly)+'_n'+str(n_poly)

        POS,VEL,ID,MASS,U = polytrope(N=N_poly,M=M_poly,R=R_poly,n=n_poly,
                                      IDoffset=np.sum(num_array))
        num_array[0] += N_poly
        Type.create_dataset('Coordinates',data=POS)
        Type.create_dataset('Velocities',data=VEL)
        Type.create_dataset('ParticleIDs',data=ID)
        Type.create_dataset('Masses',data=MASS)
        Type.create_dataset('InternalEnergy',data=U)
        
    #--- Dust Disc ---#
    if 'dust_disc' in modes:
        Ptype       = 'PartType2'
        Type        = init.create_group(Ptype)
        Rin         = init_dict['Rin']
        Rout        = init_dict['Rout']
        M_dust_disc = init_dict['M_dust_disc']
        M_star      = init_dict['M_star']
        N_dust_disc = init_dict['N_dust_disc']
        grad        = init_dict['grad']
        Teq         = init_dict['Teq']

        POS,VEL,ID,MASS,U = disc(Rin=Rin,Rout=Rout,M_disc=M_disc,
                                 M_star=M_star,N=num_dust,grad=1,
                                 Teq=Teq,Ptype=Ptype,IDoffset=np.sum(num_array))
        num_array[2] += num_dust
        Type.create_dataset('Coordinates',data=POS)
        Type.create_dataset('Velocities',data=VEL)
        Type.create_dataset('ParticleIDs',data=ID)
        Type.create_dataset('Masses',data=MASS)


    #--- Star ---#
    if 'star' in modes:
        Ptype  = 'PartType5'
        Type   = init.create_group(Ptype)
        M_star = init_dict['M_star']
        filename   = init_dict['name']
        print 'M_star', M_star
        
        POS,VEL,ID,MASS,U = star(M_star,IDoffset=np.sum(num_array))
        print 'MASS', MASS
        num_array[5] += 1
        Type.create_dataset('Coordinates',data=POS)
        Type.create_dataset('Velocities',data=VEL)
        Type.create_dataset('ParticleIDs',data=ID)
        Type.create_dataset('Masses',data=MASS)
        

    #====+ Build Header +====#
    header = init.create_group('Header')
    header.attrs.create('NumPart_ThisFile',num_array)
    header.attrs.create('NumPart_Total',num_array)
    header.attrs.create('NumPart_Total_HighWord',np.array([0,0,0,0,0,0]))
    header.attrs.create('MassTable',np.array([0,0,0,0,0,0]))
    header.attrs.create('Time',0.0)
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

    #Finalise File Details
    for char in '.':
        filename = filename.replace(char,'')
    filename += '.hdf5'
    try:
        os.remove(filename)
    except OSError:
        pass
    os.rename(placeholder,filename)

    init.close()
    return





def cut_sphere(file1,cut_pos,cut_R,cut_Rin=0,centred=False):
    '''Cut a sphere out of an sph distribution. Useful for inserting polytropes'''
    print 'Beginning sphere cut'
    S     = b.Load_Snap('graft_dir/'+file1,snapprefix='')
    print 'N initial gas', np.shape(S.gas_pos)

    if centred == True:
        com,com_v = S.max_rho()
        print 'COM', com
        S.move(com)

    gas_offset = S.gas_pos-cut_pos
    gas_R = np.sqrt(gas_offset[:,0]**2+gas_offset[:,1]**2+gas_offset[:,2]**2)
    #inds  = np.where(gas_R>cut_R)
    inds  = np.where((gas_R>cut_R) | (gas_R<cut_Rin)) #logical or
    
    S.gas_pos = S.gas_pos[inds]
    S.gas_vel = S.gas_vel[inds]
    S.gas_u   = S.gas_u[inds]
    S.gas_rho = S.gas_rho[inds]
    print 'N gas post cut', np.shape(S.gas_pos)
    
    S.save_as_gadget('graft_dir/'+file1[:-5]+'_cut.hdf5')
    print 'Saving '+'graft_dir/'+file1[:-5]+'_cut.hdf5'
    print 'Ending sphere cut\n'

    return




def graft(file1,file2,graft_file,offset_pos=np.array([0.,0.,0.]),offset_vel=np.array([0.,0.,0.]),offset_vK=False,
          COM_correction=False,rho_correction=False):

    print 'Starting graft'
    print file1
    print file2
    S1 = b.Load_Snap('graft_dir/'+file1,snapprefix='')
    S2 = b.Load_Snap('graft_dir/'+file2,snapprefix='')


    if COM_correction == True:
        S2.gas_pos -= np.mean(S2.gas_pos,axis=1)
        S2.gas_vel -= np.mean(S2.gas_vel,axis=1)
    if rho_correction == True:
        rho_pos,rho_vel  = S2.max_rho()
        print 'AAAA', rho_pos
        print offset_pos
        S2.gas_pos -= rho_pos
        S2.gas_vel -= rho_vel
        try:
            S2.dust_pos -= rho_pos
            S2.dust_vel -= rho_vel
        except:
            print 'No dust in S2'
        print 'AAAA', rho_pos
        
    if offset_vK == True:
        RP = np.sqrt(offset_pos[0]**2+offset_pos[1]**2+offset_pos[2]**2)

        S1.gas_r = np.sqrt(S1.gas_pos[:,0]**2+S1.gas_pos[:,1]**2+S1.gas_pos[:,2]**2)
        M_gas_in = S1.M_gas * len(S1.gas_r[S1.gas_r<RP])
        theta  = np.arctan2(offset_pos[1],offset_pos[0])
        vK = b.v_kepler((S1.M_star+M_gas_in)*code_M,RP*code_L)/code_L*code_time
        vK_old = b.v_kepler((S1.M_star)*code_M,RP*code_L)/code_L*code_time

        print S1.M_star, M_gas_in
        print vK, vK_old
        offset_vel = np.array([-1*np.sin(theta)*vK,np.cos(theta)*vK,0])

    print 'Offset pos', offset_pos
        
    #==== Combine Gas ====#
    S1.gas_pos = np.vstack((S1.gas_pos,S2.gas_pos+offset_pos))
    S1.gas_vel = np.vstack((S1.gas_vel,S2.gas_vel+offset_vel))
    S1.gas_u   = np.hstack((S1.gas_u,S2.gas_u))
    try:
        S.gas_rho = np.hstack((S1.gas_rho,S2.gas_rho))
    except:
        print 'No rho data'
    S1.N_gas = len(S1.gas_u)

    '''
    #==== Combine Dust ====#
    S1.dust_pos = np.vstack((S1.dust_pos,S2.dust_pos+offset_pos))
    S1.dust_vel = np.vstack((S1.dust_vel,S2.dust_vel+offset_vel))
    S1.dust_a = np.hstack((S1.dust_a,S2.dust_a))
    S1.N_dust = len(S1.dust_a)
    
    #==== Combine sinks ====#
    S1.M_planets = np.hstack((S1.M_planets,S2.M_planets))
    S1.planets_pos = np.vstack((S1.planets_pos,S2.planets_pos+offset_pos))
    S1.planets_vel = np.vstack((S1.planets_vel,S2.planets_vel+offset_vel))
    S1.N_planets = S1.N_planets+S2.N_planets
    '''
    S1.save_as_gadget('graft_dir/'+graft_file)
    
    print 'ww', S1.M_planets
    print 'ww', S1.M_star
    print 'Finishing graft for ',graft_file, '\n'
    return



def V_settle(dust_r,dust_z,r_gas,M_sph,M_star,grain_a,grain_rho):
    '''Calculate analytic dust settling velocities'''
    rbins  = np.linspace(np.min(dust_r),np.max(dust_r),501)
    rbin_mids = (rbins[1:]+rbins[:-1])/2
    drbin = rbins[1]-rbins[0]
    N_bins = np.histogram(r_gas,rbins)[0]
    binned_Sig = M_sph*N_bins /  (2*np.pi*rbin_mids*drbin)

    #Calculate analytic quantities
    T      = b.T_profile(R=rbin_mids,T0=20,R0=1,power=-0.5)
    cs     = np.sqrt(c.kb*T/(c.mu*c.mp))/code_L*code_time
    Om_K   = b.v_kepler(M_star*code_M,rbin_mids*code_L)/code_L*code_time /rbin_mids
    rho_0  = binned_Sig*Om_K / (np.sqrt(2*np.pi)*cs)         
    a      = grain_a/code_L
    rho_a  = grain_rho/code_M*code_L**3
    t_stop = a*rho_a /rho_0/cs *np.sqrt(np.pi/8)

    #Allocate analytic settling velocities for each dust particle
    VZs = np.zeros(len(dust_r))
    for i in range(len(dust_r)):
        ind = np.argmin((dust_r[i]-rbin_mids)**2)
        t_stopi = t_stop[ind]
        Om_Ki   = Om_K[ind]
        csi     = cs[ind]
        VZs[i] = -Om_Ki**2*t_stopi*dust_z[i] * np.exp(dust_z[i]**2/2 * (Om_Ki/csi)**2)

      
    return VZs
    

def Z_settled(dust_r,dust_z,r_gas,Z_gas,gas_hs,M_sph,M_star,grain_a,grain_rho,alpha_AV):
    '''Calculate analytic dust settled scale'''
    rbins  = np.linspace(np.min(dust_r),np.max(dust_r),501)
    rbin_mids = (rbins[1:]+rbins[:-1])/2
    drbin = rbins[1]-rbins[0]
    N_bins = np.histogram(r_gas,rbins)[0]
    binned_Sig = M_sph*N_bins /  (2*np.pi*rbin_mids*drbin)
    binned_hs,std_hs = b.calc_binned_data(gas_hs,r_gas,rbins)
    #binned_Hs = b.calc_binned_data(Z_gas,r_gas,rbins,H68_mode=True)
    
    #Calculate analytic quantities
    T      = b.T_profile(R=rbin_mids,T0=20,R0=1,power=-0.5)
    cs     = np.sqrt(c.kb*T/(c.mu*c.mp))/code_L*code_time
    Om_K   = b.v_kepler(M_star*code_M,rbin_mids*code_L)/code_L*code_time /rbin_mids
    rho_0  = binned_Sig*Om_K / (np.sqrt(2*np.pi)*cs)         
    a      = grain_a/code_L
    rho_a  = grain_rho/code_M*code_L**3
    t_stop_a = rho_a /rho_0/cs *np.sqrt(np.pi/8)
    print 'lah!', np.shape(t_stop_a)
    print len(dust_r)
    
    
    #Allocate analytic settling velocities for each dust particle
    Zs = np.zeros(len(dust_r))
    for i in range(len(dust_r)):
        ind = np.argmin((dust_r[i]-rbin_mids)**2)
        t_stop_ai = t_stop_a[ind]
        Om_Ki   = Om_K[ind]
        #print t_stop_ai, Om_Ki, a[ind]
        supp_fac = np.sqrt(alpha_AV/t_stop_ai/Om_Ki/a[ind])
        supp_fac = np.clip(supp_fac,-1,1)
        Zs[i] = Z_gas[i] * supp_fac
    print np.shape(Zs)
    print Zs
    return Zs



def one2one_dust(file1,M_dust_frac,Rin=0.1,Rout=1,Zsupp=1,
                 static_dust=False,settled_dust=False,alpha_AV=1.,
                 recentre_gas=False,
                 TEST_vsettle=False,TEST_dust_ring=False,Nring=1e6,
                 grain_a=1.,grain_rho=3.,dust_readin=False,logamin=-4,logamax=1,
                 name='',N_dust_frac=1.0,
                 polydust=False,poly_R=3):
    '''Match dust particles to gas particles in a given region
    Zsupp = suppression factor
    TEST_vsettle - start dust particles with analytic v settle
    TEST_dust_ring - start particles in narrow dust ring
    N_dust_frac = fraction of gas particles that recieve dust particles
    '''
    
    #========= Build new file name ========#
    oldfile = file1
    file1 = file1[:-5]+'_MD'+str(M_dust_frac).replace('.','')+'.hdf5'
    file1 = file1[:-5]+'_Z'+str(Zsupp)+'.hdf5'
    if TEST_vsettle == True:
        print 
        file1 = file1[:-5]+'_a'+str(grain_a).replace('.','')+'.hdf5'
    if TEST_dust_ring == True:
        file1 = file1[:-5]+'_dustring.hdf5'
    if dust_readin == True:
        file1 = file1[:-5]+'_dustreadin.hdf5'
    if settled_dust == True:
        file1 = file1[:-5]+'_Zset.hdf5'
    if N_dust_frac != 1.0:
        file1 = file1[:-5]+'_df'+str(N_dust_frac).replace('.','')+'.hdf5'
    try:
        os.remove(graft_dir+file1)
    except OSError:
        pass
    copyfile(graft_dir+oldfile,graft_dir+file1)
    print 'Dust addition started for ' + str(file1)

    
    #==== Load hdf5 datasets ====#
    S     = b.Load_Snap('graft_dir/'+oldfile,snapprefix='')

    if recentre_gas == True:
        S.gas_pos = S.gas_pos - np.mean(S.gas_pos)
        S.gas_vel = S.gas_vel - np.mean(S.gas_vel)

    #==== Calculate dust positions ====#
    r_gas     = np.sqrt(S.gas_pos[:,0]**2+S.gas_pos[:,1]**2+S.gas_pos[:,2]**2)
    tmp_dust_pos  = S.gas_pos[(r_gas<Rout)&(r_gas>Rin),:]
    #tmp_N_dust = int(len(r_gas)*N_dust_frac)
    tmp_M_dust    = M_dust_frac * S.M_gas / N_dust_frac

    if N_dust_frac != 1.0:
        np.random.shuffle(tmp_dust_pos)
        tmp_dust_pos = tmp_dust_pos[:tmp_N_dust]
    dust_r    = np.sqrt(tmp_dust_pos[:,0]**2+tmp_dust_pos[:,1]**2+tmp_dust_pos[:,2]**2)
    theta     = np.arctan2(tmp_dust_pos[:,1],tmp_dust_pos[:,0]) 

    #==== Vertical dust settling ====#
    mean_gas_Z = np.mean(S.gas_pos[:,2])
    tmp_dust_pos[:,2] = (tmp_dust_pos[:,2]-mean_gas_Z)/Zsupp + mean_gas_Z
    
    if dust_readin == True:
        print 'Setting dust radii!'
        dust_as = np.logspace(logamin,logamax,S.N_dust)
        np.random.shuffle(dust_as)
        if settled_dust == True:
            gas_hs = S.gas_h
            tmp_dust_pos[:,2] = Z_settled(dust_r,tmp_dust_pos[:,2],r_gas,S.gas_pos[:,2],
                                      S.gas_h,S.M_gas,S.M_star,dust_as,grain_rho,alpha_AV)
    
    #==== Dust Ring Test ====#
    if TEST_dust_ring == True:
        sortgasZ = np.sort(np.sqrt(S.gas_pos[:,2]**2))
        GasH     = sortgasZ[int(0.68*len(sortgasZ))]
        Rmid     = (Rout+Rin)/2
        dR       = 0.01
        dust_r   = Rmid + (np.random.rand(Nring)-0.5)*dR
        theta    = 2*np.pi*np.linspace(0,1,Nring)
        dustZs   = GasH/Zsupp* (np.random.rand(Nring)-0.5)*2  
        tmp_dust_pos = np.array([dust_r*np.cos(theta),dust_r*np.sin(theta),dustZs]).T
        tmp_N_dust = int(Nring)
        
    #==== Dust Velocities ====#
    VZ        = np.zeros(len(theta))
    if TEST_vsettle == True:
        VZ = V_settle(dust_r,tmp_dust_pos[:,2],r_gas,S.M_gas,S.M_star,grain_a)
    if static_dust == False:
        vK        = b.v_kepler(S.M_star*code_M,dust_r*code_L)/code_L*code_time
        tmp_dust_vel  = np.array([-1*np.sin(theta)*vK,np.cos(theta)*vK,VZ]).T
    else:
        tmp_dust_vel = np.zeros((S.N_gas,3))

    #======== Update S.dust ========#
    print 'M DUST', tmp_M_dust
    S.add_dust(tmp_M_dust,tmp_dust_pos,tmp_dust_vel)
        
    S.save_as_gadget('graft_dir/'+file1)

    print 'new mean gas pos',  np.mean(S.gas_pos[:,0]),np.mean(S.gas_pos[:,1]),np.mean(S.gas_pos[:,2])
    print 'new mean dust pos', np.mean(S.dust_pos[:,0]),np.mean(S.dust_pos[:,1]),np.mean(S.dust_pos[:,2])
    print 'Mean gas Z', np.mean(S.gas_pos[:,2])
    print 'Mean dust Z', np.mean(S.dust_pos[:,2])
    print 'New dust pos', np.shape(S.dust_pos)
    print 'New dust vel', np.shape(S.dust_vel),
    print 'Saving ', file1,'\n'
    
    return



def add_dust_profile(file1,dust_file):
    '''Add an arbitrary dust profile to a hdf5 file..'''

    S = b.Load_Snap('graft_dir/'+file1,snapprefix='')
    N_dust = S.N_gas
    
    #Load profile
    load_dir = '/home/r/rjh73/code/TWHya_rad/'
    dat = np.genfromtxt(load_dir+dust_file,skip_header=6)

    R = dat[:,1]/code_L
    R_mids = (R[1:]+R[:-1])/2
    sig = dat[:,5] /code_M * code_L**2
    sig_mids = (sig[1:]+sig[:-1])/2
    dR = np.diff(R)
    an_masses = sig_mids * 2*np.pi * R_mids * dR
    tot_mass = sum(an_masses)
    print 'M_tot', tot_mass*c.Msol/c.ME
    Mdust = tot_mass/N_dust

    v_Ks = b.v_kepler(S.M_star*code_M,R_mids*code_L) /code_L*code_time
    Ts = b.T_profile(R_mids,T0=20)
    cs = b.sound_speed(Ts) /code_L*code_time
    Hs = v_Ks*R_mids/cs

    prob_dens = an_masses/tot_mass
    an_Ns = (prob_dens*N_dust).astype(int) #dust number per annulus
    cum_an_Ns = np.pad(np.cumsum(an_Ns),(1,0),'constant',constant_values=(0))
    
    dust_pos = np.zeros((N_dust,3))
    dust_vel = np.zeros((N_dust,3))
    
    for i in range(len(R_mids)):
        i0 = cum_an_Ns[i]
        i1 = cum_an_Ns[i+1]
        
        rs   = np.random.uniform(R[i],R[i+1],an_Ns[i])
        azis = np.random.uniform(0,2*np.pi,an_Ns[i])
    
        dust_pos[i0:i1,0] = rs*np.cos(azis)
        dust_pos[i0:i1,1] = rs*np.sin(azis)
        dust_pos[i0:i1,2] = np.random.normal(0,Hs[i]/10,an_Ns[i]) #Suppressed by 10
        dust_vel[i0:i1,0] = -rs*np.sin(azis)
        dust_vel[i0:i1,1] = rs*np.cos(azis)
        dust_vel[i0:i1,2] = 0

        
    S.add_dust(Mdust,dust_pos,dust_vel)
    S.save_as_gadget('graft_dir/'+file1[:-5]+dust_file+'.hdf5')
    print 'Gas mass: ', S.M_gas*S.N_gas*code_M/c.MJ, 'MJ'
    print 'Dust mass: ', S.M_dust*S.N_dust * code_M/c.MJ , 'MJ'

    return
    


def insert_sink_planet(file1,MP,pos_P=np.array([0,0,0]),rhomax=False):
    '''Insert a planet. MP in MJ'''
    print 'Begin insert sink particle'
    S = b.Load_Snap('graft_dir/'+file1,snapprefix='')
    newname = file1[:-5]+'_MP'+str(MP).replace('.0','').replace('.','')+'.hdf5'
    MP = MP/code_M*c.MJ
    try:
        os.remove(graft_dir+newname)
    except OSError:
        pass
    copyfile(graft_dir+file1,graft_dir+newname)

    

    #==== Calculate Planet data ====# 
    print 'Pos P', pos_P*AU_scale, 'AU'
    if rhomax == True:
        rho_sort  = np.argsort(S.gas_rho)
        pos_P = np.mean(S.gas_pos[rho_sort[-1000:],:],axis=0)
    rP        = np.sqrt(pos_P[0]**2+pos_P[1]**2)
    theta     = np.arctan2(pos_P[1],pos_P[0])   
    gas_r = np.sqrt(S.gas_pos[:,0]**2+S.gas_pos[:,1]**2+S.gas_pos[:,2]**2)

    Mgas_interior = S.M_gas*len(gas_r[gas_r<rP])
    vK        = b.v_kepler((S.M_star+Mgas_interior)*code_M,rP*code_L)/code_L*code_time
    vel_P     = np.array([-1*np.sin(theta)*vK,np.cos(theta)*vK,0])

    S.N_planets += 1
    print np.shape(S.planets_pos)
    print np.shape(pos_P)
    print np.shape(pos_P[None,:])

    S.planets_pos = np.vstack((S.planets_pos,pos_P[None,:]))
    S.planets_vel = np.vstack((S.planets_vel,vel_P[None,:]))
    S.M_planets   = np.hstack((S.M_planets,np.array([MP])))

    print 'New planet:', pos_P, vel_P
    print 'New filename:', newname
    S.save_as_gadget('graft_dir/'+newname)
    print 'End insert sink particle'
    return



def Gdust_removal(Gfile):
    '''Load Sergei initial snap. Remove dust and planet. make hdf5 for dust addition!
    Place 'snapshot_000' in graft_dir/fname/..'''
    fname = 'graft_dir/'+Gfile.rstrip('/')+'.hdf5'
    
    load_dict = Q.load_Gsnap('graft_dir/',Gfile,'000','gas')

    #==== Load sph data ====#
    headertime  = load_dict['headertime']
    N_sph       = load_dict['N_sph']       
    M_sph       = load_dict['M_sph']      
    M_star      = load_dict['M_star']
    N_planets   = load_dict['N_planets']
    M_planets   = load_dict['M_planets']   
    pos_planets = load_dict['pos_planets']
    vel_planets = load_dict['vel_planets']
    sph_pos     = load_dict['sph_pos']
    sph_vel     = load_dict['sph_vel']
    sph_U       = load_dict['sph_A']
    num_gas = len(sph_U)
    
    print M_star
    print pos_planets
    print np.mean(sph_pos,axis=0)
    print sph_pos
    
    
    #==== Build hdf5 file ====#
    try:
        os.remove(fname)
    except OSError:
        pass
    init = h5py.File(fname)
    num_array =np.array([num_gas,0,0,0,0,1])
    
    #====+ Build Header +====#
    header = init.create_group('Header')
    header.attrs.create('NumPart_ThisFile',num_array)
    header.attrs.create('NumPart_Total',num_array)
    header.attrs.create('NumPart_Total_HighWord',np.array([0,0,0,0,0,0]))
    header.attrs.create('MassTable',np.array([0,0,0,0,0,0]))
    header.attrs.create('Time',0.0)
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

    '''
    plt.figure(4)
    plt.scatter(sph_pos[:,0],sph_pos[:,1])
    plt.show()
    '''
    
    Type0 = init.create_group('PartType0')
    Type0.create_dataset('Coordinates',data=sph_pos)
    Type0.create_dataset('Velocities',data=sph_vel)
    Type0.create_dataset('ParticleIDs',data=np.arange(num_gas))
    Type0.create_dataset('Masses',data=np.ones(num_gas)*M_sph)
    Type0.create_dataset('InternalEnergy',data=sph_U)
    
    Type5 = init.create_group('PartType5')
    Type5.create_dataset('Coordinates',data=np.array([0.,0.,0.],dtype=float))
    Type5.create_dataset('Velocities',data=np.array([0.,0.,0.],dtype=float))
    Type5.create_dataset('ParticleIDs',data=[num_gas])
    Type5.create_dataset('Masses',data=[M_star])

    init.close()

    
    return


    


    
    
def load_test(file1): 
    load_test = h5py.File(file1,'r')
    print '\n'
    print("Keys: %s" % load_test.keys())
    keys = load_test.keys()

    header = load_test[keys[0]]
    #print header.attrs.items()
    attrs = header.attrs.items()
    for i in range(len(attrs)):
        print attrs[i]

    for i in range(len(keys)-1):
        print '\n', keys[i+1]
        
        typei = load_test[keys[i+1]]
        print typei.attrs.items()
        print 'pos',  typei['Coordinates']
        print 'mean pos', np.mean(typei['Coordinates'],axis=0)
        print 'pos',  typei['Coordinates'][:]
        print 'vel',  typei['Velocities'][:]
        #print 'v_K',  np.sqrt(typei['Velocities'][:100,0]**2 + typei['Velocities'][:100,1]**2) 
        print 'ids',  typei['ParticleIDs'][:]
        print 'mass', typei['Masses'][:]
        if keys[i+1] == 'PartType0':
            print 'u',    typei['InternalEnergy'][:]

        try:
            typei['DustRadius'][:]
            print 'dust a', typei['DustRadius'][:]
        except:
            pass
    return




if __name__ == "__main__":

    #==== Tests on Dust Accretion project ====#
    #Gdust_removal('O_M2_beta10_a1_n2e6/')
    #one2one_dust('O_M2_beta10_a1_n2e6.hdf5',M_ddisc=0.001,Rin=0.3,Rout=2)
    #load_test('graft_dir/','O_M2_beta10_a1_n2e6_dust.hdf5')
    #Gdust_removal('O_M2_beta01_n1e6/')
    #one2one_dust('O_M2_beta01_n1e6.hdf5',M_ddisc=0.000001,Rin=0.3,Rout=2,TEST_vsettle=True,grain_a=1)
    #one2one_dust('O_M2_beta01_n1e6.hdf5',M_ddisc=0.000001,Rin=0.3,Rout=2,TEST_vsettle=True,grain_a=0.1)
    #one2one_dust('O_M2_beta01_n1e6.hdf5',M_ddisc=0.000001,Rin=0.3,Rout=2,TEST_vsettle=True,grain_a=0.01)
    #one2one_dust('O_M2_beta01_n1e6.hdf5',M_ddisc=0.000001,Rin=0.3,Rout=2,TEST_vsettle=True,grain_a=0.001)
    #load_test('graft_dir/','O_M2_beta01_n1e6_a001.hdf5')
    
    #Use Gio Disc for non-self gravity dominated discs
    #one2one_dust('Gio_N1e6_213.hdf5',M_ddisc=0.000001,Rin=0.3,Rout=0.9,TEST_vsettle=True,grain_a=0.1)
    #one2one_dust('Gio_N1e6_213.hdf5',M_ddisc=0.000001,Rin=0.3,Rout=0.9,TEST_vsettle=True,grain_a=0.01)
    #one2one_dust('Gio_N1e6_213.hdf5',M_ddisc=0.000001,Rin=0.3,Rout=0.9,TEST_vsettle=True,grain_a=0.001)
    #one2one_dust('Gio_N1e6_213.hdf5',M_ddisc=0.000001,Rin=0.3,Rout=0.9,TEST_vsettle=True,grain_a=0.0001)
    #load_test('graft_dir/','Gio_N1e6_213_a001.hdf5')

    #one2one_dust('Gio_N1e6_213.hdf5',M_ddisc=0.000001,Rin=0.3,Rout=0.9)
    #load_test('graft_dir/','Gio_N1e6_213_dust.hdf5')

    #one2one_dust('Disc_N1e6_R0130_M002_beta01_S301.hdf5',M_dust_frac=1e-8,Rin=0.49,Rout=0.51,Zsupp=10,TEST_dust_ring=True)
    #load_test('graft_dir/','Disc_N1e6_R0130_M002_beta01_S301_dustring.hdf5')


    
    #==== Giovanni Project ===#
    #write_hdf5(init_dict=Giovanni_dict)
    #write_hdf5(init_dict=Giovanni_dictN1e6)
    #insert_sink_planet('Gio_N1e6_213.hdf5',0.1,0.5,0,Z=0)
    #insert_sink_planet('Gio_N1e6_213.hdf5',0.5,0.5,0,Z=0)
    #insert_sink_planet('Gio_N1e6_213.hdf5',1.0,0.5,0,Z=0)
    #insert_sink_planet('Gio_N1e6_213.hdf5',2.0,0.5,0,Z=0)
    #insert_sink_planet('Gio_N1e6_213.hdf5',4.0,0.5,0,Z=0)
    #insert_sink_planet('Gio_N1e6_213.hdf5',8.0,0.5,0,Z=0)
    #one2one_dust('Gio_N1e6_213.hdf5',M_dust_frac=0.001,Rin=0.2,Rout=1,Zsupp=1)
    #one2one_dust('Gio_N1e6_aav1_S301.hdf5',M_dust_frac=0.001,Rin=0.2,Rout=1,Zsupp=10)
    #one2one_dust('Gio_N1e6_aav01_S301.hdf5',M_dust_frac=0.001,Rin=0.2,Rout=1,Zsupp=10)
    #one2one_dust('Gio_N1e6_aav1_S301.hdf5',M_dust_frac=0.01,Rin=0.2,Rout=1,Zsupp=10)
    #one2one_dust('Gio_N1e6_aav01_S301.hdf5',M_dust_frac=0.01,Rin=0.2,Rout=1,Zsupp=10)
    #write_hdf5(init_dict=Giovanni_dictN1e6_00120)
    #write_hdf5(init_dict=Giovanni_dictN1e5_00120)
    #one2one_dust('Disc_N1e5_R00120.hdf5',M_dust_frac=0.01,Rin=0.2,Rout=2.0,Zsupp=10,dust_readin=True)
    #one2one_dust('Gio_N1e6_aav01_R00120_S040.hdf5',M_dust_frac=0.01,Rin=0.2,Rout=2.0,Zsupp=10)
    #one2one_dust('Gio_N1e6_aav01_R00120_S040.hdf5',M_dust_frac=0.1,Rin=0.2,Rout=2.0,Zsupp=10)
    #insert_sink_planet('Gio_N1e6_aav01_R00120_S040_MD001_Z10.hdf5',0.3,0.6,0,Z=0)
    #insert_sink_planet('Gio_N1e6_aav01_R00120_S040_MD001_Z10.hdf5',1.0,0.6,0,Z=0)
    #insert_sink_planet('Gio_N1e6_aav01_R00120_S040_MD001_Z10.hdf5',3.0,0.6,0,Z=0)
    #insert_sink_planet('Gio_N1e6_aav01_R00120_S040_MD01_Z10.hdf5',0.3,0.6,0,Z=0)
    #insert_sink_planet('Gio_N1e6_aav01_R00120_S040_MD01_Z10.hdf5',1.0,0.6,0,Z=0)
    #insert_sink_planet('Gio_N1e6_aav01_R00120_S040_MD01_Z10.hdf5',3.0,0.6,0,Z=0)
    #one2one_dust('Gio_N1e6_aav01_R00120_S040.hdf5',M_dust_frac=0.01,Rin=0.2,Rout=2.0,dust_readin=True,settled_dust=True,alpha_AV=0.1)
    #one2one_dust('Gio_N1e6_aav01_R00120_S040.hdf5',M_dust_frac=0.1,Rin=0.2,Rout=2.0,dust_readin=True,settled_dust=True,alpha_AV=0.1)
    #load_test('graft_dir/','Gio_N1e6_aav01_R00120_S040_MD001_dustreadin_Zset.hdf5')
    #one2one_dust('Gio_N1e6_aav01_R00120_S100.hdf5',M_dust_frac=0.01,Rin=0.2,Rout=2.0,Zsupp=10)
    #one2one_dust('Gio_N1e6_aav01_R00120_S100.hdf5',M_dust_frac=0.1,Rin=0.2,Rout=2.0,Zsupp=10)
    #one2one_dust('Gio_N1e6_aav01_R00120_S100.hdf5',M_dust_frac=0.01,Rin=0.2,Rout=2.0,dust_readin=True,settled_dust=True,alpha_AV=0.1)
    #one2one_dust('Gio_N1e6_aav01_R00120_S100.hdf5',M_dust_frac=0.1,Rin=0.2,Rout=2.0,dust_readin=True,settled_dust=True,alpha_AV=0.1)
    #load_test('graft_dir/','Gio_N1e6_aav01_R00120_S040_MD01_dustreadin_Zset.hdf5')
    #load_test('/rfs/TAG/rjh73/Gio_disc/','Gio_N1e6_aav01_R00120_MD001_dustreadin_Zset/snapshot_040.hdf5')
    #load_test('graft_dir/','Disc_N1e5_R00120_dust_Z10_dustreadin.hdf5')
    #load_test('/rfs/TAG/rjh73/Gio_disc/','Gio_N1e5_aav01_dustin/snapshot_006.hdf5')



    
    #Dust Work
    #one2one_dust('Disc_N100000_M001.hdf5',M_ddisc=0.001)
    #one2one_dust('Disc_N100000_M001_relaxed.hdf5',M_ddisc=0.001,Rin=0.3,Rout=2)
    #one2one_dust('Disc_N100000_M001_relaxed_sink.hdf5',M_ddisc=0.001,Rin=0.3,Rout=2)
    #one2one_dust('Disc_N100000_Poly.hdf5',M_ddisc=0.001,Rin=0.3,Rout=2)
    


    
    #===== Clement disruption runs ====#
    #write_hdf5(init_dict=Poly_D_dict)
    #write_hdf5(init_dict=Poly_E_dict)
    #write_hdf5(init_dict=Poly_F_dict)
    #write_hdf5(init_dict=Poly_G_dict)


    #one2one_dust('Poly_N1e5_M5_R5_n15_S008.hdf5',M_dust_frac=1e-4,Rin=0.0,Rout=2.0,static_dust=True,recentre_gas=True)
    #load_test('graft_dir/','Poly_N1e5_M5_R5_n15_S008_dust.hdf5')
    #one2one_dust('Poly_N1e5_M5_R5_n15_S21.hdf5',M_dust_frac=1e-4,Rin=0.0,Rout=2.0,static_dust=True,recentre_gas=True)
    #one2one_dust('Poly_N1e6_M5_R5_n15_S089.hdf5',M_dust_frac=1e-4,Rin=0.0,Rout=2.0,static_dust=True,recentre_gas=True)
    #write_hdf5(init_dict=Star_A_dict)
    #graft('Mstar1.hdf5','Poly_N1e6_M5_R5_n15_S089_dust.hdf5','Poly_N1e6_M5_dust_r30.hdf5',offset_pos=np.array([0.3,0,0]),offset_vK=True,COM_correction=True)
    #graft('Mstar1.hdf5','Poly_N1e6_M5_R5_n15_S089_dust.hdf5','Poly_N1e6_M5_dust_r40.hdf5',offset_pos=np.array([0.4,0,0]),offset_vK=True,COM_correction=True)

    #Dust Shell
    #one2one_dust('Poly_N1e6_M5_R5_n15_S089.hdf5',M_dust_frac=1e-2,Rin=0.03,Rout=0.05,static_dust=True,recentre_gas=True,name='shell35')
    #graft('Mstar1.hdf5','Poly_N1e6_M5_R5_n15_S089_dustshell35.hdf5','Poly_N1e6_M5_dustshell35_r30.hdf5',offset_pos=np.array([0.3,0,0]),offset_vK=True,COM_correction=True)
    #graft('Mstar1.hdf5','Poly_N1e6_M5_R5_n15_S089_dustshell35.hdf5','Poly_N1e6_M5_dustshell35_r40.hdf5',offset_pos=np.array([0.4,0,0]),offset_vK=True,COM_correction=True)
    #graft('Mstar1.hdf5','Poly_N1e6_M5_R5_n15_S089_dustshell35.hdf5','Poly_N1e6_M5_dustshell35_r50.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,COM_correction=True)
    #graft('Mstar1.hdf5','Poly_N1e6_M5_R5_n15_S089_dustshell35.hdf5','Poly_N1e6_M5_dustshell35_r60.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)

    #N1e5 Runs
    #one2one_dust('Poly_N1e5_M5_R5_n15_S100.hdf5',M_dust_frac=1e-2,Rin=0.03,Rout=0.05,static_dust=True,recentre_gas=True,name='shell35')
    #graft('Mstar1.hdf5','Poly_N1e5_M5_R5_n15_S100_dustshell35.hdf5','Poly_N1e5_M5_dustshell35_r50.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,COM_correction=True)
    #graft('Mstar1.hdf5','Poly_N1e5_M5_R5_n15_S100_dustshell35.hdf5','Poly_N1e5_M5_dustshell35_r60.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)
    #one2one_dust('Poly_N100000_M0005_R001_n15.hdf5',M_dust_frac=1e-2,Rin=0.006,Rout=0.01,static_dust=True,recentre_gas=True,name='shell')
    #one2one_dust('Poly_N100000_M0005_R003_n15.hdf5',M_dust_frac=1e-2,Rin=0.02,Rout=0.03,static_dust=True,recentre_gas=True,name='shell')
    #graft('Mstar1.hdf5','Poly_N100000_M0005_R003_n15_dustshell.hdf5','Poly_N1e5_M5_R003_dustshell_r60.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)
    #graft('Mstar1.hdf5','Poly_N100000_M0005_R001_n15_dustshell.hdf5','Poly_N1e5_M5_R001_dustshell_r60.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)

    #one2one_dust('Poly_N1e5_M5_R3_evap_rho1e12_T30_S010.hdf5',M_dust_frac=1e-2,Rin=0.02,Rout=0.03,static_dust=True,recentre_gas=True,name='shell')
    #one2one_dust('Poly_N1e5_M5_R3_evap_rho1e12_T50_S010.hdf5',M_dust_frac=1e-2,Rin=0.02,Rout=0.03,static_dust=True,recentre_gas=True,name='shell')
    #one2one_dust('Poly_N1e5_M5_R3_evap_rho2e12_T30_S010.hdf5',M_dust_frac=1e-2,Rin=0.02,Rout=0.03,static_dust=True,recentre_gas=True,name='shell')
    #one2one_dust('Poly_N1e5_M5_R3_evap_rho2e12_T50_S010.hdf5',M_dust_frac=1e-2,Rin=0.02,Rout=0.03,static_dust=True,recentre_gas=True,name='shell')
    #graft('Mstar1.hdf5','Poly_N1e5_M5_R3_evap_rho1e12_T30_S010_dustshell.hdf5','Poly_N1e5_M5_R3_rho1e12_T30_dust_r60.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)
    #graft('Mstar1.hdf5','Poly_N1e5_M5_R3_evap_rho1e12_T50_S010_dustshell.hdf5','Poly_N1e5_M5_R3_rho1e12_T50_dust_r60.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)
    #graft('Mstar1.hdf5','Poly_N1e5_M5_R3_evap_rho2e12_T30_S010_dustshell.hdf5','Poly_N1e5_M5_R3_rho2e12_T30_dust_r60.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)
    #graft('Mstar1.hdf5','Poly_N1e5_M5_R3_evap_rho2e12_T50_S010_dustshell.hdf5','Poly_N1e5_M5_R3_rho2e12_T50_dust_r60.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)
    #one2one_dust('Poly_N1e5_M5_R3_S050.hdf5',M_dust_frac=1e-2,Rin=0.02,Rout=0.03,static_dust=True,recentre_gas=True,name='shell')
    #graft('Mstar1.hdf5','Poly_N1e5_M5_R3_S050_dustshell.hdf5','Poly_N1e5_M5_R3_dust_r60.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)

    #write_hdf5(init_dict=disc_D_dict)
    #graft('Mstar1.hdf5','Poly_N100000_M0005_R003_n15_dustshell.hdf5','Poly_N1e5_M5_R003_dustshell_r50.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,COM_correction=True)
    #graft('Mstar1.hdf5','Poly_N100000_M0005_R003_n15_dustshell.hdf5','Poly_N1e5_M5_R003_dustshell_r40.hdf5',offset_pos=np.array([0.4,0,0]),offset_vK=True,COM_correction=True)
    #graft('Mstar1.hdf5','Poly_N100000_M0005_R003_n15_dustshell.hdf5','Poly_N1e5_M5_R003_dustshell_r30.hdf5',offset_pos=np.array([0.3,0,0]),offset_vK=True,COM_correction=True)


    #Why is shell off centre?
    #one2one_dust('Poly_N100000_M0005_R003_n15.hdf5',M_dust_frac=1e-2,Rin=0.02,Rout=0.03,static_dust=True,recentre_gas=True)
    #graft('Mstar1.hdf5','Poly_N100000_M0005_R003_n15_MD001.hdf5','Poly_N1e5_M5_R003_MD001_r60.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)
    #write_hdf5(init_dict=disc_E_dict)
    #graft('Disc_N1e5_R00120_M0005.hdf5','Poly_N100000_M0005_R003_n15_MD001.hdf5','Polydisc_N1e5_M5_R3_dust_r60.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)
    #graft('Disc_N1e5_M0005_R01200_aav1_beta1_T30_Ti34_S060.hdf5','Poly_N100000_M0005_R003_n15_MD001.hdf5','Polydisc_N1e5_M5_R3_dust_r60_b1_T30_Ti34.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)




    #==== Clump in disc runs ====#
    #write_hdf5(init_dict=disc_F_dict)
    #write_hdf5(init_dict=Poly_H_dict)
    #one2one_dust('Disc_N3e5_R00120_M0009_b5_S075.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.5,recentre_gas=True)
    #graft('Disc_N3e5_R00120_M0009_b5_S075_MD001.hdf5','Poly_N100000_M0003_R003_n15_S075.hdf5','Polydisc_N4e5_M3_R3_a1_r60_b5.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)

    #write_hdf5(init_dict=disc_G_dict)
    #write_hdf5(init_dict=Poly_I_dict)

    #one2one_dust('Disc_N3e5_R00120_M0009_b5_S075.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.5,recentre_gas=True,Zsupp=10)
    #one2one_dust('Disc_N3e5_R00120_M0009_b5_S075.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.5,recentre_gas=True,Zsupp=100)
    #one2one_dust('Disc_N3e5_R00120_M0009_b5_S075.hdf5',M_dust_frac=1e-1,Rin=0.2,Rout=1.5,recentre_gas=True,Zsupp=10)
    #graft('Disc_N3e5_R00120_M0009_b5_S075_MD001_Z100.hdf5','Poly_N100000_M0003_R003_n15_S075.hdf5','Polydisc_N4e5_M3_R3_r60_b5_MD001_Z100.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)
    #graft('Disc_N3e5_R00120_M0009_b5_S075_MD001_Z10.hdf5','Poly_N100000_M0003_R003_n15_S075.hdf5','Polydisc_N4e5_M3_R3_r60_b5_MD001_Z10.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)
    #graft('Disc_N3e5_R00120_M0009_b5_S075_MD01_Z10.hdf5','Poly_N100000_M0003_R003_n15_S075.hdf5','Polydisc_N4e5_M3_R3_r60_b5_MD01_Z10.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)

    #Higher res
    #one2one_dust('Disc_N12e5_R00120_M0009_b5_S075.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.5,recentre_gas=True,Zsupp=10)
    #graft('Disc_N12e5_R00120_M0009_b5_S075_MD001_Z10.hdf5','Poly_N400000_M0003_R003_n15_S075.hdf5','Polydisc_N16e5_M3_R3_r60_b5_MD001_Z10.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)

    #Higher mass disc
    #write_hdf5(init_dict=disc_H_dict)
    #write_hdf5(init_dict=Poly_F_dict)
    #one2one_dust('Disc_N15e5_R00120_M0075_b5_S080.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.5,recentre_gas=True,Zsupp=100)
    #one2one_dust('Disc_N15e5_R00120_M0075_b5_S080.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.5,recentre_gas=True,Zsupp=10)
    #one2one_dust('Disc_N15e5_R00120_M0075_b5_S080.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.5,recentre_gas=True,Zsupp=1)
    #graft('Disc_N15e5_R00120_M0075_b5_S080_MD001_Z100.hdf5','Poly_N100000_M0005_R003_n15_S080.hdf5','Polydisc_N16e5_M5_R3_r100_b5_MD001_Z100.hdf5',offset_pos=np.array([1.0,0,0]),offset_vK=True,COM_correction=True)
    #graft('Disc_N15e5_R00120_M0075_b5_S080_MD001_Z10.hdf5','Poly_N100000_M0005_R003_n15_S080.hdf5','Polydisc_N16e5_M5_R3_r100_b5_MD001_Z10.hdf5',offset_pos=np.array([1.0,0,0]),offset_vK=True,COM_correction=True)
    #graft('Disc_N15e5_R00120_M0075_b5_S080_MD001_Z1.hdf5','Poly_N100000_M0005_R003_n15_S080.hdf5','Polydisc_N16e5_M5_R3_r100_b5_MD001_Z1.hdf5',offset_pos=np.array([1.0,0,0]),offset_vK=True,COM_correction=True)
    #write_hdf5(init_dict=disc_I_dict)
    

    #Polytrope disruption tests
    #graft('Mstar1.hdf5','Poly_N100000_M0005_R003_n15_S080.hdf5','Poly_N1e5_M5_r30.hdf5',offset_pos=np.array([0.3,0,0]),offset_vK=True,COM_correction=True)
    #graft('Mstar1.hdf5','Poly_N100000_M0005_R003_n15_S080.hdf5','Poly_N1e5_M5_r40.hdf5',offset_pos=np.array([0.4,0,0]),offset_vK=True,COM_correction=True)
    #graft('Mstar1.hdf5','Poly_N100000_M0005_R003_n15_S080.hdf5','Poly_N1e5_M5_r50.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,COM_correction=True)
    #graft('Mstar1.hdf5','Poly_N100000_M0005_R003_n15_S080.hdf5','Poly_N1e5_M5_r60.hdf5',offset_pos=np.array([0.6,0,0]),offset_vK=True,COM_correction=True)

    #one2one_dust('Disc_N15e5_R00110_M0075_b5_S060.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,recentre_gas=True,Zsupp=10)
    #graft('Disc_N15e5_R00110_M0075_b5_S060_MD001_Z10.hdf5','Poly_N100000_M0005_R003_n15_S080.hdf5','Polydisc_N16e5_M5_R3_r50_b5_MD001_Z10.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,COM_correction=True)


    #CDS3 runs
    #write_hdf5(init_dict=Poly_J_dict)
    #write_hdf5(init_dict=disc_J_dict)
    #one2one_dust('Disc_N12e6_R00110_M0075_S038.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,recentre_gas=True,Zsupp=10)
    #graft('Disc_N12e6_R00110_M0075_S038_MD001_Z10.hdf5','Poly_N800000_M0005_R003_n25_S129.hdf5','Polydisc_N13e6_M5_R3_r50_b5_MD001_Z10.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,COM_correction=True)
    #graft('Disc_N12e6_R00110_M0075_S038.hdf5','Poly_N800000_M0005_R003_n25_S129.hdf5','Polydisc_N13e6_M5_R3_r50_b5.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,COM_correction=True)
    #write_hdf5(init_dict=Poly_K_dict)
    #one2one_dust('Disc_N15e5_R00110_M0075_S070.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,recentre_gas=True,Zsupp=10)
    #graft('Disc_N15e5_R00110_M0075_S070_MD001_Z10.hdf5','Poly_N100000_M0005_R003_n25_S070.hdf5','Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,COM_correction=True)
    #graft('Disc_N15e5_R00110_M0075_S070.hdf5','Poly_N100000_M0005_R003_n25_S070.hdf5','Polydisc_N16e5_M5_R3_r50_b5_g75.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,COM_correction=True)
    #write_hdf5(init_dict=disc_L_dict)

    #one2one_dust('Disc_N15e5_R00110_M0075_S070.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,recentre_gas=True,Zsupp=10,N_dust_frac=0.1)
    #graft('Disc_N15e5_R00110_M0075_S070_MD001_Z10_df01.hdf5','Poly_N100000_M0005_R003_n25_S070.hdf5','Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_df01.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,COM_correction=True)
    #write_hdf5(init_dict=Poly_L_dict)
    #write_hdf5(init_dict=Poly_M_dict)
    #graft('Disc_N15e5_R00110_M0075_S070.hdf5','Poly_N100000_M0005_R003_n25_S070.hdf5','Polydisc_N16e5_M5_R3_r75_b5_g75.hdf5',offset_pos=np.array([0.75,0,0]),offset_vK=True,COM_correction=True)

    #one2one_dust('Polydisc_N16e5_M5_R3_r50_b5_g75_S032.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,recentre_gas=True,Zsupp=10,N_dust_frac=0.1)
    #one2one_dust('Polydisc_N16e5_M5_R3_r75_b5_g75_S032.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,recentre_gas=True,Zsupp=10,N_dust_frac=0.1)
    #insert_sink_planet('Polydisc_N16e5_M5_R3_r50_b5_g75_S032_MD001_Z10_df01.hdf5',3e-6,rhomax=True)
    #insert_sink_planet('Polydisc_N16e5_M5_R3_r75_b5_g75_S032_MD001_Z10_df01.hdf5',3e-6,rhomax=True)
    #graft('Disc_N15e5_R00110_M0075_S070.hdf5','Poly_N20000_M0001_R001_n25_S070.hdf5','Polydisc_N16e5_M1_R1_r50_b5_g75.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,COM_correction=True)
    #graft('Disc_N15e5_R00110_M0075_S070.hdf5','Poly_N20000_M0001_R001_n25_S070.hdf5','Polydisc_N16e5_M1_R1_r75_b5_g75.hdf5',offset_pos=np.array([0.75,0,0]),offset_vK=True,COM_correction=True)
    #graft('Disc_N15e5_R00110_M0075_S070.hdf5','Poly_N60000_M0003_R002_n25_S070.hdf5','Polydisc_N16e5_M3_R2_r50_b5_g75.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,COM_correction=True)
    #graft('Disc_N15e5_R00110_M0075_S070.hdf5','Poly_N60000_M0003_R002_n25_S070.hdf5','Polydisc_N16e5_M3_R2_r75_b5_g75.hdf5',offset_pos=np.array([0.75,0,0]),offset_vK=True,COM_correction=True)

    
    #one2one_dust('Polydisc_N16e5_M5_R3_r50_b5_g75.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,Zsupp=10,N_dust_frac=0.1)
    #one2one_dust('Polydisc_N16e5_M5_R3_r75_b5_g75.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,Zsupp=10,N_dust_frac=0.1)
    #one2one_dust('Polydisc_N16e5_M3_R2_r50_b5_g75.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,Zsupp=10,N_dust_frac=0.1)
    #one2one_dust('Polydisc_N16e5_M3_R2_r75_b5_g75.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,Zsupp=10,N_dust_frac=0.1)
    #one2one_dust('Polydisc_N16e5_M1_R1_r50_b5_g75.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,Zsupp=10,N_dust_frac=0.1)
    #one2one_dust('Polydisc_N16e5_M1_R1_r75_b5_g75.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,Zsupp=10,N_dust_frac=0.1)

    #insert_sink_planet('Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_df01.hdf5',3e-6,X=0.5,Y=0,Z=0)
    #insert_sink_planet('Polydisc_N16e5_M5_R3_r75_b5_g75_MD001_Z10_df01.hdf5',3e-6,X=0.75,Y=0,Z=0)
    #insert_sink_planet('Polydisc_N16e5_M3_R2_r50_b5_g75_MD001_Z10_df01.hdf5',3e-6,X=0.5,Y=0,Z=0)
    #insert_sink_planet('Polydisc_N16e5_M3_R2_r75_b5_g75_MD001_Z10_df01.hdf5',3e-6,X=0.75,Y=0,Z=0)
    #insert_sink_planet('Polydisc_N16e5_M1_R1_r50_b5_g75_MD001_Z10_df01.hdf5',3e-6,X=0.5,Y=0,Z=0)
    #insert_sink_planet('Polydisc_N16e5_M1_R1_r75_b5_g75_MD001_Z10_df01.hdf5',3e-6,X=0.75,Y=0,Z=0)

    #Z height fix?
    #graft('Disc_N15e5_R00110_M0075_S070.hdf5','Poly_N100000_M0005_R003_n25_S070.hdf5','Pd_N16e5_M5_R3_r50_b5_g75.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,rho_correction=True)
    #one2one_dust('Pd_N16e5_M5_R3_r50_b5_g75.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,Zsupp=10,N_dust_frac=0.1)
    #insert_sink_planet('Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_df01.hdf5',3e-6,X=0.5,Y=0,Z=0)
    #graft('Disc_N15e5_R00110_M0075_S070.hdf5','Poly_N60000_M0003_R002_n25_S070.hdf5','Pd_N16e5_M3_R2_r50_b5_g75.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,rho_correction=True)
    #one2one_dust('Pd_N16e5_M3_R2_r50_b5_g75.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,Zsupp=10,N_dust_frac=0.1)
    #insert_sink_planet('Pd_N16e5_M3_R2_r50_b5_g75_MD001_Z10_df01.hdf5',3e-6,X=0.5,Y=0,Z=0)
    #graft('Disc_N15e5_R00110_M0075_S070.hdf5','Poly_N20000_M0001_R001_n25_S070.hdf5','Pd_N16e5_M1_R1_r50_b5_g75.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,rho_correction=True)
    #one2one_dust('Pd_N16e5_M1_R1_r50_b5_g75.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,Zsupp=10,N_dust_frac=0.1)
    #insert_sink_planet('Pd_N16e5_M1_R1_r50_b5_g75_MD001_Z10_df01.hdf5',3e-6,X=0.5,Y=0,Z=0)

    #Paper runs: no dust core
    #one2one_dust('Disc_N15e5_R00110_M0075_S070.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,Zsupp=10,N_dust_frac=0.3)
    #graft('Disc_N15e5_R00110_M0075_S070_MD001_Z10_df03.hdf5','Poly_N100000_M0005_R003_n25_S070.hdf5','Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_df03.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,rho_correction=True)
    #insert_sink_planet('Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_df03.hdf5',3e-6,X=0.5,Y=0,Z=0)


    
    #Dust core
    #graft('Disc_N15e5_R00110_M0075_S070.hdf5','Poly_N100000_M0005_R003_n25_S070.hdf5','Pod_N16e5_M5_R3_r50_b5_g75.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,rho_correction=True)
    #one2one_dust('Pod_N16e5_M5_R3_r50_b5_g75.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,Zsupp=10,N_dust_frac=0.3)
    #insert_sink_planet('Pod_N16e5_M5_R3_r50_b5_g75_MD001_Z10_df03.hdf5',3e-6,X=0.5,Y=0,Z=0)
    #graft('Disc_N15e5_R00110_M0075_S070.hdf5','Poly_N100000_M0005_R003_n25_S070.hdf5','Pd_N16e5_M5_R3_r50_b5_g75.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,rho_correction=True)
    
    #Paper runs: no dust core. Hi res
    #one2one_dust('Disc_N15e5_R00110_M0075_S070.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,Zsupp=10,N_dust_frac=1.)
    #graft('Disc_N15e5_R00110_M0075_S070_MD001_Z10.hdf5','Poly_N100000_M0005_R003_n25_S070.hdf5','Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,rho_correction=True)
    #insert_sink_planet('Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10.hdf5',3e-6,X=0.5,Y=0,Z=0)

    #Lower mass planets
    #graft('Disc_N15e5_R00110_M0075_S070_MD001_Z10_df03.hdf5','Poly_N60000_M0003_R002_n25_S070.hdf5','Pd_N16e5_M3_R2_r50_b5_g75_MD001_Z10_df03.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,rho_correction=True)
    #insert_sink_planet('Pd_N16e5_M3_R2_r50_b5_g75_MD001_Z10_df03.hdf5',3e-6,X=0.5,Y=0,Z=0)
    #graft('Disc_N15e5_R00110_M0075_S070_MD001_Z10_df03.hdf5','Poly_N20000_M0001_R001_n25_S070.hdf5','Pd_N16e5_M1_R1_r50_b5_g75_MD001_Z10_df03.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,rho_correction=True)
    #insert_sink_planet('Pd_N16e5_M1_R1_r50_b5_g75_MD001_Z10_df03.hdf5',3e-6,X=0.5,Y=0,Z=0)

    #High res runs
    #frag_pos = np.array([0.5,0,0])
    #cut_sphere('Disc_N12e6_R00110_M0075_g75_S026.hdf5',cut_pos=np.copy(frag_pos),cut_R=0.015)
    #one2one_dust('Disc_N12e6_R00110_M0075_g75_S026_cut.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,Zsupp=10,N_dust_frac=0.2)
    #graft('Disc_N12e6_R00110_M0075_g75_S026_cut_MD001_Z10_df02.hdf5',
    #      'Poly_N800000_M0005_R003_n25_S129.hdf5',
    #      'Pd_N14e6_M5_R3_r50_b5_g75_MD001_Z10.hdf5',offset_pos=np.copy(frag_pos),offset_vK=True,rho_correction=True)
    #insert_sink_planet('Pd_N14e6_M5_R3_r50_b5_g75_MD001_Z10.hdf5',3e-6,pos_P=frag_pos)    
    #load_test('graft_dir/Pd_N14e6_M5_R3_r50_b5_g75_MD001_Z10_MP3e-06.hdf5')

    #Redo paper runs
    #frag_pos = np.array([0.5,0,0])    
    #load_test('graft_dir/'+'Disc_N15e5_R00110_M0075_g75_S070.hdf5')
    #cut_sphere('Disc_N15e5_R00110_M0075_g75_S070.hdf5',cut_pos=np.copy(frag_pos),cut_R=0.015)
    #one2one_dust('Disc_N15e5_R00110_M0075_g75_S070_cut.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,Zsupp=10,N_dust_frac=1.)
    #graft('Disc_N15e5_R00110_M0075_g75_S070_cut_MD001_Z10.hdf5','Poly_N100000_M0005_R003_n25_S070.hdf5','Pd2_N16e5_M5_R3_r50_b5_g75_MD001_Z10.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,rho_correction=True)
    #insert_sink_planet('Pd2_N16e5_M5_R3_r50_b5_g75_MD001_Z10.hdf5',3e-6,pos_P=frag_pos)


    #Feedback paper runs 2019
    #write_hdf5(init_dict=disc_M_dict)
    #write_hdf5(init_dict=disc_N_dict)
    #frag_pos = np.array([0.5,0,0])
    #cut_sphere('Disc_N2e6_R00110_M01_g75_S068.hdf5',cut_pos=np.copy(frag_pos),cut_R=0.015)
    #one2one_dust('Disc_N2e6_R00110_M01_g75_S068_cut.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,Zsupp=10,N_dust_frac=0.3)
    #graft('Disc_N2e6_R00110_M01_g75_S068_cut_MD001_Z10_df03.hdf5','Poly_N100000_M0005_R003_n25_S070.hdf5','Pd3_N2e6_M5_R3_r50_b5_g75_MD001_df03.hdf5',offset_pos=frag_pos,offset_vK=True,rho_correction=True)
    #insert_sink_planet('Pd3_N2e6_M5_R3_r50_b5_g75_MD001_df03.hdf5',3e-6,pos_P=frag_pos)

    #2019 Hires
    #cut_sphere('Disc_N16e6_R00110_M01_g75_S068.hdf5',cut_pos=np.copy(frag_pos),cut_R=0.015)
    #one2one_dust('Disc_N16e6_R00110_M01_g75_S068_cut.hdf5',M_dust_frac=1e-2,Rin=0.2,Rout=1.0,Zsupp=10,N_dust_frac=0.3)
    #graft('Disc_N16e6_R00110_M01_g75_S068_cut_MD001_Z10_df03.hdf5','Poly_N800000_M0005_R003_n25_S129.hdf5','Pd3_N16e6_M5_R3_r50_b5_g75_MD001_df03.hdf5',offset_pos=frag_pos,offset_vK=True,rho_correction=True)
    #insert_sink_planet('Pd3_N16e6_M5_R3_r50_b5_g75_MD001_df03.hdf5',3e-6,pos_P=frag_pos)


    #Migration comp paper
    #graft('Disc_N15e5_R00110_M0075_g75_S070.hdf5','Poly_N20000_M0001_R001_n25_S070.hdf5','Pd_N16e5_M1_R1_r50_b5_g75.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,rho_correction=True)
    #graft('Disc_N15e5_R00110_M0075_g75_S070.hdf5','Poly_N60000_M0003_R002_n25_S070.hdf5','Pd_N16e5_M3_R2_r50_b5_g75.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,rho_correction=True)
    #graft('Disc_N15e5_R00110_M0075_g75_S070.hdf5','Poly_N100000_M0005_R003_n25_S070.hdf5','Pd_N16e5_M5_R3_r50_b5_g75.hdf5',offset_pos=np.array([0.5,0,0]),offset_vK=True,rho_correction=True)

    #insert_sink_planet('Disc_N15e5_R00110_M0075_g75_S070.hdf5',1.0,pos_P=np.array([0.5,0,0]))
    #insert_sink_planet('Disc_N15e5_R00110_M0075_g75_S070.hdf5',3.0,pos_P=np.array([0.5,0,0]))
    #insert_sink_planet('Disc_N15e5_R00110_M0075_g75_S070.hdf5',5.0,pos_P=np.array([0.5,0,0]))
    

    #Observing remnant fragments
    #frag_pos = np.array([0.5,0,0])
    frag_pos = np.array([0.4,0,0])
    frag_pos2 = np.array([1.,0,0])
    #write_hdf5(init_dict=disc_P_dict)
    #graft('Disc_N2e5_R530_M001_g53.hdf5','Poly_N100000_M0005_R003_n25_S070.hdf5','HP_N3e5_M5_R3_r50_b5_g75.hdf5',offset_pos=frag_pos,offset_vK=True,rho_correction=True)
    #one2one_dust('HP_N3e5_M5_R3_r50_b5_g75.hdf5',M_dust_frac=1e-2,Rin=0.1,Rout=1.2,Zsupp=1,N_dust_frac=1.0)

    #write_hdf5(init_dict=Poly_R_dict)
    #write_hdf5(init_dict=Poly_S_dict)
    #write_hdf5(init_dict=Poly_T_dict)
    #write_hdf5(init_dict=disc_Q_dict)
    
    #cut_sphere('Poly_N1e5_M1_R1_n25_T30_S200.hdf5',np.array([0,0,0]),cut_R=1000,cut_Rin=0.011,centred=True) #Cut out random dust around 1MJ PP 

    #graft('Disc_N1e5_R530_M0001_g53.hdf5','Poly_N1e5_M1_R1_n25_T20_S200.hdf5','HP_N2e5_M1_R1_r100_b5_g75_T20.hdf5',offset_pos=frag_pos2,offset_vK=True,rho_correction=True)
    #graft('Disc_N1e5_R530_M0001_g53.hdf5','Poly_N1e5_M1_R1_n25_T30_S200_cut.hdf5','HP_N2e5_M1_R1_r40_b5_g75_T30.hdf5',offset_pos=frag_pos,offset_vK=True,rho_correction=True)
    #graft('Disc_N1e5_R530_M0001_g53.hdf5','Poly_N3e5_M3_R2_n25_T20_S200.hdf5','HP_N4e5_M3_R2_r100_b5_g75_T20.hdf5',offset_pos=frag_pos2,offset_vK=True,rho_correction=True)
    #graft('Disc_N1e5_R530_M0001_g53.hdf5','Poly_N3e5_M3_R2_n25_T30_S200.hdf5','HP_N4e5_M3_R2_r40_b5_g75_T30.hdf5',offset_pos=frag_pos,offset_vK=True,rho_correction=True)

    #one2one_dust('HP_N2e5_M1_R1_r100_b5_g75_T20.hdf5',M_dust_frac=1e-2,Rin=0.1,Rout=0.3,Zsupp=10,N_dust_frac=1.0)
    #one2one_dust('HP_N2e5_M1_R1_r40_b5_g75_T30.hdf5',M_dust_frac=1e-2,Rin=0.1,Rout=0.3,Zsupp=10,N_dust_frac=1.0)
    #one2one_dust('HP_N4e5_M3_R2_r100_b5_g75_T20.hdf5',M_dust_frac=1e-2,Rin=0.1,Rout=0.3,Zsupp=10,N_dust_frac=1.0)
    #one2one_dust('HP_N4e5_M3_R2_r40_b5_g75_T30.hdf5',M_dust_frac=1e-2,Rin=0.1,Rout=0.3,Zsupp=10,N_dust_frac=1.0)

    #one2one_dust('HP_N2e5_M1_R1_r100_b5_g75_T20_MD001_Z10.hdf5',M_dust_frac=1e-2,Rin=0.35,Rout=1.2,Zsupp=1,N_dust_frac=1.0)
    #one2one_dust('HP_N2e5_M1_R1_r40_b5_g75_T30_MD001_Z10.hdf5',M_dust_frac=1e-2,Rin=0.35,Rout=1.2,Zsupp=1,N_dust_frac=1.0)
    #one2one_dust('HP_N4e5_M3_R2_r100_b5_g75_T20_MD001_Z10.hdf5',M_dust_frac=1e-2,Rin=0.35,Rout=1.2,Zsupp=1,N_dust_frac=1.0)
    #one2one_dust('HP_N4e5_M3_R2_r40_b5_g75_T30_MD001_Z10.hdf5',M_dust_frac=1e-2,Rin=0.35,Rout=1.2,Zsupp=1,N_dust_frac=1.0)

    #Heavy discs
    #write_hdf5(init_dict=disc_R_dict)
    #graft('Disc_N1e6_R530_M001_g53.hdf5','Poly_N1e5_M1_R1_n25_T20_S200.hdf5','HP_N11e5_M1_R1_r100_b5_g75_T20.hdf5',offset_pos=frag_pos2,offset_vK=True,rho_correction=True)
    #graft('Disc_N1e6_R530_M001_g53.hdf5','Poly_N1e5_M1_R1_n25_T30_S200.hdf5','HP_N11e5_M1_R1_r40_b5_g75_T30.hdf5',offset_pos=frag_pos,offset_vK=True,rho_correction=True)
    #graft('Disc_N1e6_R530_M001_g53.hdf5','Poly_N3e5_M3_R2_n25_T20_S200.hdf5','HP_N13e5_M3_R2_r100_b5_g75_T20.hdf5',offset_pos=frag_pos2,offset_vK=True,rho_correction=True)
    #graft('Disc_N1e6_R530_M001_g53.hdf5','Poly_N3e5_M3_R2_n25_T30_S200.hdf5','HP_N13e5_M3_R2_r40_b5_g75_T30.hdf5',offset_pos=frag_pos,offset_vK=True,rho_correction=True)
    #one2one_dust('HP_N11e5_M1_R1_r100_b5_g75_T20.hdf5',M_dust_frac=1e-2,Rin=0.1,Rout=0.3,Zsupp=10,N_dust_frac=1.0)
    #one2one_dust('HP_N11e5_M1_R1_r40_b5_g75_T30.hdf5',M_dust_frac=1e-2,Rin=0.1,Rout=0.3,Zsupp=10,N_dust_frac=1.0)
    #one2one_dust('HP_N13e5_M3_R2_r100_b5_g75_T20.hdf5',M_dust_frac=1e-2,Rin=0.1,Rout=0.3,Zsupp=10,N_dust_frac=1.0)
    #one2one_dust('HP_N13e5_M3_R2_r40_b5_g75_T30.hdf5',M_dust_frac=1e-2,Rin=0.1,Rout=0.3,Zsupp=10,N_dust_frac=1.0)

    #one2one_dust('HP_N11e5_M1_R1_r100_b5_g75_T20_MD001_Z10.hdf5',M_dust_frac=1e-2,Rin=0.4,Rout=1.2,Zsupp=1,N_dust_frac=1.0)
    #one2one_dust('HP_N11e5_M1_R1_r40_b5_g75_T30_MD001_Z10.hdf5',M_dust_frac=1e-2,Rin=0.4,Rout=1.2,Zsupp=1,N_dust_frac=1.0)
    #one2one_dust('HP_N13e5_M3_R2_r100_b5_g75_T20_MD001_Z10.hdf5',M_dust_frac=1e-2,Rin=0.4,Rout=1.2,Zsupp=1,N_dust_frac=1.0)
    #one2one_dust('HP_N13e5_M3_R2_r40_b5_g75_T30_MD001_Z10.hdf5',M_dust_frac=1e-2,Rin=0.4,Rout=1.2,Zsupp=1,N_dust_frac=1.0)

    #Smaller PP
    graft('Disc_N1e5_R530_M0001_g53.hdf5','Poly_N1e5_M1_R05_n25_T20.hdf5','HP_N2e5_M1_R05_r100_b5_g75_T20.hdf5',offset_pos=frag_pos2,offset_vK=True,rho_correction=True)
    graft('Disc_N1e5_R530_M0001_g53.hdf5','Poly_N1e5_M1_R05_n25_T20.hdf5','HP_N2e5_M1_R05_r40_b5_g75_T30.hdf5',offset_pos=frag_pos,offset_vK=True,rho_correction=True)
    one2one_dust('HP_N2e5_M1_R05_r100_b5_g75_T20.hdf5',M_dust_frac=1e-2,Rin=0.1,Rout=0.3,Zsupp=10,N_dust_frac=1.0)
    one2one_dust('HP_N2e5_M1_R05_r40_b5_g75_T30.hdf5',M_dust_frac=1e-2,Rin=0.1,Rout=0.3,Zsupp=10,N_dust_frac=1.0)
    one2one_dust('HP_N2e5_M1_R05_r100_b5_g75_T20_MD001_Z10.hdf5',M_dust_frac=1e-2,Rin=0.35,Rout=1.2,Zsupp=1,N_dust_frac=1.0)
    one2one_dust('HP_N2e5_M1_R05_r40_b5_g75_T30_MD001_Z10.hdf5',M_dust_frac=1e-2,Rin=0.35,Rout=1.2,Zsupp=1,N_dust_frac=1.0)

    

    
    #load_test('graft_dir/','Disc_N100000_M001_dust.hdf5')
    #load_test('','Disc_N2000000_M001.hdf5')
    #load_test('graft_dir/','Disc_N100000_M001_relaxed_dust.hdf5')
    #load_test('graft_dir/','Disc_N100000_Poly_dust.hdf5')
    #load_test('graft_dir/','Disc_N100000_Poly.hdf5')
    #load_test('graft_dir/','Disc_N100000_M001_relaxed_sink_dust.hdf5')
    #load_test('graft_dir/','O_M2_beta10_a1_n2e6.hdf5')
    #load_test('graft_dir/','O_M2_beta10_a1_n2e6_dust.hdf5')
    #load_test('/rfs/TAG/rjh73/','O_M2_beta10_a1_n2e6_dust_INIT.hdf5')
    #load_test('/rfs/TAG/rjh73/O_M2_beta10_a1_n2e6_dust/','snapshot_000.hdf5')


    #TW Hya project
    #write_hdf5(init_dict=TW_Hya)
    #add_dust_profile('TW_Hya_.hdf5',dust_file='dump0050_52AU')
    
    #write_hdf5(init_dict=TW_Hya2)
    #graft('TW_Hya2.hdf5','Poly_N1e5_M1_R1_n25_T30_S200.hdf5','TW_Hya2_PP1MJ52AU.hdf5',offset_pos=np.array([0.52,0,0]),offset_vK=True,rho_correction=True)

    #one2one_dust('Poly_N1e5_M1_R1_n25_T20_S200.hdf5',M_dust_frac=1e-1,Rin=0,Rout=0.02,Zsupp=1,N_dust_frac=1.0,static_dust=True)
    #write_hdf5(init_dict=TW_Hya2)
    #add_dust_profile('TW_Hya2.hdf5',dust_file='dump0050_52AU')
    #graft('TW_Hya2dump0050_52AU.hdf5','Poly_N1e5_M1_R1_n25_T20_S200_MD01_Z1.hdf5','TW_Hya2_PP1MJ52AU.hdf5',offset_pos=np.array([0.52,0,0]),offset_vK=True,rho_correction=True)
    #load_test('graft_dir/TW_Hya2_PP1MJ52AU.hdf5')

    #write_hdf5(init_dict=Poly_U_dict)
    
    
    plt.show()
