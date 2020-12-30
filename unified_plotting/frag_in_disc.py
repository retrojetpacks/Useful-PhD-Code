'''Investigating disc fragments'''

from __future__ import division
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
import basic_astro_functions as b
import astrophysical_constants_cgs as c
import os
from pygadgetreader import *
import glob
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

#Font styles, sizes and weights
def set_rcparams(fsize=10):
    font = {'weight' : 'normal',
            'size'   : fsize,
            'serif' : 'Minion Pro'}
    plt.rc('font', **font)
    plt.rc('legend', fontsize=fsize)
    plt.rc('lines',linewidth = 1)
    
    return
set_rcparams()


#Code Units
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
#run_cols = ['#cd6622','#ffc82e','#440044','#0055ff','#666666']
run_cols = ['#2EC4C6','#CD6622','#440044','#FFC82E','#FF1493','#6a5acd','#cd6622','#ffc82e','#0055ff']
plot_cols = ['#2EC4C6','#FFC82E','#CD6622','#FF1493','#440044','#6a5acd','#33cc33','#0055ff','#dd44cc','#cd6622','#A8CFFF','#62D0FF']
pastels  = colors.LinearSegmentedColormap.from_list('pastels',['#FF8E8E','#FFF9CE','#A8CFFF'])
#pastels  = colors.LinearSegmentedColormap.from_list('pastels',['#FF7575','#FFF284','#FFFFFF','#BBDAFF','#62D0FF'])


filepath = '/rfs/TAG/rjh73/Frag_in_disc/'
#filepath = '/rfs/TAG/rjh73/Poly_rad/'

'''
runfolders = [#'Poly_N100000_M0003_R003_n15/',
              #'Disc_N3e5_R00120_M0009_b5/',
    #'Polydisc_N4e5_M3_R3_a1_r60_b5_np8/', #Z1 MD001
    #'Polydisc_N4e5_M3_R3_r60_b5_a1_MD001_Z100/',
    #'Polydisc_N4e5_M3_R3_r60_b5_a1_MD001_Z10/',
    #'Polydisc_N4e5_M3_R3_r60_b5_a01_MD001_Z10/',
    #'Polydisc_N4e5_M3_R3_r60_b5_a10_MD001_Z10/',
    #'Polydisc_N4e5_M3_R3_r60_b5_a1_MD01_Z10/'
    #'post_vfrag/Polydisc_N4e5_M3_R3_r60_b5_a1_MD001_Z10_s1e4/',
    #'post_vfrag/Polydisc_N4e5_M3_R3_r60_b5_a1_MD001_Z10_s5e4/',
    #'post_vfrag/Polydisc_N4e5_M3_R3_r60_b5_a1_MD001_Z10/'
]
''''''
runfolders = [#'Disc_N15e5_R00120_M0075_b5/',
              #'Poly_N100000_M0005_R003_n15/',
    #'Polydisc_N16e5_M5_R3_r100_b5_MD001_Z10_a001/',
    #'Polydisc_N16e5_M5_R3_r100_b5_MD001_Z10_a01/',
    #'Polydisc_N16e5_M5_R3_r100_b5_MD001_Z10_a1/',
    #'Polydisc_N16e5_M5_R3_r100_b5_MD001_Z10_a10/',
    #'Polydisc_N16e5_M5_R3_r100_b5_MD001_Z10_a001_1/',
    #'Polydisc_N16e5_M5_R3_r100_b5_MD001_Z10_a001_1_sink1e11/'
    #'Polydisc_N16e5_M5_R3_r100_b5_MD001_Z10_a10_vf10/',
    #'Polydisc_N16e5_M5_R3_r100_b5_MD001_Z10_a1_vf10/',
    'Disc_N15e5_R00110_M0075_b5_quick/'
]
''''''
runfolders = ['Poly_N1e5_M5_r30/',
              'Poly_N1e5_M5_r40/',
              'Poly_N1e5_M5_r50/',
              'Poly_N1e5_M5_r60/',
]'''


runfolders = [#'Polydisc_N16e5_M5_R3_r50_b5_MD001_Z10_a01/',
              #'Polydisc_N16e5_M5_R3_r50_b5_MD001_Z10_a1/',
              #'Polydisc_N16e5_M5_R3_r50_b5_MD001_Z10_a10/',
              #'Polydisc_N16e5_M5_R3_r50_b5_MD001_Z10_a001_1/',
    #'Polydisc_N16e5_M5_R3_r50_b5_MD001_Z10_a1_sink1e10/',
    #'Polydisc_N16e5_M5_R3_r50_b5_MD001_Z10_a1_sink1e11/',
    #'Polydisc_N16e5_M5_R3_r50_b5_MD001_Z10_a1_sink1e12/',
    #'Polydisc_N16e5_M5_R3_r50_b5_MD001_Z10_a1_sink1e10_res/',
    #'Polydisc_N16e5_M5_R3_r50_b5_MD001_Z10_a1_sink1e11_res/',
    #Polydisc_N16e5_M5_R3_r50_b5_MD001_Z10_a1_sink1e12_res/'
    #'Polydisc_N13e6_M5_R3_r50_b5_MD001_Z10_a1/',
    #'Polydisc_N13e6_M5_R3_r50_b5/',
]

runfolders = [#'Poly_N800000_M0005_R003_n25/',
              #'Poly_N100000_M0005_R003_n25/',
              #'Disc_N12e6_R00110_M0075/',
              #'Disc_N15e5_R00110_M0075/',
              #'Disc_N15e5_R00110_M0075_g75/',
    #'Polydisc_N13e6_M5_R3_r50_b5_MD001_Z10_a1_Rin5/',
    #'Polydisc_N13e6_M5_R3_r50_b5_MD001_Z10_a1/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1/',
    #'Polydisc_N16e5_M5_R3_r50_b5_MD001_Z10_a1/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_soft2e3/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a01_soft2e3/',
    
    'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a01_df01/',
    'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01/',
    'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df01/',
    
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD01_Z10_a01_df01/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD01_Z10_a1_df01/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD01_Z10_a10_df01/',
    
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01_sink1e11/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01_sink1e10/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01_sink1e9/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01_sink1e11_lowres/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a01_df01_sink1e10/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a01_df01_sink1e11/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df01_sink1e10/',

    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01_sink1e10_lowres/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01_sink1e10_hires/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01_sink1e10_shires/',
    ]             

'''
runfolders = [#'Poly_N20000_M0001_R001_n25/',
              #'Poly_N60000_M0003_R002_n25/',
              #'Poly_N100000_M0005_R003_n25/',
              'Polydisc_N16e5_M5_R3_r50_b5_g75/',
              'Polydisc_N16e5_M5_R3_r75_b5_g75/',
    #'Polydisc_N16e5_M3_R2_r50_b5_g75/',
    #'Polydisc_N16e5_M3_R2_r75_b5_g75/',
    'Polydisc_N16e5_M1_R1_r50_b5_g75/',
    'Polydisc_N16e5_M1_R1_r75_b5_g75/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01_sink1e10_shires/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01_MP3e-06/',
    #'Polydisc_N16e5_M5_R3_r75_b5_g75_MD001_Z10_a1_df01_MP3e-06/',
    #'Polydisc_N16e5_M3_R2_r50_b5_g75_MD001_Z10_a1_df01_MP3e-06/',
    #'Polydisc_N16e5_M3_R2_r75_b5_g75_MD001_Z10_a1_df01_MP3e-06/',
    #'Polydisc_N16e5_M1_R1_r50_b5_g75_MD001_Z10_a1_df01_MP3e-06/',
    #'Polydisc_N16e5_M1_R1_r75_b5_g75_MD001_Z10_a1_df01_MP3e-06/',
]'''


runfolders = [#'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01_MP3e-06_2/',
              #'Polydisc_N16e5_M5_R3_r75_b5_g75_MD001_Z10_a1_df01_MP3e-06_2/',
    #'Polydisc_N16e5_M3_R2_r50_b5_g75_MD001_Z10_a1_df01_MP3e-06_2/',
    #'Polydisc_N16e5_M3_R2_r75_b5_g75_MD001_Z10_a1_df01_MP3e-06_2/',
    #'Polydisc_N16e5_M1_R1_r50_b5_g75_MD001_Z10_a1_df01_MP3e-06_2/',
    #'Polydisc_N16e5_M1_R1_r75_b5_g75_MD001_Z10_a1_df01_MP3e-06_2/',
    #'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a01_df01_MP3e-06/',
    #'Polydisc_N16e5_M5_R3_r75_b5_g75_MD001_Z10_a01_df01_MP3e-06/',
    'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df01_MP3e-06/',
    #'Polydisc_N16e5_M5_R3_r75_b5_g75_MD001_Z10_a10_df01_MP3e-06/',
    #'Polydisc_N16e5_M5_R3_r50_b5_MD001_Z10_a1/'
    #'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01_MP3e-06/',
    #'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01_MP3e-06_0718/',

    'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df01_MP3e-06_bf01/',
    'Polydisc_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df01_MP3e-06_bf1/'
]

runfolders = [#'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a01_df01_MP3e-06/',
              #'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01_MP3e-06/',
              'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df01_MP3e-06/',
              'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df01_MP3e-06_bf01/',
              #'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df01_MP3e-06_bf1/',
              #'Pd_N16e5_M3_R2_r50_b5_g75_MD001_Z10_a1_df01_MP3e-06/',
              #'Pd_N16e5_M1_R1_r50_b5_g75_MD001_Z10_a1_df01_MP3e-06/',
              #'Pd_N16e5_M1_R1_r50_b5_g75_MD001_Z10_a1_df01_MP3e-06_bf01/',
              #'Pd_N16e5_M1_R1_r50_b5_g75_MD001_Z10_a1_df01_MP3e-06_bf1/',
]

'''
runfolders = [#'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a01_df01_MP3e-06_i2/',
              #'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a01_df01_MP3e-06_i3/',
              #'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a01_df01_MP3e-06_i4/',
              #'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a01_df01_MP3e-06_i5/',
              'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a01_df01_MP3e-06/',
              'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01_MP3e-06/',
              'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df01_MP3e-06/'

]'''
'''
runfolders = [#'Pod_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a01_df01_MP3e-06/',
              #'Pod_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01_MP3e-06/',
              
              #'Pod_N16e5_M5_R3_r50_b5_g75_MD0001_Z10_a10_df01_MP3e-06/',
              #'Pod_N16e5_M5_R3_r50_b5_g75_MD0001_Z10_a10_df01_MP3e-06_bf01/',
              #'Pod_N16e5_M5_R3_r50_b5_g75_MD0001_Z10_a10_df01_MP3e-06_bf1/',
              #'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df01_MP3e-06/',
              #'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df01_MP3e-06_bf01/',
              #'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df01_MP3e-06_bf1/',
              'Pod_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df01_MP3e-06/',
              'Pod_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df01_MP3e-06_bf01/',
              'Pod_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df01_MP3e-06_bf02/',
              'Pod_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df01_MP3e-06_bf05/',
              'Pod_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df01_MP3e-06_bf1/',
]'''

runfolders = [#'Pod_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a01_df01_MP3e-06/',
              #'Pod_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df01_MP3e-06/',
              #'Pod_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df01_MP3e-06/',
              #'Pod_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df01_MP3e-06/',
              #'Pod_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df01_MP3e-06_bf02/',
              #'Pod_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df01_MP3e-06_bf05/',
              #'Pod_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df01_MP3e-06_bf1/',

              #'Pod_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df03_MP3e-06/',
              #'Pod_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df03_MP3e-06_bf01/',
              #'Pod_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df03_MP3e-06_bf02/',
              #'Pod_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df03_MP3e-06_bf02_dt10/',
              #'Pod_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df03_MP3e-06_bf02_dt10_2/',

              #'Pod_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df03_MP3e-06_bf05/',
              #'Pod_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df03_MP3e-06_bf1/',
              'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a01_df03_MP3e-06/',
              'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a1_df03_MP3e-06/',

              #'Pod_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df03_MP3e-06_bf1_tv300/',
              #'Pod_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df03_MP3e-06_bf1_tv1000/',

              'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df03_MP3e-06/',
              #'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df03_MP3e-06_bf01/',
              #'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df03_MP3e-06_bf02/',
              #'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df03_MP3e-06_bf05/',
              #'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df03_MP3e-06_bf1/',
              #'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df03_MP3e-06_bf1_soft1e3/',

              #'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df03_MP3e-06_bf1_tv10000/',
              #'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df03_MP3e-06_bf1_tv300/',
              #'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df03_MP3e-06_bf1_tv1000/',
              #'Pd_N16e5_M5_R3_r50_b5_g75_MD0003_Z10_a10_df03_MP3e-06_bf1_tv1000/',
              #'Pd_N16e5_M5_R3_r50_b5_g75_MD003_Z10_a10_df03_MP3e-06_bf1_tv1000/',
              #'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df03_MP3e-06_bf1_tv100_rc1e9/',
              #'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df03_MP3e-06_bf1_tv1000_rc1e9/',
              #'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a001_10_MP3e-06_bf1_tv1000/',
              #'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a01_10_MP3e-06_bf1_tv1000/',
              #'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df03_MP3e-06_bf1_tv1000_dt10/',
    #'Pd_N16e5_M1_R1_r50_b5_g75_MD001_Z10_a10_df03_MP3e-06/',
    #'Pd_N16e5_M1_R1_r50_b5_g75_MD001_Z10_a10_df03_MP3e-06_tv1000/',
    #'Pd_N16e5_M1_R1_r50_b5_g75_MD0003_Z10_a10_df03_MP3e-06_tv1000/',
    #'Pd_N16e5_M1_R1_r50_b5_g75_MD0001_Z10_a10_df03_MP3e-06_tv1000/',
    #'Pd_N16e5_M3_R2_r50_b5_g75_MD001_Z10_a10_df03_MP3e-06/',
    #'Pd_N16e5_M3_R2_r50_b5_g75_MD001_Z10_a10_df03_MP3e-06_tv1000/',
    #'Pd_N16e5_M3_R2_r50_b5_g75_MD0003_Z10_a10_df03_MP3e-06_tv1000/',
    #'Pd_N16e5_M3_R2_r50_b5_g75_MD0001_Z10_a10_df03_MP3e-06_tv1000/',
    #'Disc_N12e6_R00110_M0075/',
    #'Pd_N14e6_M5_R3_r50_b5_g75_MD001_Z10_a10_df02_MP3e-06_bf1_tv1000_dt50/',
    ]


runfolders = [
    #'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df03_MP3e-06/',
    #'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df03_MP3e-06_bf1_tv1000/',
    #'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df03_MP3e-06_bf1_tv10000/',
    #'Pd_N16e5_M5_R3_r50_b5_g75_MD0003_Z10_a10_df03_MP3e-06_bf1_tv1000/',
    #'Pd_N16e5_M5_R3_r50_b5_g75_MD003_Z10_a10_df03_MP3e-06_bf1_tv1000/',

    #'Pd_N14e6_M5_R3_r50_b5_g75_MD001_Z10_a10_df02_MP3e-06_bf1_tv1000_dt50_e2/',
    #'Pd_N14e6_M5_R3_r50_b5_g75_MD001_Z10_a10_df02_MP3e-06_bf1_tv1000_dt50_2/',
    #'Pd_N14e6_M5_R3_r50_b5_g75_MD001_Z10_a10_df02_MP3e-06_bf1_tv1000_dt50_3/',
    #'Pd_N14e6_M5_R3_r50_b5_g75_MD001_Z10_a10_df02_MP3e-06_bf1_tv1000_dt50_4/',

    #'Pd2_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df1_MP3e-06/',
    #'Pd2_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df1_MP3e-06_tv1000/',
    #'Pd2_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df1_MP3e-06_tv10000/',
    #'Pd2_N16e5_M5_R3_r50_b5_g75_MD003_Z10_a10_df1_MP3e-06_tv1000/',
    #'Pd2_N16e5_M5_R3_r50_b5_g75_MD0003_Z10_a10_df1_MP3e-06_tv1000/',
    #'Pd2_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df1_MP3e-06_tv1000_part1/',
    #'Pd2_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df1_MP3e-06_S0500_shutdown/',

]

runfolders = [
    #'Disc_N2e6_R00110_M01_g75/',
    #'Disc_N16e6_R00110_M01_g75/',
    #'Pd_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df03_MP3e-06_bf1_tv1000/',
    #'Pd2_N16e5_M5_R3_r50_b5_g75_MD001_Z10_a10_df1_MP3e-06_tv1000/',
    'Pd3_N2e6_M5_R3_r50_b5_g75_MD001_df03_MP3e-06_a01/',
    'Pd3_N2e6_M5_R3_r50_b5_g75_MD001_df03_MP3e-06_a1/',
    'Pd3_N2e6_M5_R3_r50_b5_g75_MD001_df03_MP3e-06_a10/',
    
    'Pd3_N2e6_M5_R3_r50_b5_g75_MD001_df03_MP3e-06_a10_tv1000/',
    'Pd3_N2e6_M5_R3_r50_b5_g75_MD001_df03_MP3e-06_a10_tv10000/',
    
    'Pd3_N2e6_M5_R3_r50_b5_g75_MD0003_df03_MP3e-06_a10_tv1000/',
    'Pd3_N2e6_M5_R3_r50_b5_g75_MD003_df03_MP3e-06_a10_tv1000/',
    #'Pd3_N2e6_M5_R3_r50_b5_g75_MD001_df03_MP3e-06_a10_tv1000_vap/',


    #'Pd3_N2e6_M5_R3_r50_b5_g75_MD001_df03_MP3e-06_a10_tv1000_NN1/',
    #'Pd3_N2e6_M5_R3_r50_b5_g75_MD001_df03_MP3e-06_a10_tv1000_NN2/',
    #'Pd3_N2e6_M5_R3_r50_b5_g75_MD001_df03_MP3e-06_a10_tv1000/',
    #'Pd3_N2e6_M5_R3_r50_b5_g75_MD001_df03_MP3e-06_a10_tv1000_NN8/',
    
    'Pd3_N16e6_M5_R3_r50_b5_g75_MD001_df03_MP3e-06_a10_tv1000/',    #

]
'''
runfolders = ['Poly_N1e5_M1_R1_n25_T20/',
              'Poly_N1e5_M1_R1_n25_T30/',
              #'Poly_N1e5_M1_R2_n25_T20/',
              #'Poly_N1e5_M1_R2_n25_T30/',
              'Poly_N2e5_M2_R2_n25_T20/',
              'Poly_N2e5_M2_R2_n25_T30/',
              'Poly_N3e5_M3_R2_n25_T20/',
              'Poly_N3e5_M3_R2_n25_T30/',

]'''

NNs = np.array([1,2,4,8,4])
res_labels = ['$NN_{FB}$=40','$NN_{FB}$=80','$NN_{FB}$=160','$NN_{FB}$=320','Hi res']
sd_labels = ['Feedback','Switch off at ... Yrs']

fb_labels = ['No Feedback','Fb = 10%','Fb = 20%', 'Fb = 50%','Fb = 100%',]
fb_labels = ['a = 1mm', 'a = 10cm']#'a = 1cm',
fb_labels = runfolders

a_labels  = ['a = 1mm', 'a = 1cm', 'a = 10cm']
tv_labels = ['No Feedback', r't$_{\rm{acc}}=10^3$ Yrs', r't$_{\rm{acc}}=10^4$ Yrs',r'$Z_0$=0.3%',r'$Z_0$=3%','High  res.']
shutdown_labels = [r'$L_{\rm core}$ = 0',r't$_{\rm{acc}}=10^3$ Yrs',r'$L_{\rm core}$ = 0 after 1600 Yrs']




def frag_plot(plot_vars=['sep','MP','MD'],rerun=False,Z_frac=False,link_i=np.array([])):
    '''Plot frag a, frag mass and accreted dust mass
    plot_vars 
    sep = separation plot
    MP  = Mass of planet inside RH/2
    U   = Internal energy of planet
    MD  = Mass of metals
    Z   = Planet metallicity
    RD  = Sedimentaiton height of dust
    '''

    fig11,axes = b.make_plots(11,len(plot_vars),figsize=(6.4,2*len(plot_vars)))
    toffset = 0
    fig2 = plt.figure(2)
    axb = fig2.add_axes([0.2,0.2,0.75,0.75])

    
    
    for i in range(len(runfolders)):
        print 'Runfolder', runfolders[i]
        beta = float(runfolders[i].split('_b')[1].split('_g')[0])
        print 'Beta =', beta
        
        try:
            if rerun == True:
                1/0
            load = np.load(savedir+runfolders[i].rstrip('//')+'_store.npy')
            print 'Loaded data'
        except:
            print 'No file, need to compute save array'
            load = b.bin_data_save(filepath,runfolders[i],Rin=0.1,Rout=2,beta=beta,
                                   Poly_data=True,sink_centre=True,sig_plots=False)

        if link_i != np.array([]):
            if i == link_i[0]:
                load_store = load[1:]
            if i in link_i+1:               
                try:
                    load_store = np.vstack((load_store,load[1:]))
                except:
                    print 'hack for shutdown. Lost original files so cant recalc'
                    tmp = np.zeros((150,2,200))
                    tmp_load = np.hstack((load[1:],tmp))
                    load_store = np.vstack((load_store,tmp_load))
                
                if i == link_i[-1]+1:
                    load = load_store
                    i = link_i[0]
                else:
                    continue
        print np.shape(load)
            
        #=== Read Gadget output from saved np array ===#
        N_planets   = load[0,0,20]
        time        = load[1:,0,0] +toffset#Years
        M_star      = load[1:,0,1] #Msol
        core_u      = load[1:,0,2]*code_L**2/code_time**2 #cgs
        rhoc        = load[1:,0,3] *code_M/code_L**3 #central density
        hc          = load[1:,0,4] * code_L #central smoothing length
        M_enc       = load[1:,0,5] * code_M #Mass inside core orbit
        core_vtot   = load[1:,0,6] * code_L/code_time #core velocity
        L_PP        = load[1:,0,7] * code_M *code_L**2/code_time**3/c.Lsol #Protoplanet Lum [Lsol]
        Z_gas       = load[1:,0,10]*AU_scale
        Z_dust      = load[1:,0,11]*AU_scale
        clump_pos   = load[1:,0,12:15]*AU_scale
        R_env       = load[1:,0,15]*AU_scale
        uc          = load[1:,0,16]*code_L**2/code_time**2 #cgs
        MD_env      = load[1:,0,17]*c.Msol/c.ME
        Mean_dust_R = load[1:,0,18]*AU_scale
        E_frag      = load[1:,0,19]*code_M*code_L**2/code_time**2
        MH_2        = load[1:,0,21]*c.Msol/c.MJ #MJ inside RH_2
        rP          = load[1:,0,22]*AU_scale #AU
        RH_2        = load[1:,0,23] #AU
        Macc_dust   = load[1:,0,24]*c.Msol/c.MJ #MJ inside RH_2
        M_bigsink   = load[1:,0,25]*c.Msol/c.ME
        planet_pos  = load[1:,0,26:29]*AU_scale #AU
        M_othersinks = np.sum(load[1:,0,29::4],axis=1)*c.Msol/c.ME
        RH =  rP* (MH_2/c.Msol*c.MJ/3/M_star)**(1/3)#AU

        
        if i in link_i:
            toffset = time[-1]
            print 'toffset', toffset
        else:
            toffset = 0
        
        planet_pos_rel = planet_pos - clump_pos
        core_dr = np.sqrt(planet_pos_rel[:,0]**2+planet_pos_rel[:,1]**2+planet_pos_rel[:,2]**2)*c.AU
        
        #==== Cass plot ====#
        #plt.figure(2)
        #plt.plot(time,E_frag,label=runfolders[i])
        #plt.legend(frameon=False)

        M_dust = Macc_dust*c.MJ/c.ME
        core_T = (c.gamma_dia-1)*core_u * c.mu * c.mp/c.kb
        Tc     = (c.gamma_dia-1)*uc * c.mu * c.mp/c.kb
        c_cs   = np.sqrt(uc*(c.gamma_dia-1))

        #==== Find sink masses ====#
        print 'N Planets', N_planets
        Mean_metal_R = Mean_dust_R * M_dust/(M_dust+M_bigsink) #AU

        #==== Find core luminosities ====#
        try:
            t_visc = float(runfolders[i].split('tv')[1].split('_')[0].rstrip('/'))*c.sec_per_year #sec
        except:
            print 'No t_visc found!'
            t_visc=0
        dts = np.diff(time[1:])*c.sec_per_year #sec
        M_disc = np.zeros(len(time)-1)
        Mdot_acc  = np.diff(M_bigsink[1:]*c.ME)/dts #g/sec
        Mdot_disc = 0
        for k in range(len(time)-2):
            if dts[k] == 0:
                print k
                print 'AHHH!'
            Mdot_disc = Mdot_acc[k] - M_disc[k]/t_visc #g/sec
            M_disc[k+1] = M_disc[k] + Mdot_disc*dts[k] #g
        Mdot_core = M_disc/t_visc #erg/sec
        L = (M_bigsink[1:]*c.ME)**(2/3)*Mdot_core*c.G * (4*np.pi*5/3)**(1/3)/c.Lsol

    
        fig13 = plt.figure(13,facecolor='white',figsize=(6,3))
        ax1  = fig13.add_axes([0.12,0.2,0.35,0.7])
        ax2  = fig13.add_axes([0.6,0.2,0.35,0.7])
        ax1.plot(planet_pos[:,0]-clump_pos[:,0],planet_pos[:,1]-clump_pos[:,1],color=plot_cols[i])
        ax1.set_xlabel('X [AU]')
        ax1.set_ylabel('Y [AU]')
        ax2.plot(planet_pos[:,0]-clump_pos[:,0],planet_pos[:,2]-clump_pos[:,2],color=plot_cols[i])
        ax2.set_xlabel('X [AU]')
        ax2.set_ylabel('Z [AU]')

        if i in link_i+1:
            i = link_i[0]
            tv_labels[i] = ''
            shutdown_labels[i] = ''

        #Stop plotting after disruption
        for k in range(len(MH_2)):
            if MH_2[k] < 1:
                break
        print 'Disruption at ', time[k], ' years'
        #k=800
        
        #======== Set data for each relevant plot var ========#
        for j in range(len(plot_vars)):
            if plot_vars[j] == 'sep':
                axes[j].set_ylabel(r'Orbital Sep [AU]')
                axes[j].plot(time[:k],rP[:k],color=plot_cols[i],label=tv_labels[i])#fb_labels[i]),label=runfolders[i])#
                axes[j].legend(frameon=False,loc=1,ncol=2)

            elif plot_vars[j] == 'MP': 
                axes[j].set_ylabel(r'$M_{\rm P}$ [M$_J$]')#'R$_H$/2 Mass [M$_J$]')
                axes[j].plot(time[:k],MH_2[:k],color=plot_cols[i])#,label='M$_{RH/2}$')
                axes[j].legend(frameon=False,loc=4)

            elif plot_vars[j] == 'U':
                E_rel = np.cumsum(L *c.Lsol* (time[1:]-time[:-1])*c.sec_per_year)
                E_frag_smooth  = np.convolve(E_frag, np.ones((5,))/5, mode='valid')
                time_smooth = np.convolve(time, np.ones((5,))/5, mode='valid')
                axes[j].set_ylabel(r'$E_{tot}$ [10$^{41}$ erg]')
                #Norm_E_frag = E_frag[1:k]/abs(E_frag[1])
                #axes[j].set_ylabel(r'Normalised $E_{frag}$')
                #axes[j].plot(time[1:],Norm_E_frag,color=plot_cols[i],label=r'$E_{frag}$')
                axes[j].plot(time_smooth[:k],E_frag_smooth[:k]/1e41,color=plot_cols[i],label=r'$E_{tot}$')
                #axes[j].plot(time[:k],E_rel[:k]/1e41,ls='--',color=plot_cols[i],label=r'$E_{rel}$')

            elif plot_vars[j] == 'MD':
                axes[j].set_ylabel(r'Metal Mass [M$_\oplus$]')
                if j != 0:
                    axes[j].plot(time[:k],M_bigsink[:k]+M_othersinks[:k]+M_dust[:k],label='Total metal mass',color=plot_cols[i])
                    axes[j].plot(time[:k],M_bigsink[:k],color=plot_cols[i],ls='--',label='Core mass')
                    #axes[j].plot(time,MD_env,color=plot_cols[i],ls='-.',label='Envelope Mass')
                    #axes[j].plot(time,M_othersinks,color=plot_cols[i],ls=':',label='Other sinks')
                    if i == 0:
                        axes[j].legend(frameon=False,loc=1,ncol=2)

                if j == 0:
                    axes[j].plot(time[:k],M_bigsink[:k]+M_othersinks[:k]+M_dust[:k],label=a_labels[i],color=plot_cols[i])
                    axes[j].plot(time[:k],M_bigsink[:k],color=plot_cols[i],ls='--')
                    if i == len(runfolders)-1:
                        axes[j].plot([],[],color=plot_cols[i],label='Total metal mass')
                        axes[j].plot([],[],color=plot_cols[i],label='Core mass',ls='--')
                    axes[j].legend(frameon=False,loc=4,ncol=2)
                    
            elif plot_vars[j] == 'Z':
                axes[j].set_ylabel(r'Z')
                Z_disc = runfolders[i].split('MD')[1].split('_')[0]
                Z_disc = float('0.'+Z_disc[1:])
                Z = (Macc_dust+(M_bigsink+M_othersinks)*c.ME/c.MJ) / MH_2
                Z_env = MD_env*c.ME/c.MJ / (MH_2-MH_2[1])
                axes[j].plot(time[:k],Z[:k],color=plot_cols[i],label='Total protoplanet')
                axes[j].plot(time[:k],Z_env[:k],color=plot_cols[i],ls='-.',label='Atmosphere only')
                if i == 0:
                    axes[j].legend(frameon=False,loc=4)

            elif plot_vars[j] == 'RD':
                axes[j].set_ylabel(r'Mean dust $r$ [AU]')
                axes[j].plot(time[:k],Mean_metal_R[:k],color=plot_cols[i])#,label=a_labels[i])
                axes[j].plot(time[:k],R_env[:k],color=plot_cols[i],ls=':')
                axes[j].plot(time[:k],RH[:k],color=plot_cols[i],ls='--')

                if i == len(runfolders)-1:
                    axes[j].plot([],[],color=plot_cols[0],label=r'$r_{\rm grains}$')
                    axes[j].plot([],[],color=plot_cols[0],ls=':',label=r'$r_P(M_0)$')

                axes[j].legend(frameon=False,loc=1)

            elif plot_vars[j] == 'Tc':
                axes[j].set_ylabel(r'$T_C$ [K]')
                #axes[j].plot(time[:k],Tc[:k],color=plot_cols[i],ls='--')
                axes[j].plot(time[:k],core_T[:k],color=plot_cols[i])


            elif plot_vars[j] == 'Fb':
                axes[j].set_ylabel(r'Luminosity [L$_\odot$]')
                axes[j].plot(time[:k],L[:k],color=plot_cols[i])#,label='Core')
                if len(runfolders) == 1:
                    axes[j].plot(time[:k],L_PP[:k],color=plot_cols[1],label='Protoplanet',ls='--')
                axes[j].semilogy()
                axes[j].legend(frameon=False)

            elif plot_vars[j] == 'R_core':
                R_label = res_labels[i]
                if len(runfolders)==1:
                    axes[j].plot(time,hc/c.AU,label='Smoothing length',ls='--',color=plot_cols[1])
                    R_label = 'Core offset'
                axes[j].plot(time,core_dr/c.AU,color=plot_cols[i],label=shutdown_labels[i])#label=runfolders[i])#R_label)

                axes[j].set_ylabel('Offset [AU]')
                axes[j].semilogy()
                axes[j].legend(frameon=False)

            elif plot_vars[j] == 'M_core':
                M_enc_smooth  = np.convolve(M_enc, np.ones((5,))/5, mode='valid')
                time_smooth = np.convolve(time, np.ones((5,))/5, mode='valid')
                axes[j].plot(time,M_bigsink,color=plot_cols[i],label='Core mass')
                if len(runfolders)==1:
                    axes[j].plot(time_smooth,M_enc_smooth/c.ME,color=plot_cols[1],label='Enclosed mass',ls='--')
                axes[j].set_ylabel(r'Mass [$M_E$]')
                axes[j].semilogy()
                if i == 0:
                    axes[j].legend(frameon=False,loc=1)

            elif plot_vars[j] == 'v_core':
                chi = hc*c_cs #* NNs[i]**(1/3) #smooth x injection N
                v_MVR17 = 3* c.gamma_dia*(c.gamma_dia-1)*L*c.Lsol*c_cs[1:] / (8*np.pi*rhoc[1:]*c.G*M_bigsink[1:]*c.ME) / chi[1:]
                #opac = 1e-4 #hack? Need table?
                opac = b.Zhu_op(Tc,rhoc)
                c_p = c.kb /c.mu/c.mp/(c.gamma_dia-1)
                th_cond = 16*c.sigma_SB*core_T**3 / (3 * rhoc *opac)
                th_diff = th_cond/rhoc/c_p
                v_raddiff = v_MVR17 *chi[1:]/th_diff[1:]
                
                m1 = M_bigsink*c.ME
                m2 = M_enc
                #v_K  = abs(np.sqrt(c.G/core_dr/(m1+m2))*(m2-m1))
                v_K  = np.sqrt(c.G*M_enc/core_dr)

                #smooth v_K
                v_K_smooth  = np.convolve(v_K, np.ones((5,))/5, mode='valid')
                v_core_smooth  = np.convolve(core_vtot, np.ones((5,))/5, mode='valid')
                v_MVR17_smooth  = np.convolve(v_MVR17, np.ones((5,))/5, mode='valid')

                time_smooth = np.convolve(time, np.ones((5,))/5, mode='valid')

                axes[j].plot(time_smooth,v_core_smooth,color=plot_cols[i],label=r'$v_{\rm c}$')
                if len(runfolders)==1:
                       axes[j].plot(time_smooth,v_K_smooth,color=plot_cols[i],ls='--',label=r'$v_{\rm Kep}$')
                       axes[j].plot(time[1:],v_MVR17,color=plot_cols[i],ls=':',label=r'$v_{\rm MRV17}$')
                       #axes[j].plot(time[1:],v_raddiff,color=plot_cols[i],ls='-.',label='$v_{RDA}$')
                       axes[j].plot(time,c_cs,color=plot_cols[i+1],ls='-.',label=r'Sound speed')
                axes[j].semilogy()
                axes[j].set_ylabel('Velocity [cms$^{-1}$]')
                if i == 0:
                    axes[j].legend(frameon=False)#,loc=4,ncol=2)
                


        #ax2.plot(time,Z_gas,color=plot_cols[i],label='Z$_{gas}$',ls='--')
        #ax2.plot(time,Z_dust,color=plot_cols[i],label='Z$_{dust}$',ls=':')
        #ax2.plot(time,M_star,color=plot_cols[i],label=r'$M_*$',ls=':')
        #ax2.axhline(0)

        #For Cass U comparison
        #fig2 = plt.figure(2)
        #plt.plot(time[1:],E_frag[1:],color=plot_cols[i],label=r'Normalised $E_{frag}$',ls='-.')


    #if betaval != 'nan':
    #    ax1.text(0.75,0.9,r'$\beta$ = '+str(betaval), transform=ax1.transAxes)

    axes[0].set_xlim(time[0])#,time[-1])
    axes[-1].set_xlabel('Time [Years]')


    axb.plot(planet_pos[:,0],planet_pos[:,1])

    return




def feedback_plot():
    '''Calculate the analytic fraction of feedback energy on a planet'''

    xi = 1
    mdot_t = np.logspace(26,30,100) #g
    rho_c = 5
    M_f = 5*c.MJ
    R_f1 = 1*c.AU
    R_f2 = 3*c.AU

    tmp = xi * (rho_c*4*np.pi/3)**(1/3)  * (mdot_t)**(5/3) /M_f**2
    E_U1 = tmp * R_f1
    E_U2 = tmp * R_f2

    fig1 = plt.figure(4,facecolor='w')
    ax1 = fig1.add_axes([0.15,0.15,0.8,0.8])
    ax1.plot(mdot_t/c.ME,E_U1,label=r'$r_{\rm p}$= 1 AU',color=run_cols[0])
    ax1.plot(mdot_t/c.ME,E_U2,label=r'$r_{\rm p}$= 3 AU',color=run_cols[1])
    ax1.axhline(1,color=run_cols[2],ls='--')
    ax1.semilogx()
    ax1.semilogy()
    ax1.set_xlabel(r'M$_{\rm c}$ [$M_\oplus$]')
    ax1.set_ylabel(r'$E_{\rm fb}/E_{\rm tot}$')
    plt.legend(frameon=False,loc=2)
    plt.show()
    
    
def dust_sedimentation(filepath,runfolders,snapid):
    '''Check sedimentation velocities inside fragments'''
    
    #==== Radial binning ====#
    Rin=0.001
    Rout=0.05
    N_Rbins  = 50
    Rbins = np.linspace(Rin,Rout,N_Rbins+1)
    Rbin_mids = (Rbins[1:]+Rbins[:-1])/2

    plt.figure(0,facecolor='w')
    shades = []
    for i in range(len(runfolders)):
        S = b.Load_Snap(filepath,runfolders[i],snapid)
        M_gas  = S.M_gas
        M_dust = S.M_dust

        #Zoom on clump
        rho_sort  = np.argsort(S.gas_rho)
        zoom_pos = np.mean(S.gas_pos[rho_sort[-10:],:],axis=0)
        zoom_vel = np.mean(S.gas_vel[rho_sort[-10:],:],axis=0)

        S.gas_pos  = S.gas_pos  - zoom_pos
        S.dust_pos = S.dust_pos - zoom_pos
        S.dust_vel = S.dust_vel - zoom_vel
        S.gas_vel  = S.gas_vel  - zoom_vel

        r_gas  = np.sqrt(S.gas_pos[:,0]**2+S.gas_pos[:,1]**2+S.gas_pos[:,2]**2)
        r_dust = np.sqrt(S.dust_pos[:,0]**2+S.dust_pos[:,1]**2+S.dust_pos[:,2]**2)

        gas_count    = np.histogram(r_gas,Rbins)[0]
        dust_count   = np.histogram(r_dust,Rbins)[0]
        M_enc        = (np.cumsum(gas_count)*M_gas + np.cumsum(dust_count)*M_dust )
        gas_u        = (b.calc_binned_data(S.gas_u,r_gas,Rbins)[0])
        gas_cs       = np.sqrt((c.gamma_dia-1)*gas_u)
        v_th         = np.sqrt(8/np.pi)*gas_cs
        
        Rbin_volumes = 4*np.pi/3 * (Rbins[1:]**3-Rbins[:-1]**3)
        gas_rho      = gas_count*M_gas/Rbin_volumes

        a     = np.mean(S.dust_a) /code_L
        rho_a = 3 /code_M*code_L**3
        v_sed = a*rho_a * code_G * M_enc / (v_th * (Rbin_mids)**2 *gas_rho)


        #dust velocities
        v_r, v_azi = b.v_r_azi(S.dust_pos,S.dust_vel)
        tmp     = b.calc_binned_data(v_r,r_dust,Rbins)
        dust_vr,sig_dust_vr = tmp[0],tmp[1]
        pvr   = -dust_vr
        pvr_p = (-dust_vr + sig_dust_vr)
        pvr_m = (-dust_vr - sig_dust_vr)

        #gas velocities
        v_r, v_azi = b.v_r_azi(S.gas_pos,S.gas_vel)
        tmp     = b.calc_binned_data(v_r,r_gas,Rbins)
        gas_vr,sig_gas_vr = -tmp[0], tmp[1]


        #==== Plotting code ====#
        plot_units = code_time/c.sec_per_year #code_L/code_time #cm/s
        #plot_units = 100/code_time*c.sec_per_year #AU/year
        
        #plt.scatter(r_dust,v_r)
        #plt.scatter(r_dust,S.dust_vel[:,1],s=1)
        #fill_min = np.maximum(pvr_m*plot_units,np.ones(len(pvr_m))*1e-5)
        #plt.fill_between(Rbin_mids*100,pvr_p*plot_units,fill_min,alpha=0.2,color=run_cols[i],label = runfolders[i])
        plt.plot(Rbin_mids*100,Rbin_mids/v_sed*plot_units,color=run_cols[i],ls='--')#r'an $V_{set}$'
        plt.plot(Rbin_mids*100,Rbin_mids/pvr*plot_units,label=runfolders[i],color=run_cols[i])#r'Mean dust $V_r$'
        plt.plot(Rbin_mids*100,Rbin_mids/gas_vr*plot_units,color=run_cols[i],ls=':')#label=r'Gas $V_r$'
        plt.plot(Rbin_mids*100,Rbin_mids/(pvr-gas_vr)*plot_units,color=run_cols[i],ls='-.')#r'Dust-Gas $V_r$'
        #plt.plot(Rbin_mids*100,gas_cs*plot_units,label=r'Gas c_s',ls='-.',color=run_cols[i])

    plt.plot([],[],color=run_cols[0],label=r'Mean dust $t_{set}$')
    plt.plot([],[],color=run_cols[0],label=r'an $t_{set}$',ls='--')
    plt.plot([],[],color=run_cols[i],label=r'Dust-Gas $t_{set}$',ls='-.')
    plt.plot([],[],color=run_cols[i],label=r'Gas $t_{set}$',ls=':')

    plt.legend(loc=2,frameon=False)
    
    plt.yscale('symlog', linthreshy=10)
    plt.xlabel('Fragment radius [AU]')
    plt.ylabel(r'$t_{set}$ [Years]')#'cm s$^{-1}$')
    #plt.semilogy()
    return



def break_om(M_enc,r):
    '''Breakup velocities'''
    om_br = np.sqrt(M_enc*c.G/r**3)
    return om_br


def L_collapse(PP_rs,PP_rhos,PP_vazis):
    '''Analyse ang mom profile'''

    #v_azi = 2e4 #cm/s
    #rs = np.linspace(0,1,101)*c.AU

    rs = PP_rs*code_L
    v_azi = PP_vazis *code_L/code_time    
    drs = np.diff(rs)
    r_mids = (rs[1:]+rs[:-1])/2
    oms = v_azi/r_mids
    
    
    rhos = PP_rhos
    R_PP = rs[np.where(rhos<2e-11)[0][0]]
    print 'R_PP', R_PP/c.AU
    
    M_sheaths =  2*np.pi * rs* np.sqrt(R_PP**2-rs**2) *rhos * drs 
    I_sheaths = 2/3 * M_sheaths * r_mids**2
    L_PPs = oms*I_shells
    L_PP = np.sum(L_PPs)
    M_encs = np.cumsum(M_shells)
    M_PP = M_encs[-1]
    
    print 'L_PP, M_PP', 'R_PP'. L_PP, M_PP/c.MJ, R_PP/c.AU
    om_brs = break_om(M_encs,r_mids)

    plt.figure(30)
    plot_rs = r_mids/c.AU
    plt.plot(plot_rs,oms,color=plot_cols[0],label='Oms')
    plt.plot(plot_rs,om_brs,color=plot_cols[1],label='Breakup oms')
    plt.legend()
    plt.semilogy()

    plt.figure(31)
    R_P = 2*c.MJ

    M_leftover = M_PP-M_encs
    om_P_brs = break_om(M_encs,R_P)
    L_Ps = 2/5*M_encs*R_P**2 * om_P_brs

    r_circ 
    
    plt.plot(M_encs/c.MJ,L_Ps)
    plt.axhline(L_PP)
    plt.semilogy()
    plt.semilogx()

    return

    

def profile_plots(zoom='Zrho',ind=0,rerun=False,plot_vars=['T','rho','s','Z','M_jeans','h'],vr_mode=False):
    '''Plot 1D profiles of gas and dust'''
    dg_0 = 0.01
    
    fig12,axes = b.make_plots(ind,len(plot_vars))
    axes[0].semilogx()
    
    for i in range(len(runfolders)):
        print 'Runfolder', runfolders[i]
        try:
            print 'Try: ', savedir+runfolders[i].rstrip('//')+zoom+'_store.npy'
            if rerun == True:
                1/0
            load = np.load(savedir+runfolders[i].rstrip('//')+zoom+'_store.npy')
            print 'Loaded data'
        except:
            print 'No file, need to compute save array'
            load = b.bin_data_save(filepath,runfolders[i],Rin=0.0,Rout=0.05,zoom=zoom,
                                   max_dust_dens=False,sink_centre=True,vr_mode=vr_mode,ind=ind-1)
            
        #=== Read Gadget output from saved np array ===#
        time      = load[:,0,0] #Years
        M_star    = load[:,0,1] #code M
        M_gas     = load[0,0,2] #code M
        M_dust    = load[0,0,3] #code M

        N_Rbins,Rin,Rout = load[0,0,4],load[0,0,5],load[0,0,6]
        Rbins = np.linspace(Rin,Rout,N_Rbins+1)
        Rbin_mids = (Rbins[1:]+Rbins[:-1])/2 
        dRbin = Rbins[1]-Rbins[0]
        Rbin_areas = Rbin_mids*2*np.pi*dRbin
        Rbin_volumes = 4*np.pi/3 * (Rbins[1:]**3-Rbins[:-1]**3)
        N_abins = int(load[0,0,7])
        
        gas_rho  = load[ind,1,:]*M_gas/Rbin_volumes*code_M/code_L**3
        dust_rho = np.sum(load[ind,7:7+N_abins,:],axis=0)*M_dust/Rbin_volumes*code_M/code_L**3
        rho_tot = gas_rho+dust_rho

        print np.shape(rho_tot), rho_tot
        print np.shape(Rbins[1:]), Rbins[1:]
        M_enc_gas  = np.cumsum(gas_rho * Rbin_volumes*code_L**3)
        M_enc_dust = np.cumsum(dust_rho * Rbin_volumes*code_L**3)
        M_enc = M_enc_gas + M_enc_dust
        
        Z = dust_rho/gas_rho
        gas_cs = np.sqrt(load[ind,2,:] * (c.gamma_dia-1)) *code_L/code_time
        gas_T = gas_cs**2 * c.mu*c.mp/c.kb
        
        #mu_R = c.mu*(1+dust_rho/gas_rho)
        gas_s = gas_cs**2 * (rho_tot)**(1-c.gamma_dia)
        gas_h = load[ind,3,:]*AU_scale #[AU]

        M_jeans = np.pi/6 * gas_cs**3 / np.sqrt(c.G**3 * rho_tot)

        #max_rho_local = load[ind,7+N_abins,:]
        #max_dust_gas = load[ind,7+N_abins+1,:]

        if vr_mode == True:
            v_azi = load[:,6,:]
            print np.shape(v_azi)
            L_collapse(Rbins,gas_rho,v_azi[ind])

        for j in range(len(axes)):
            if plot_vars[j] == 'T':
                axes[j].plot(Rbin_mids*AU_scale,gas_T,color=run_cols[i])
                axes[j].set_ylabel(r'Temp. [K]')

            elif plot_vars[j] == 'rho':
                axes[j].plot(Rbin_mids*AU_scale,gas_rho,color=run_cols[i])
                axes[j].plot(Rbin_mids*AU_scale,dust_rho,color=run_cols[i],ls='--')
                #axes[j].plot(Rbin_mids*AU_scale,max_rho_local,color=run_cols[i],ls=':')

                axes[j].set_ylabel(r'$\rho$ [gcm$^{-3}$]')
                axes[j].semilogy()

            elif plot_vars[j] == 's':
                axes[j].plot(Rbin_mids*AU_scale,gas_s,color=run_cols[i])
                axes[j].set_ylabel(r'Entropy')

            elif plot_vars[j] == 'Z':    
                axes[j].plot(Rbin_mids*AU_scale,Z,color=run_cols[i])
                #axes[j].plot(Rbin_mids*AU_scale,max_dust_gas,color=run_cols[i],ls=':')
                axes[j].set_ylabel('Dust to Gas Ratio')
                axes[j].semilogy()

            elif plot_vars[j] == 'h':
                axes[j].plot(Rbin_mids*AU_scale,gas_h,color=run_cols[i])
                axes[j].set_ylabel(r'Smoothing [AU]')
                axes[j].semilogy()

            elif plot_vars[j] == 'M_jeans':
                axes[j].plot(Rbin_mids*AU_scale,M_jeans/c.MJ,color=run_cols[i],label='Jeans mass')
                axes[j].plot(Rbin_mids*AU_scale,M_enc/c.MJ,ls='--',color=run_cols[i],label='Enclosed mass')

                axes[j].set_ylabel(r'Jeans Mass [$M_J$]')
                axes[j].legend(frameon=False,loc=2)

            elif plot_vars[j] == 'M_enc':
                axes[j].plot(Rbin_mids*AU_scale,M_enc_gas/c.ME,color=run_cols[i],label='Gas')
                axes[j].plot(Rbin_mids*AU_scale,M_enc_dust/c.ME,ls='--',color=run_cols[i],label='Dust')
                axes[j].set_ylabel(r'Enclosed mass [$M_{\oplus}$]')
                axes[j].legend(frameon=False,loc=2)
                axes[j].semilogy()

                
            #axes[j].axhline(1)
            if j == len(axes)-1:
                axes[j].set_xlabel('Protoplanet radius [AU]')
                
            RP = abs(load[ind,0,26]*AU_scale)
            print 'RP', RP
            axes[j].axvline(RP,color=run_cols[len(runfolders)])
            #axes[j].axvline(gas_h[0],ls='--',color=run_cols[len(runfolders)+1])

        if i == 0:
            axes[0].text(0.75,1.02,'Time: {:.0f} '.format(time[ind]) + ' Yrs', transform=axes[0].transAxes)

        axes[0].legend(frameon=False,loc=4)



def cass_runs():
    ''''''
    cass_M = c.Msol
    cass_L = c.AU
    cass_t = c.sec_per_year
    
    fig7,axes = b.make_plots(7,2,figsize=(6,6))
    axes[0].set_yscale('symlog', linthreshy=1e39)
    axes[0].set_xlabel('Time [Years]')
    axes[0].set_ylabel(r'$E_{tot}$ [erg]')

    runs = ['run1','run2','run3','run4','run5','run6','run7','run8','run9',]
    #U0,Uf,dt = [],[],[]
    Us,ts = [],[]
    
    for k in range(len(runs)):
        cass_load = '/home/r/rjh73/stored_data/tarquin/'+runs[k]+'.dat'
        cass_data = np.genfromtxt(cass_load,skip_header=8,dtype='str')

        inds,pid,pid_old = [],0,0
        spheres = cass_data[:,1]

        for i in range(len(spheres)):
            pid = spheres[i].split('sphere')[1].split('.')[0]
            if pid != pid_old:
                inds.append(i)
            pid_old = pid
        print inds
        inds.append(len(spheres))


        for i in range(len(inds)-1):

            clump = np.reshape(cass_data[inds[i]:inds[i+1],:],(-1,100,7))
            time  = (clump[:,0,2].astype(np.float))*cass_t #+(k*2000*cass_t)  #s
            T     = clump[:,:,6].astype(np.float)
            u     = np.sum(T *c.kb/c.mu/c.mp/(c.gamma_mono-1),axis=1) #cm^2/s^2

            R     = clump[:,:,3].astype(np.float)*cass_L #cm
            M_enc = clump[:,:,4].astype(np.float)*cass_M #g
            dM    = M_enc[:,1:] - M_enc[:,:-1]
            M     = np.hstack((M_enc[:,0][:,None],dM))

            G_pot = c.G * np.sum(M_enc[:,1:]*dM/R[:,1:],axis=1)
            G_pot_core = 3/5*c.G*M_enc[:,0]**2/R[:,0]  #Central bin

            T = clump[:,:,6].astype(np.float)
            u_spec = T *c.kb/c.mu/c.mp/(c.gamma_mono-1) #cm^2/s^2
            u = np.sum(u_spec * M,axis=1)

            U = u - G_pot-G_pot_core
            thresh = -1e39
            
            try:
                t_shift = time[U<thresh][0]
            except:
                t_shift = 0
            axes[0].plot((time[U<thresh]-t_shift)/c.sec_per_year,U[U<thresh])

            try:
                a = U[U<thresh][0]
                Us.append(U[U<thresh])
                ts.append(time[U<thresh]-t_shift)
            except:
                pass


    #======== Make a fun U plot ========#
    U0,Uf,dt = [],[],[]
    for i in range(len(Us)):
        U0.append(Us[i][0])
        Uf.append(Us[i][-1])
        dt.append(ts[i][-1]-ts[i][0])
    U0,Uf,dt = np.array(U0),np.array(Uf),np.array(dt)/c.sec_per_year

    dU_dts = (Uf-U0)/dt/c.sec_per_year
    dU_dt,dU_dt_sig = norm.fit(dU_dts)
    print 'Mean U0', np.mean(U0)
    print 'Mean Uf', np.mean(Uf)
    print 'Mean dt', np.mean(dt)
    print 'Mean dU_dt', dU_dt,dU_dt_sig

    Mdots_plot = np.array([4,10,40])
    Mdots = Mdots_plot*c.ME/1000/c.sec_per_year #g/s
    t = np.logspace(0,3.5,100) #Yr
    rho_c = 5 #g/cm^3
    L_Md = c.G * (4*np.pi*rho_c/3)**(1/3) *(t*c.sec_per_year)**(2/3)

    #==== Make prob heat map ====#
    min_dU,max_dU = 1e28,1e34
    dU_dts[dU_dts>0] = -min_dU #Ignore positive U growth
    dU_dts = np.flip(np.sort(dU_dts),0)

    L = len(dU_dts)
    per = np.arange(0.2,1,0.2)
    dU_dt_lines = []
    for i in range(len(per)):
        dU_dt_lines.append(dU_dts[int(per[i]*L)])

    axes[0].set_xlim(0,1500)
    
    

    #---- Edot plot ----#
    axes[1].semilogy()
    #axes[1].set_xlim(t[0],t[-1])
    axes[1].set_ylim(4e28,1.6e32)
        
    for i in range(len(dU_dt_lines)):
        linecol = '#FFC300'
        axes[1].text(50,-dU_dt_lines[i]*1.1,str(int(per[i]*100))+'%',color=linecol)#, transform=ax1.transAxes)
        axes[1].axhline(-1*dU_dt_lines[i],color=linecol)

    for i in range(len(Mdots)):
        axes[1].plot(t,L_Md*Mdots[i]**(5/3),label=r'$\dot{M_{\rm c}}$ = '+str(Mdots_plot[i])+r' M$_{\oplus}/1000 Yr$',
                 color=run_cols[i])
        
    axes[1].set_xlabel('Time [Years]')
    axes[1].set_ylabel(r'$\dot{E}_{\rm tot}$ [erg/s]')
    plt.legend(frameon=False,loc=4)

    
    #---- Core mass plot ----#
    M_cores = np.zeros((len(Mdots),len(Us)))
    tfs     = np.zeros((len(Mdots),len(Us)))

    for i in range(len(Mdots)):
        for j in range(len(Us)):
            for k in range(len(ts[j])):
                t = ts[j][k]
                Ec = 3/5 * c.G * (4*np.pi*rho_c/3)**(1/3) * (Mdots[i]*t)**(5/3)

                if Ec > abs(Us[j][k]):
                    M_cores[i,j] = Mdots[i]*t /c.ME
                    tfs[i,j] = t/c.sec_per_year
                    break           
    M_bins = np.arange(1,int(np.max(M_cores)+4),2)

    plt.figure(8,figsize=(6,4))
    for i in range(len(Mdots)):
        plt.hist(M_cores[i],bins=M_bins,label=r'$\dot{M_{\rm c}}$ = '+str(Mdots_plot[i])+r' $M_{\oplus}$/1000Yr',
                     color=run_cols[i],histtype='step')
    plt.legend(frameon=False)

    plt.xlabel(r'Core mass [$M_{\oplus}$]')
    plt.ylabel('Count')
    plt.xlim(1,M_bins[-1])

    
    return



def dust_cluster_analysis(snapid):
    ''''''
    N_clusters = 12
    S = b.Load_Snap(filepath,runfolders[0],snapid,slen=4)

    zoom_lim = 0.02
    zoom_pos,zoom_vel = S.max_rho()
    S.zoom(zoom_pos,zoom_vel)
    S.subsample(zoom_lim)

    '''
    kmeans = KMeans(n_clusters=N_clusters)
    kmeans.fit(S.dust_pos)

    print kmeans.cluster_centers_
    #y_km = kmeans.fit_predict(S.dust_pos)

    plt.figure(14)
    for i in range(N_clusters):
        #plt.scatter(S.dust_pos[y_km==i,0],S.dust_pos[y_km==i,1])
        plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],alpha=0.5)
    '''
    hist_dim = 1000
    x_edges = np.linspace(-zoom_lim,zoom_lim,hist_dim+1)
    y_edges = np.linspace(-zoom_lim,zoom_lim,hist_dim+1)
    x_bins = (x_edges[1:]+x_edges[:-1])/2
    y_bins = (x_edges[1:]+x_edges[:-1])/2

    dust_hist2d,xedges,yedges = np.histogram2d(S.dust_pos[:,0],S.dust_pos[:,1],bins=(x_edges,y_edges))

    plt.contour(x_bins,y_bins,dust_hist2d,levels=[10,100,1000,10000],color=plot_cols[0])

    norm = LogNorm(vmin=1,vmax=1e4)
    #plt.hist2d(S.dust_pos[:,0],S.dust_pos[:,1],bins=1000,norm=norm)
    plt.pcolormesh(x_bins,y_bins,dust_hist2d,cmap='YlOrRd',norm=norm)


    plt.scatter(S.planets_pos[0,0],S.planets_pos[0,1],color=plot_cols[3])
    return



    
def eccentricty_kick():
    M_star = c.Msol
    r = 40*c.AU
    v = np.sqrt(c.G*M_star/r)

    dv = 0.2*v

    a = 1 / (2/r - (v+dv)**2/(c.G*M_star))

    print a
    e = 1 - r/a
    print 'If collision at periastron, e = ', e
    return
    

def solar_system_mass_radius():
    Masses = np.array([3.3e23,4.87e24,5.97e24,6.42e23,1.9e27,5.68e26,8.68e25,1.02e26,1.31e22])
    Radii  = np.array([5.8e10,1.08e11,1.49e11,2.28e11,7.78e11,1.43e12,2.87e12,4.49e12,5.87e12])




    
if __name__ == "__main__":
    #cass_runs()
    
    #feedback_plot()
    #frag_plot(plot_vars=['sep','MP','U','Fb','MD'],rerun=False)#,link_i=np.array([2,3,4]))
    #frag_plot(plot_vars=['sep','MP','Fb','MD'],rerun=False)#,link_i=np.array([5,6,7]))
    #frag_plot(plot_vars=['R_core','M_core','Fb','v_core'],rerun=False)#,link_i=np.array([2,3,4]))

    #frag_plot(plot_vars=['MD','RD','Z'],rerun=False)
    #frag_plot(plot_vars=['R_core','M_core','Fb'],rerun=False,link_i=np.array([2]))#,link_i=np.array([2,3,4]))

    #b.animate_1d(filepath,runfolders,var2='T',Rin=0.1,Rout=1.2,rerun=True,norm_y=False,write=False)
    #b.animate_1d(filepath,runfolders,var1='T',var2='rho',Rin=0.0,Rout=0.05,rerun=False,norm_y=False,zoom='Zrho',write=False,vr_mode=False)

    #b.animate_1d(filepath,runfolders,var1='T',var2='rho',Rin=0.0,Rout=0.05,rerun=False,norm_y=False,zoom='Zrho',write=False,vr_mode=False)

    profile_plots(zoom='Zrho',ind=280,plot_vars=['T','M_enc','rho','Z','h'],vr_mode=True,rerun=False)
    #profile_plots(zoom='Zdust',ind=388,plot_vars=['T','M_enc','rho','Z','h'],rerun=True)

    #inds = np.arange(380,391,1)
    #for i in inds:
    #    profile_plots(zoom='Zrho',ind=i,plot_vars=['T','M_enc','rho','Z','h'])

        
    #dust_cluster_analysis(snapid=387)
    
    #dust_sedimentation(filepath,runfolders,30)
    #mixture_model(runfolders[1],snapid=100)
    #eccentricty_kick()

    #L_collapse()
    
    plt.show()



    
