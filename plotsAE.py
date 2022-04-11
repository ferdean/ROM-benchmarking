"""
Plotting library, complement of main.py and lib.py created with the objective 
of keeping the main script clean
=========================================================================
 Created by:   Ferran de Andrés (12.2021) 
 Reviewed by:  -
=========================================================================
"""
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

import libAE

def plot_original(xx,yy,v):
    
    v_mean = v.mean(axis = 0)
    v_var  = v - v_mean
    
    plt.rcParams.update({'font.size' : 9})
    plt.rcParams['axes.linewidth'] = 0.7 #set the value globally
    
    CM2INCH = 0.393701
    GR = (1+np.sqrt(5))/2 # Golden ratio
    
    fig, ax = plt.subplots(2, 2, sharex = True, sharey = True, 
                           figsize = (20*CM2INCH, 13*CM2INCH/GR))
    
    
    
    ax[0, 0].tick_params(direction= 'in', which= 'major', length= 2, bottom= True, top= True,
                    right= True, left=True, width = 0.7)
    ax[0, 1].tick_params(direction= 'in', which= 'major', length= 2, bottom= True, top= True,
                    right= True, left=True, width = 0.7)
    ax[1, 0].tick_params(direction= 'in', which= 'major', length= 2, bottom= True, top= True,
                    right= True, left=True, width = 0.7)
    ax[1, 1].tick_params(direction= 'in', which= 'major', length= 2, bottom= True, top= True,
                    right= True, left=True, width = 0.7)
    
    plt.set_cmap('RdBu_r')
    
    ax[0, 0].contourf(xx, yy, v[0, :, :, 0], levels = 20)
    ax[0, 1].contourf(xx, yy, v[0, :, :, 1], levels = 20)
    ax[1, 0].contourf(xx, yy, v_var[0, :, :, 0], levels = 20)
    ax[1, 1].contourf(xx, yy, v_var[0, :, :, 1], levels = 20)
    
    ax[1, 0].set_xlabel('x')
    ax[1, 1].set_xlabel('x')
    
    ax[0, 0].set_ylabel('y')
    ax[1, 0].set_ylabel('y')
    
    
    ax[0, 0].set_title('u')
    ax[0, 1].set_title('v')
    ax[1, 0].set_title('u (no mean flow)')
    ax[1, 1].set_title('v (no mean flow)')
    
    ax[0, 0].set_aspect('equal')
    ax[0, 1].set_aspect('equal')
    ax[1, 0].set_aspect('equal')
    ax[1, 1].set_aspect('equal')

    plt.savefig('figures/remove_mean_flow.pdf',bbox_inches='tight')


def data_animation(xx,yy,v):  
    
    v_mean = v.mean(axis = 0)
    v_var  = v - v_mean
    
    fig, ax = plt.subplots()

    ax.tick_params(direction= 'out', which= 'major', length= 2, bottom= True, top= True,
                    right= True, left=True, width = 0.7)
    # ax[0].tick_params(direction= 'out', which= 'major', length= 2, bottom= True, top= True,
    #                right= True, left=True, width = 0.7)
    
    
    plt.set_cmap('RdBu_r')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('u')
    
    ax.set_aspect('equal')
        
    
    """
    animation.FuncAnimation(fig, func, frames=None, init_func=None, fargs=None, 
                            save_count=None, *, cache_frame_data=True, **kwargs)[source]¶
    """
    
    def animation_frame(i):
        ax.clear()
        ax.contourf(xx, yy, v_var[i, :, :, 0], levels = 50)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('u')
        ax.set_aspect('equal')
    
    ani = animation.FuncAnimation(fig, animation_frame, 1000, interval=20, blit=False)
    ani.save('figures/flow_data.gif', writer='imagemagick', fps=5)
    # ani.save('figures/flow_data.gif', writer='Pillow', fps=5)

def POD_error(considered_modes, error):
    
    CM2INCH = 0.393701
    GR = (1+np.sqrt(5))/2 # Golden ratio
    
    plt.figure(1,figsize=(15*CM2INCH, 15*CM2INCH/GR))
    
    plt.plot(considered_modes,error[:,0],linestyle ='-', marker ='^', label ='u', 
             color ='r', mfc ='w', alpha= 0.7, markersize = 8)
    plt.plot(considered_modes,error[:,1],linestyle ='-', marker ='o', label ='v', 
             color ='b', mfc ='w', alpha= 0.7, markersize = 8)
    
    plt.xlim([0, 550])
    plt.ylim([0, 1])
    
    plt.tick_params(direction= 'in', which= 'minor', length= 5,bottom= True, top= True,
                    right= True, left= True)
    plt.tick_params(direction= 'in', which= 'major', length= 10, bottom= True, top= True,
                    right= True, left=True)
    plt.minorticks_on()
    
    plt.xlabel(r'Number of modes (-)')
    plt.ylabel(r'Reconstruction error (-)')
    
    plt.legend(fontsize=9)
    
    plt.savefig('figures/POD_reconstruction_error_rate.pdf',bbox_inches='tight') 
    
 
def POD_reconstruction(xx, yy, v_var, v_SVD):
    
    v_SVD_10   = v_SVD["10"].T.reshape(v_var.shape)
    v_SVD_50   = v_SVD["50"].T.reshape(v_var.shape)
    v_SVD_100  = v_SVD["100"].T.reshape(v_var.shape)
    v_SVD_250  = v_SVD["250"].T.reshape(v_var.shape)
    v_SVD_500  = v_SVD["500"].T.reshape(v_var.shape)
    
    fig, ax = plt.subplots(6, 2, sharex = True, sharey = True, figsize = (14, 14))
    plt.set_cmap('RdBu_r')
    # plt.set_cmap('viridis')
    
    ax[0, 0].contourf(xx, yy, v_var[0, :, :, 0], levels = 20)
    ax[0, 1].contourf(xx, yy, v_var[0, :, :, 1], levels = 20)
    
    ax[1, 0].contourf(xx, yy, v_SVD_500[0, :, :, 0], levels = 20)
    ax[1, 1].contourf(xx, yy, v_SVD_500[0, :, :, 1], levels = 20)
    
    ax[2, 0].contourf(xx, yy, v_SVD_250[0, :, :, 0], levels = 20)
    ax[2, 1].contourf(xx, yy, v_SVD_250[0, :, :, 1], levels = 20)
    
    ax[3, 0].contourf(xx, yy, v_SVD_100[0, :, :, 0], levels = 20)
    ax[3, 1].contourf(xx, yy, v_SVD_100[0, :, :, 1], levels = 20)
    
    ax[4, 0].contourf(xx, yy, v_SVD_50[0, :, :, 0], levels = 20)
    ax[4, 1].contourf(xx, yy, v_SVD_50[0, :, :, 1], levels = 20)
    
    ax[5, 0].contourf(xx, yy, v_SVD_10[0, :, :, 0], levels = 20)
    ax[5, 1].contourf(xx, yy, v_SVD_10[0, :, :, 1], levels = 20)
    
    
    ax[5, 0].set_xlabel('x')
    ax[5, 1].set_xlabel('x')
    
    ax[0, 0].set_ylabel('y')
    ax[1, 0].set_ylabel('y')
    ax[2, 0].set_ylabel('y')
    ax[3, 0].set_ylabel('y')
    ax[4, 0].set_ylabel('y')
    ax[5, 0].set_ylabel('y')
    
     
    ax[0, 0].set_title('u original')
    ax[0, 1].set_title('v original')
    
    ax[1, 0].set_title('u POD-500 ($\epsilon$ = 2.83 %)')
    ax[1, 1].set_title('v POD-500 ($\epsilon$ = 2.64 %)')
    
    ax[2, 0].set_title('u POD-250 ($\epsilon$ = 8.91 %)')
    ax[2, 1].set_title('v POD-250 ($\epsilon$ = 8.10 %)')
    
    ax[3, 0].set_title('u POD-100 ($\epsilon$ = 24.57 %)')
    ax[3, 1].set_title('v POD-100 ($\epsilon$ = 22.13 %)')
    
    ax[4, 0].set_title('u POD-50 ($\epsilon$ = 41.79 %)')
    ax[4, 1].set_title('v POD-50 ($\epsilon$ = 36.69 %)')
    
    ax[5, 0].set_title('u POD-10 ($\epsilon$ = 79.08 %)')
    ax[5, 1].set_title('v POD-10 ($\epsilon$ = 70.57 %)')
    
    
    ax[0, 0].set_aspect('equal')
    ax[0, 1].set_aspect('equal')
    ax[1, 0].set_aspect('equal')
    ax[1, 1].set_aspect('equal')
    ax[2, 0].set_aspect('equal')
    ax[2, 1].set_aspect('equal')
    ax[3, 0].set_aspect('equal')
    ax[3, 1].set_aspect('equal')
    ax[4, 0].set_aspect('equal')
    ax[4, 1].set_aspect('equal')
    ax[5, 0].set_aspect('equal')
    ax[5, 1].set_aspect('equal')
    
    plt.savefig('figures/POD_decomposition.pdf',bbox_inches='tight') 
    

def POD_modes(xx, yy, U, S, V, v, mode):
    v_mode = {}


    for i in mode:
        v_mode[str(i+1)] = S[i]* np.outer(U[:,i],V[i,:])

        
    v_mode_1 = v_mode["1"].T.reshape(v.shape)
    v_mode_2 = v_mode["2"].T.reshape(v.shape)
    v_mode_3 = v_mode["3"].T.reshape(v.shape)
    v_mode_4 = v_mode["4"].T.reshape(v.shape)
        
    fig, ax = plt.subplots(4, 2, sharex = True, sharey = True, figsize = (14,10))
    plt.set_cmap('RdBu_r')
    
    ax[0, 0].contourf(xx, yy, v_mode_1[0, :, :, 0], levels = 20)
    ax[0, 1].contourf(xx, yy, v_mode_1[0, :, :, 1], levels = 20)
    ax[1, 0].contourf(xx, yy, v_mode_2[0, :, :, 0], levels = 20)
    ax[1, 1].contourf(xx, yy, v_mode_2[0, :, :, 1], levels = 20)
    ax[2, 0].contourf(xx, yy, v_mode_3[0, :, :, 0], levels = 20)
    ax[2, 1].contourf(xx, yy, v_mode_3[0, :, :, 1], levels = 20)
    ax[3, 0].contourf(xx, yy, v_mode_4[0, :, :, 0], levels = 20)
    ax[3, 1].contourf(xx, yy, v_mode_4[0, :, :, 1], levels = 20)
    
    ax[3, 0].set_xlabel('x')
    ax[3, 1].set_xlabel('x')
    
    ax[0, 0].set_ylabel('y')
    ax[1, 0].set_ylabel('y')
    ax[2, 0].set_ylabel('y')
    ax[3, 0].set_ylabel('y')
    
    ax[0, 0].set_title('u mode 1')
    ax[0, 1].set_title('v mode 1')
    ax[1, 0].set_title('u mode 2')
    ax[1, 1].set_title('v mode 2')
    ax[2, 0].set_title('u mode 3')
    ax[2, 1].set_title('v mode 3')
    ax[3, 0].set_title('u mode 4')
    ax[3, 1].set_title('v mode 4')
    
    ax[0, 0].set_aspect('equal')
    ax[0, 1].set_aspect('equal')
    ax[1, 0].set_aspect('equal')
    ax[1, 1].set_aspect('equal')
    ax[2, 0].set_aspect('equal')
    ax[2, 1].set_aspect('equal')
    ax[3, 0].set_aspect('equal')
    ax[3, 1].set_aspect('equal')
       
    plt.savefig('figures/POD_modes.pdf',bbox_inches='tight')
    
    
def POD_energies(S): 
    

    eigs            = S**2
    eigs_cumulative = eigs.sum() 
    eigs_local_cum  = np.zeros(eigs.shape)

    for i in range(eigs.size):
        eigs_local_cum[i] = eigs[0:i].sum()
    
    
    CM2INCH = 0.393701
    
    fig = plt.figure(figsize=(16*CM2INCH, 8*CM2INCH))
    
    plt.rcParams["font.family"] = "sans"
    
    ax = fig.add_subplot(1, 2, 1)
    
    plt.semilogx(eigs/eigs_cumulative)
    
    plt.xlim([1, 1000])
    plt.ylim([0, 0.1])
    
    plt.tick_params(direction= 'in', which= 'major', length= 4, bottom= True,
            top=True, right= True, left=True, width = 1)
    
    plt.tick_params(direction= 'in', which= 'minor', length= 2, bottom= True,
            top=True, right= True, left=True, width = 1)
    
    plt.grid(which='both')
    
    plt.xlabel('$i$')
    plt.ylabel('$\lambda_i^{}$ / $\sum_{j=1}^m \lambda_j$')
    
    
    ax = fig.add_subplot(1, 2, 2)
    
    plt.semilogx(eigs_local_cum/eigs_cumulative)
    plt.plot([219, 219], [0, 1], color = 'red')
    
    
    plt.xlim([1, 1000])
    plt.ylim([0, 1])
    
    plt.text(125, 0.1, '219', rotation=90, color='red')
    
    plt.tick_params(direction= 'in', which= 'major', length= 4, bottom= True,
            top=True, right= True, left=True, width = 1)
    
    plt.tick_params(direction= 'in', which= 'minor', length= 2, bottom= True,
            top=True, right= True, left=True, width = 1)
    
    plt.grid(which='both')
    
    plt.xlabel('$i$')
    plt.ylabel('$\sum_{k=1}^{i} \lambda_k^{}$ / $\sum_{j=1}^m \lambda_j$')
    
    
    fig.tight_layout()
    
    plt.savefig('figures/POD_eigs.pdf',bbox_inches='tight')
    
  
def AE_plot(AE_1, v_val, xx, yy):
    
    v_val_rec = AE_1.predict(v_val)
    err_rec   = libAE.error_rec(v_val, v_val_rec)
    
    fig, ax = plt.subplots(2, 2, sharex = True, sharey = True, figsize = (10, 4))
    
    plt.set_cmap('RdBu_r')
    
    vmin0 = v_val[-1, :, :, 0].min()
    vmax0 = v_val[-1, :, :, 0].max()
    vmin1 = v_val[-1, :, :, 1].min()
    vmax1 = v_val[-1, :, :, 1].max()
    
    ax[0, 0].contourf(xx, yy, v_val[-1, :, :, 0], vmin = vmin0, vmax = vmax0, levels = 20)
    ax[0, 1].contourf(xx, yy, v_val[-1, :, :, 1], vmin = vmin1, vmax = vmax1,levels = 20)
    ax[1, 0].contourf(xx, yy, v_val_rec[-1, :, :, 0], vmin = vmin0, vmax = vmax0, levels = 20)
    ax[1, 1].contourf(xx, yy, v_val_rec[-1, :, :, 1], vmin = vmin1, vmax = vmax1, levels = 20)
    
    ax[0, 0].set_ylabel('y')
    ax[1, 0].set_ylabel('y')
    ax[1, 0].set_xlabel('x')
    ax[1, 1].set_xlabel('x')
    
    ax[0, 0].set_title('$u$')
    ax[0, 1].set_title('$v$')
    ax[1, 0].set_title("$\\tilde{u}$ (AE-10) - $\epsilon$ = %2.2f" % err_rec[0])
    ax[1, 1].set_title("$\\tilde{v}$ (AE-10) - $\epsilon$ = %2.2f" % err_rec[1])
    
    for axx in ax.flatten():
        axx.set_aspect('equal')
    
    fig.tight_layout()
    plt.savefig('figures/AE-10_rec.pdf',bbox_inches='tight')
    
    
def AE_POD_comp(AE_1, v_val, xx, yy):
        
    v_val_rec    = AE_1.predict(v_val)
    
    
    nt = v_val.shape[0]
    U, S, V = np.linalg.svd(v_val.reshape((nt, -1)).T, full_matrices = False)  

    d = 10

    v_SVD = (U[:,:d] @ np.diag(S)[:d,:d] @ V[:d,:]).T.reshape(v_val.shape)
    
        
    
    err_rec_AE   = libAE.error_rec(v_val, v_val_rec)
    err_rec_SVD  = libAE.error_rec(v_val, v_SVD)
    
    fig, ax = plt.subplots(3, 2, sharex = True, sharey = True, figsize = (10, 7))
    
    plt.set_cmap('RdBu_r')
    
    vmin0 = v_val[-1, :, :, 0].min()
    vmax0 = v_val[-1, :, :, 0].max()
    vmin1 = v_val[-1, :, :, 1].min()
    vmax1 = v_val[-1, :, :, 1].max()
    
    ax[0, 0].contourf(xx, yy, v_val[-1, :, :, 0], vmin = vmin0, vmax = vmax0, levels = 20)
    ax[0, 1].contourf(xx, yy, v_val[-1, :, :, 1], vmin = vmin1, vmax = vmax1,levels = 20)
    ax[1, 0].contourf(xx, yy, v_val_rec[-1, :, :, 0], vmin = vmin0, vmax = vmax0, levels = 20)
    ax[1, 1].contourf(xx, yy, v_val_rec[-1, :, :, 1], vmin = vmin1, vmax = vmax1, levels = 20)
    ax[2, 0].contourf(xx, yy, v_SVD[-1, :, :, 0], vmin = vmin0, vmax = vmax0, levels = 20)
    ax[2, 1].contourf(xx, yy, v_SVD[-1, :, :, 1], vmin = vmin1, vmax = vmax1, levels = 20)
    
    ax[0, 0].set_ylabel('y')
    ax[1, 0].set_ylabel('y')
    ax[2, 0].set_ylabel('y')
    ax[2, 0].set_xlabel('x')
    ax[2, 1].set_xlabel('x')
    
    ax[0, 0].set_title('$u$')
    ax[0, 1].set_title('$v$')
    ax[1, 0].set_title("$\\tilde{u}$ (AE-10) - $\epsilon$ = %2.2f" % err_rec_AE[0])
    ax[1, 1].set_title("$\\tilde{v}$ (AE-10) - $\epsilon$ = %2.2f" % err_rec_AE[1])
    ax[2, 0].set_title("$\\tilde{u}$ (POD-10) - $\epsilon$ = %2.2f" % err_rec_SVD[0])
    ax[2, 1].set_title("$\\tilde{v}$ (POD-10) - $\epsilon$ = %2.2f" % err_rec_SVD[1])
    
    for axx in ax.flatten():
        axx.set_aspect('equal')
    
    fig.tight_layout()
    plt.savefig('figures/AE-10_POD_comp.pdf',bbox_inches='tight')
    
def AEKO_plot(AEKO, v_var, xx, yy):
    v_var_rec, v_var_rec_next = AEKO.predict(v_var)
    err_rec      = libAE.error_rec(v_var, v_var_rec)
    err_rec_next = libAE.error_rec(v_var, v_var_rec_next)
    
    
    fig, ax = plt.subplots(2, 2, sharex = True, sharey = True, figsize = (10, 4))
    
    plt.set_cmap('RdBu_r')
    
    vmin0 = v_var[0, :, :, 0].min()
    vmax0 = v_var[0, :, :, 0].max()
    vmin1 = v_var[1, :, :, 0].min()
    vmax1 = v_var[1, :, :, 0].max()
    
    ax[0, 0].contourf(xx, yy, v_var[0, :, :, 0], vmin = vmin0, vmax = vmax0, levels = 20)
    ax[0, 1].contourf(xx, yy, v_var[1, :, :, 0], vmin = vmin1, vmax = vmax1, levels = 20)
    ax[1, 0].contourf(xx, yy, v_var_rec[0, :, :, 0], vmin = vmin0, vmax = vmax0, levels = 20)
    ax[1, 1].contourf(xx, yy, v_var_rec_next[0, :, :, 0], vmin = vmin1, vmax = vmax1, levels = 20)
    
    ax[0, 0].set_ylabel('y')
    ax[1, 0].set_ylabel('y')
    ax[1, 0].set_xlabel('x')
    ax[1, 1].set_xlabel('x')
    
    ax[0, 0].set_title('$u(t)$')
    ax[0, 1].set_title('$u(t+1)$')
    ax[1, 0].set_title("$\\tilde{u}(t)$ (AE-10) - $\epsilon$ = %2.2f" % err_rec[0])
    ax[1, 1].set_title("$\\tilde{u}(t+1)$ (AE-10) - $\epsilon$ = %2.2f" % err_rec_next[0])
    
    for axx in ax.flatten():
        axx.set_aspect('equal')
    
    fig.tight_layout()
    plt.savefig('figures/AEKO-10_rec.pdf',bbox_inches='tight')

def CNNAE_POD_comp(AE_1, CNNAE, v_val, xx, yy):
        
    # Data adaptation
    v_val_CNN     = v_val[-1:, :, :, 0]
    
    # Data reconstruction (NN)
    v_val_rec     = AE_1.predict(v_val)
    v_val_rec_CNN = CNNAE.predict(v_val_CNN)
    
    # Data reconstruction (POD)
    nt = v_val.shape[0]
    U, S, V = np.linalg.svd(v_val.reshape((nt, -1)).T, full_matrices = False)  

    d = 10

    v_SVD = (U[:,:d] @ np.diag(S)[:d,:d] @ V[:d,:]).T.reshape(v_val.shape)
    
    # Reconstruction error
    err_rec_AE    = libAE.error_rec(v_val, v_val_rec)
    err_rec_CNNAE = libAE.error_rec(v_val_CNN, v_val_rec_CNN)
    err_rec_SVD   = libAE.error_rec(v_val, v_SVD)
    
    # Plotting
    fig, ax = plt.subplots(2, 2, sharex = True, sharey = True, figsize = (10, 4))
    
    plt.set_cmap('RdBu_r')
    
    vmin = v_val[-1, :, :, 0].min()
    vmax = v_val[-1, :, :, 0].max()

    ax[0, 0].contourf(xx, yy, v_val[-1, :, :, 0], vmin = vmin, vmax = vmax, levels = 20)
    ax[0, 1].contourf(xx, yy, v_val_rec_CNN[-1, :, :, 0], vmin = vmin, vmax = vmax,levels = 20)
    ax[1, 0].contourf(xx, yy, v_val_rec[-1, :, :, 0], vmin = vmin, vmax = vmax, levels = 20)
    ax[1, 1].contourf(xx, yy, v_SVD[-1, :, :, 0], vmin = vmin, vmax = vmax, levels = 20)
   
    ax[0, 0].set_ylabel('y')
    ax[1, 0].set_ylabel('y')
    ax[1, 0].set_xlabel('x')
    ax[1, 1].set_xlabel('x')
    
    ax[0, 0].set_title('$u$ - reference')
    ax[0, 1].set_title("$\\tilde{u}$ (CNN-AE-10) - $\epsilon$ = %2.2f" % err_rec_CNNAE[0])
    ax[1, 0].set_title("$\\tilde{u}$ (AE-10) - $\epsilon$ = %2.2f" % err_rec_AE[0])
    ax[1, 1].set_title("$\\tilde{u}$ (POD-10) - $\epsilon$ = %2.2f" % err_rec_SVD[0])
    
    for axx in ax.flatten():
        axx.set_aspect('equal')
    
    fig.tight_layout()
    plt.savefig('figures/CNNAE-10_POD_comp.pdf',bbox_inches='tight')

def CNNHAE_comp(v_val, v_rec_CNNAE, v_rec_CNNHAE, xx, yy):
        
    # Data adaptation
    v_val_CNN     = v_val[-1:, :, :, 0:1]
    

    # Data reconstruction (POD)
    nt = v_val.shape[0]
    U, S, V = np.linalg.svd(v_val.reshape((nt, -1)).T, full_matrices = False)  

    d = 10

    v_SVD = (U[:,:d] @ np.diag(S)[:d,:d] @ V[:d,:]).T.reshape(v_val.shape)
    
    # Reconstruction error
    err_rec_CNNHAE = libAE.error_rec(v_val_CNN, v_rec_CNNHAE)
    err_rec_CNNAE  = libAE.error_rec(v_val_CNN, v_rec_CNNAE)
    err_rec_SVD    = libAE.error_rec(v_val, v_SVD)
    
    # Plotting
    fig, ax = plt.subplots(2, 2, sharex = True, sharey = True, figsize = (10, 4))
    
    plt.set_cmap('RdBu_r')
    
    vmin = v_val[-1, :, :, 0].min()
    vmax = v_val[-1, :, :, 0].max()

    ax[0, 0].contourf(xx, yy, v_val[-1, :, :, 0],        vmin = vmin, vmax = vmax, levels = 20)
    ax[0, 1].contourf(xx, yy, v_rec_CNNAE[-1, :, :, 0],  vmin = vmin, vmax = vmax, levels = 20)
    ax[1, 0].contourf(xx, yy, v_rec_CNNHAE[-1, :, :, 0], vmin = vmin, vmax = vmax, levels = 20)
    ax[1, 1].contourf(xx, yy, v_SVD[-1, :, :, 0],        vmin = vmin, vmax = vmax, levels = 20)
   
    ax[0, 0].set_ylabel('y')
    ax[1, 0].set_ylabel('y')
    ax[1, 0].set_xlabel('x')
    ax[1, 1].set_xlabel('x')
    
    ax[0, 0].set_title('$u$ - reference')
    ax[0, 1].set_title("$\\tilde{u}$ (CNN-AE-10) - $\epsilon$ = %2.2f" % err_rec_CNNAE[0])
    ax[1, 0].set_title("$\\tilde{u}$ (CNN-HAE-10) - $\epsilon$ = %2.2f" % err_rec_CNNHAE[0])
    ax[1, 1].set_title("$\\tilde{u}$ (POD-10) - $\epsilon$ = %2.2f" % err_rec_SVD[0])
    
    for axx in ax.flatten():
        axx.set_aspect('equal')
    
    fig.tight_layout()
    plt.savefig('figures/CNNHAE-10_comp.pdf',bbox_inches='tight')


def reconstruction(v_ref, xx, yy,
                   v_rec_POD = np.zeros((1000, 128, 64, 2)),
                   v_rec_DMD = np.zeros((1000, 128, 64, 2)),
                   v_rec_AE = np.zeros((1000, 128, 64, 2)),
                   v_rec_CNNAE = np.zeros((1000, 128, 64, 2)),
                   v_rec_CNNHAE = np.zeros((1000, 128, 64, 2)),
                   v_rec_HVAE = np.zeros((1000, 128, 64, 2)),
                   v_rec_SAE = np.zeros((1000, 128, 64, 2)),
                   v_rec_VAE = np.zeros((1000, 128, 64, 2))):
  
    plt.rcParams.update({'font.size' : 8})
    plt.rcParams['axes.linewidth'] = 0.7 #set the value globally

    TKE_POD      = libAE.percentageTKE(v_ref[:,:,:,0:1], v_rec_POD[:,:,:,0:1])
    # TKE_DMD      = libAE.percentageTKE(v_ref[:,:,:,0:1], v_rec_DMD[:,:,:,0:1])
    TKE_AE      = libAE.percentageTKE(v_ref[:,:,:,0:1], v_rec_AE[:,:,:,0:1])
    TKE_CNNAE   = libAE.percentageTKE(v_ref[:,:,:,0:1], v_rec_CNNAE[:,:,:,0:1])
    TKE_CNNHAE  = libAE.percentageTKE(v_ref[:,:,:,0:1], v_rec_CNNHAE[:,:,:,0:1])
    TKE_HVAE    = libAE.percentageTKE(v_ref[:,:,:,0:1], v_rec_HVAE[:,:,:,0:1])
    TKE_VAE     = libAE.percentageTKE(v_ref[:,:,:,0:1], v_rec_VAE[:,:,:,0:1])
    # TKE_SAE      = libAE.percentageTKE(v_ref[:,:,:,0:1], v_rec_SAE[:,:,:,0:1])

    GR = (1+np.sqrt(5))/2 # Golden ratio  

    fig, ax = plt.subplots(4, 3, figsize= (16, 8), dpi= 330)

    ax[0, 0].tick_params(direction= 'in', which= 'major', length= 2, bottom= True, top= True,
                    right= True, left=True, width = 0.7)
    ax[0, 1].tick_params(direction= 'in', which= 'major', length= 2, bottom= True, top= True,
                    right= True, left=True, width = 0.7)
    ax[0, 2].tick_params(direction= 'in', which= 'major', length= 2, bottom= True, top= True,
                    right= True, left=True, width = 0.7)
    ax[1, 1].tick_params(direction= 'in', which= 'major', length= 2, bottom= True, top= True,
                    right= True, left=True, width = 0.7)
    ax[1, 2].tick_params(direction= 'in', which= 'major', length= 2, bottom= True, top= True,
                    right= True, left=True, width = 0.7)
    ax[2, 1].tick_params(direction= 'in', which= 'major', length= 2, bottom= True, top= True,
                    right= True, left=True, width = 0.7)
    ax[2, 2].tick_params(direction= 'in', which= 'major', length= 2, bottom= True, top= True,
                    right= True, left=True, width = 0.7)
    ax[3, 1].tick_params(direction= 'in', which= 'major', length= 2, bottom= True, top= True,
                    right= True, left=True, width = 0.7)
    ax[3, 2].tick_params(direction= 'in', which= 'major', length= 2, bottom= True, top= True,
                    right= True, left=True, width = 0.7)

    plt.set_cmap('viridis')

    v_min = v_ref[-1,:,:,0].min()
    v_max = v_ref[-1,:,:,0].max()

    ax[0, 0].clear()
    ax[0, 1].clear()
    ax[0, 2].clear()
    ax[1, 0].clear()
    ax[1, 1].clear()
    ax[1, 2].clear()

    i = -1

    ref = ax[0, 0].contourf(xx, yy, v_ref[i, :, :, 0],  vmin= v_min, vmax = v_max, levels = 20)
    ax[0, 1].contourf(xx, yy, v_rec_AE[i, :, :, 0],     vmin= v_min, vmax = v_max, levels = 20)
    ax[0, 2].contourf(xx, yy, v_rec_CNNAE[i, :, :, 0],  vmin= v_min, vmax = v_max, levels = 20)
    ax[1, 1].contourf(xx, yy, v_rec_CNNHAE[i, :, :, 0], vmin= v_min, vmax = v_max, levels = 20)
    ax[1, 2].contourf(xx, yy, v_rec_HVAE[i, :, :, 0],   vmin= v_min, vmax = v_max, levels = 20)
    ax[2, 1].contourf(xx, yy, v_rec_VAE[i, :, :, 0],    vmin= v_min, vmax = v_max, levels = 20)   
    ax[2, 2].contourf(xx, yy, v_rec_SAE[i, :, :, 0],  vmin= v_min, vmax = v_max, levels = 20)
    ax[3, 1].contourf(xx, yy, v_rec_POD[i, :, :, 0],  vmin= v_min, vmax = v_max, levels = 20)
    ax[3, 2].contourf(xx, yy, v_rec_DMD[i, :, :, 0],  vmin= v_min, vmax = v_max, levels = 20)

    ax[1, 0].set_visible(False)
    ax[2, 0].set_visible(False)
    ax[3, 0].set_visible(False)

    ax[0, 0].plot([0,0],[-1.5,-0.5],'k') # body 1
    ax[0, 0].plot([0,0],[0.5,1.5],'k')   # body 2
    ax[0, 1].plot([0,0],[-1.5,-0.5],'k') # body 1
    ax[0, 1].plot([0,0],[0.5,1.5],'k')   # body 2
    ax[0, 2].plot([0,0],[-1.5,-0.5],'k') # body 1
    ax[0, 2].plot([0,0],[0.5,1.5],'k')   # body 2
    ax[1, 1].plot([0,0],[-1.5,-0.5],'k') # body 1
    ax[1, 1].plot([0,0],[0.5,1.5],'k')   # body 2
    ax[1, 2].plot([0,0],[-1.5,-0.5],'k') # body 1
    ax[1, 2].plot([0,0],[0.5,1.5],'k')   # body 2
    ax[2, 1].plot([0,0],[-1.5,-0.5],'k') # body 1
    ax[2, 1].plot([0,0],[0.5,1.5],'k')   # body 2
    ax[2, 2].plot([0,0],[-1.5,-0.5],'k') # body 1
    ax[2, 2].plot([0,0],[0.5,1.5],'k')   # body 2
    ax[3, 1].plot([0,0],[-1.5,-0.5],'k') # body 1
    ax[3, 1].plot([0,0],[0.5,1.5],'k')   # body 2
    ax[3, 2].plot([0,0],[-1.5,-0.5],'k') # body 1
    ax[3, 2].plot([0,0],[0.5,1.5],'k')   # body 2

    plt.setp(ax[0, 1].get_xticklabels(), visible = False)
    plt.setp(ax[0, 2].get_xticklabels(), visible = False)
    plt.setp(ax[1, 1].get_xticklabels(), visible = False)
    plt.setp(ax[2, 1].get_xticklabels(), visible = False)
    plt.setp(ax[1, 2].get_xticklabels(), visible = False)
    plt.setp(ax[2, 2].get_xticklabels(), visible = False)

    plt.setp(ax[0, 1].get_yticklabels(), visible = False)
    plt.setp(ax[0, 2].get_yticklabels(), visible = False)
    plt.setp(ax[1, 2].get_yticklabels(), visible = False)
    plt.setp(ax[2, 2].get_yticklabels(), visible = False)
    plt.setp(ax[3, 2].get_yticklabels(), visible = False)

    ax[0, 0].set_aspect('equal')
    ax[0, 1].set_aspect('equal')
    ax[0, 2].set_aspect('equal')
    ax[1, 0].set_aspect('equal')
    ax[1, 1].set_aspect('equal')
    ax[1, 2].set_aspect('equal')
    ax[2, 0].set_aspect('equal')
    ax[2, 1].set_aspect('equal')
    ax[2, 2].set_aspect('equal')
    ax[3, 0].set_aspect('equal')
    ax[3, 1].set_aspect('equal')
    ax[3, 2].set_aspect('equal')

    ax[0, 0].set_ylabel('y')
    ax[1, 1].set_ylabel('y')
    ax[2, 1].set_ylabel('y')
    ax[3, 1].set_ylabel('y')

    ax[0, 0].set_xlabel('x')
    ax[3, 1].set_xlabel('x')
    ax[3, 2].set_xlabel('x')

    ax[0, 0].set_title('Reference')

    ax[0, 1].set_title(r'AE-10 (%.2f %%)'%(TKE_AE))
    ax[0, 2].set_title(r'CNN AE-10 (%.2f %%)'%(TKE_CNNAE))
    ax[1, 1].set_title(r'CNN HAE-10 (%.2f %%)'%(TKE_CNNHAE))
    ax[1, 2].set_title(r'CNN HVAE-10 (%.2f %%)'%(TKE_HVAE))
    ax[2, 1].set_title(r'CNN $\beta$-VAE-10 (%.2f %%)'%(TKE_VAE))
    ax[2, 2].set_title(r'Symmetric AE-10 (- %)')
    ax[3, 1].set_title(r'POD-10 (%.2f %%)'%(TKE_POD))
    ax[3, 2].set_title(r'DMD (- %)')

    # plt.colorbar(ref, ax = ax[0,0], shrink = 0.8, orientation='horizontal')

    plt.subplots_adjust(wspace=0.1, hspace=0.1)