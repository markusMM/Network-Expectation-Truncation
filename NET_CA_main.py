# -*- coding: utf-8 -*-
"""

    Binary Sparse Coding (BSC) feeding a Multi Layer Perceptron (MLP)

From a given limited number of data points x from a sparse data set X, BSC is
trained to generate more data points for a MLP to train, test and validate.

BSC and MLP has to given by a parameter file.
The parameters has to fit the given model given in the parameter file.
Ground truth (GT) parameters has to be given correctly.

BSC:
    learns using expectation truncation method (ET) [1]
    uses the PyLib MLOL [1] version


MLP:
    u

cites:

    [1] M. Henniges, G. Puertas, J. Bornschein, J. Eggert, and J. Lücke (2010).
        Binary Sparse Coding. Proc. LVA/ICA 2010, LNCS 6365, 450-457.y

    [2]


@author: Markus Meister, University of Oldenburg (Olb), Germany
"""
#%%
import tables
import operator as op
import os
import sys
os.stat('pylib')
sys.path.append("pylib")
sys.path.append("*/utils")
#import timeit
#import cPickle
#import numba as nb
import numpy as np
import pandas as pd
import time
import h5py
import distutils.dir_util
import matplotlib.pyplot as plt
#import sklearn_theano as skth
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.model_selection import train_test_split
#import theano.tensor as T
#from numba import guvectorize
#from ReportInterface import ReportInterface as dic2h5
#from scipy.stats import bernoulli
#from mpi4py import MPI
#from sklearn import metrics
try:
    from scipy import comb
except ImportError:
    from scipy.misc import comb

try:
    from mpi4py import MPI
except ImportError:
    sys.path.append('mpi4py/src/mpi4py')
    try:
        from mpi4py import MPI
    except ImportError:
        try:
            from mpi4py.src.mpi4py import MPI
        except ImportError:
            print('Error: No module mpi4py installed!')

from pulp.utils import create_output_path
from pulp.utils.parallel import pprint, stride_data

from pulp.utils.barstest import generate_bars_dict
from pulp.utils.datalog import dlog, StoreToH5, TextPrinter, StoreToTxt
import pulp.utils.tracing as tracing

from pulp.em import EM
from pulp.em.annealing import LinearAnnealing
from pulp.utils.preprocessing import ZCA

#import string

from pulp.functions import \
softmax,gammaint,\
sav,lod,\
getSelection,table_prep,\
RecNegRMSE,RecNegMSAE,MLPrecon,\
RecPerCent,RecAccScre#,\
#Vdot

from utils.score_fcns import rmsle

#%%--standardizer--#############################################################

def std_mlp():
    """Makes a standard MLPCLassifier object"""

    mlp = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
                    beta_1=0.9, beta_2=0.999, early_stopping=True,
                    epsilon=1e-08, hidden_layer_sizes=(2048,103,),
                    learning_rate='constant', learning_rate_init=0.001,
                    max_iter=250, momentum=0.9, nesterovs_momentum=True,
                    power_t=0.5, random_state=1, shuffle=True, solver='adam',
                    tol=1e-05, validation_fraction=0.3,
                    verbose=False, warm_start=False
                    )
    return mlp

def std_bsc(params):
    """Makes a standard BSC prob. model obj."""

    size = params.get('size', 5)         # width / height of bars images
    D = params.get('D', size**2)         # observed dimensionality
    H = params.get('H', 2*size)          # latent dimensionality
    Hprime = params.get('Hprime',  size) # truncated latent dimensionality
    gamma = params.get('gamma',int(size/2)) # expected max. no. causes active

    from pulp.em.camodels.bsc_et import BSC_ET
    model = BSC_ET(D, H, Hprime, gamma)

    return model

#%%###########--Main--###########################################################
if __name__ == "__main__":
    import argparse

    comm = MPI.COMM_WORLD

    if len(sys.argv) < 2:
        pprint("Usage: %s <parameter-file>" % sys.argv[0])
        pprint()
        exit(1)

    param_fname = sys.argv[1]

    params = {}
    execfile('%s' % param_fname, params)
    
    # Extract some parameters
    SCALE = params.get('SCALE', 10)      # SCALE factor of the data
    pprint("SCALE:")
    pprint(SCALE)
    N = params.get('N', 500)             # no. of datapoint in the testset
    #size = params.get('size', 5)        # width / height of bars images
    D = params.get('D', 25)              # observed dimensionality
    pi = params.get('pi', 1./np.sqrt(D)) # prob. for a bar to be active
    sigma = params.get('sigma', .35)     # normed noise strength
    H = params.get('H', 2*np.sqrt(D))    # latent dimensionality
    Hprime = params.get('Hprime', H//2)  # truncated H
    gamma = params.get('gamma', Hprime//2) # expected no. causes per data point
    neg_bars = params.get('neg_bars', False) # if negative 'bars' for tests
    sigma_init  = params.get('sigma_init'   ,  None) # variable inits
    pi_init     = params.get('pi_init'      ,  None) #  -----||-----
    W_init      = params.get('W_init'       ,  None) #  -----||-----
    model = params.get('model',std_bsc)  # the actual generative model
    mod_fun = params.get('mod_fun','bsc_et')# which model is model
    mlp = params.get('mlp',std_mlp)      # the actual MLP classifier
    batches = params.get('batches',1)    # no. data batches
    scorstor = params.get('scorstor',0)  # if storing score in txt
    iterstor = params.get('iterstor',0)  # if stoting iterations in txt
    # if having a noiseless validation
    noiseless_valid = params.get('noiseless_valid',1)
    datatype = params.get('datatype','BARS') # sting what data we use here
    my_data = params.get('my_data',None) # iff the data is already given as struct
    te_data = params.get('te_data',None) # iff the data is already given as struct
    va_data = params.get('va_data',None) # iff the data is already given as struct
    
    sc_predict = params.get('sc_predict',False)
    
    pronoise = params.get('pronoise',True)
    
    PeakAmp = SCALE
    
    grayscale = params.get('grayscale',False)
    
    save_recon = params.get('save_recon',False)
    
    gen_datas = params.get('gen_datas',True) # if generating the data for mlp with BSC
    
    scorstrg = params.get('scorstrg','accscre') # if saving scores (internal - no api atm)
    
    zca_flag = params.get('zca_flag',0)         # if using zca in preprocessing    
    bsc_classify = params.get('bsc_classify',1) # if using BSC as classifier
    mlp_classify = params.get('mlp_classify',0) # if using MLP as classifier
    
    full_em_classify = params.get('full_em_classify',0) # if using a full em step for reconstruction
    
    #for grid search
    R   = params.get('R',   None)
    Run = params.get('Run', None)
    
    # for bootstrapping
    mlp_Ks_ratio = params.get('mlp_ratio',.0)
    mlp_et_train = params.get('gen_train',.0)
    mlp_et_fullI = params.get('mlp_lin_i',.8)
    mlp_et_fullN = params.get('mlp_lin_n',.8)
    
    # MLP after training params
    Ntr = params.get('Ntr', 15000)       # training set size
    Nte = params.get('Nte',  7500)       # test set size
    Nva = params.get('Nva',  2500)       # noiseless test set size

    # annealing params
    Iters = params.get('Iters',  50)     # (max) iterations
    Tmax  = params.get('Tmax',  1.2)     # max annealing dT
    priT  = params.get('priT',    1)     # if using prior annealing

    #flags
    impaint = params.get('impaint', 0)   # if doing an impainting test
    predict = params.get('predict', 0)   # if predicting causes from MLP
    figFlag = params.get('figFlag', 0)   # if creating/saving figures
    
    K_mlp   = params.get('K_mlp',0)      # if using MLP as K-select function

    # if saving the binary labeled data
    sav_dat_flg = params.get('sav_dat_flg',0)

    # Ground truth parameters -- only used for generation
    params_gt = params.get('params_gt')  # Ground truth param
    
    # if deleting the h5 file later
    del_h5 = params.get('del_h5',False)
    
    # top-layer
    top_layer = params.get('top_layer',None)
    
    # load id, if set
    id_path = params.get('id_path',False)
    
    # Create output path
    output_path = params.get('h5path','not defined')+'/'
    
    if output_path == 'not defined/':
        output_path = create_output_path(param_fname)+'/'
    
    if id_path:
        output_path += '_'
        output_path += str(id_path)
    
    try:
        distutils.dir_util.mkpath(output_path)
    except:
        try:
            os.stat(output_path)
        except:
            os.mkdir(output_path)

    #%% -- BSC ET: Header --
    
    # Disgnostic output
    pprint("="*40)
    pprint(" Running %s (%d parallel processes)" % (str.upper(mod_fun),comm.size))
    pprint("  size of training set:   %d" % N)
    pprint("  size of %s patches:     %d" % (datatype,D))
    pprint("  number of hiddens:      %d" % H)
    pprint("  saving results to:      %s" % output_path)
    pprint()

    #%% -- BSC ET: Data --
    pprint(' Loading Data . . .  ')
    
    if datatype=='BARS' or datatype=='BARSTEST':
        
        params_te = params_gt
        for prm in params_te:
            if prm+'_test' in params.keys():
                params_te[prm] = params[prm+'_test']
                pprint('%s_test = %.02f' %(prm,params[prm+'_test']))
        
        # Generate 'bars' test data
        te_data = model.generate_data(params_te, Nte // comm.size)
        va_data = model.generate_data(params_gt, Nva // comm.size)
        my_data = model.generate_data(params_gt, N   // comm.size)
        va_data['y'] = va_data['S']
    
    if datatype=='sheffield' or datatype[:5]=='sheff':

        my_N = N   // comm.size # fractions of full dataset
        te_N = Nte // comm.size
        va_N = Nva // comm.size
        
        # init data
        my_data = {}
        my_data['y'] = np.empty([comm.size,my_N,D])
        my_data['S'] = np.empty([comm.size,my_N,D])
#        my_data['s'] = np.empty([comm.size,my_N,H])
#        my_data['l'] = np.empty([comm.size,my_N])
        te_data = {}
        te_data['y'] = np.empty([comm.size,te_N,D])
        te_data['S'] = np.empty([comm.size,te_N,D])
#        te_data['s'] = np.empty([comm.size,te_N,H])
#        te_data['l'] = np.empty([comm.size,te_N])
        va_data = {}
        va_data['y'] = np.empty([comm.size,va_N,D])
        va_data['S'] = np.empty([comm.size,va_N,D])
#        va_data['s'] = np.empty([comm.size,va_N,H])
#        va_data['l'] = np.empty([comm.size,va_N])

        if comm.rank == 0:
            
            
            no_ims = 4
            
            imf = {}
            for im in range(no_ims):
                
                #read in data
                imf[im] = plt.imread(\
                   'data/sheffield_sky/Sheffield_Skyline_%d.jpg' \
                   %(im+1)\
                   ,0)/255
            
            DDx,DDy,C = imf[0].shape
            
            sys.path.append('dataata/sheffield_sky/')

            var = 95 # variance kept in percent
            p = int(np.sqrt(D/C)) #patch size
            
            
            data = np.zeros((N+Nte+Nva,D//C,C),dtype='float64')
            #ims = np.zeros([no_ims,DDx,DDy,C])
            for im in range(no_ims):
                ims = imf[im]
                
                this_Ntr = N   // no_ims
                this_Nte = Nte // no_ims
                this_Nva = Nva // no_ims
                total_N = this_Ntr+this_Nte+this_Nva
            
                #crop patches
                indw = np.random.randint(0,ims.shape[1]-p,total_N)
                indh = np.random.randint(0,ims.shape[0]-p,total_N)
                for i in range(total_N):
                	data[i*im] = ims[indh[i]:indh[i]+p,indw[i]:indw[i]+p].reshape(D//C,C)
            
            if grayscale:
                data = data.mean(axis=-1)
                C=1
                D=D//C
            
            #px,py = data.shape
            wdata = np.zeros_like(data)
            if zca_flag:
                for c in range(C):
                    #whiten files
                    zca = ZCA()
                    zca.fit(data[:1000,c],var=var/100.)
                    wdata[:,c] = zca.transform(data[:,c])
            
            if not grayscale:
                wdata = wdata.reshape([N+Nte+Nva,D])
                C=1
            wdata = wdata.squeeze()
            #print(wdata.shape)
            
            my_data['S'] = wdata[:N].reshape(comm.size,my_N,D)
            my_data['y'] = my_data['S'] #+ (params.get('dnoise',0)) * np.random.randn(my_data['S'].shape)
            te_data['S'] = wdata[N:Nte+N].reshape(comm.size,te_N,D)
            te_data['y'] = te_data['S'] #+ (params.get('dnoise',0)) * np.random.randn(te_data['S'].shape)
            va_data['S'] = wdata[N+Nte:N+Nte+Nva].reshape(comm.size,va_N,D)
            va_data['y'] = va_data['S'] #+ (params.get('dnoise',0)) * np.random.randn(va_data['S'].shape)

        else:

            my_data['y'] = np.empty([comm.size,my_N,D])
            my_data['S'] = np.empty([comm.size,my_N,D])
            te_data['y'] = np.empty([comm.size,te_N,D])
            te_data['S'] = np.empty([comm.size,te_N,D])
            va_data['y'] = np.empty([comm.size,va_N,D])
            va_data['S'] = np.empty([comm.size,va_N,D])            
        
        my_data['y'] = comm.scatter(my_data['y'], root=0)
        my_data['S'] = comm.scatter(my_data['S'], root=0)
        my_data['s'] = np.zeros([my_N,H])
        te_data['y'] = comm.scatter(te_data['y'], root=0)
        te_data['S'] = comm.scatter(te_data['S'], root=0)
        te_data['s'] = np.zeros([te_N,H])
        va_data['y'] = comm.scatter(va_data['y'], root=0)
        va_data['S'] = comm.scatter(va_data['S'], root=0)
        va_data['s'] = np.zeros([te_N,H])
        
        print(my_data['y'].shape)
    
    if datatype=='MNIST':
        
        PeakAmp += 255
        
        # init data
        if zca_flag:
            zca = ZCA()
        
        # load in some MNIST data
        import cPickle, gzip
        
        sys.path.append('data/')
        try:
            f = open('data/mnist.pkl', 'rb')
            train_set, valid_set, test_set = cPickle.load(f)
            f.close()
        except:
            f = gzip.open('data/mnist.pkl.gz', 'rb')
            train_set, valid_set, test_set = cPickle.load(f)
            f.close()
        
        N   = int(train_set[1].shape[0])   and N
        Nte = int(test_set[1].shape[0])    and Nte
        Nva = int(valid_set[1].shape[0])   and Nva
        
        my_N = N    // comm.size
        te_N = Nte  // comm.size
        va_N = Nva  // comm.size
        
        par_tr_y = train_set[0][comm.rank*my_N:(comm.rank+1)*my_N]
        par_tr_l = train_set[1][comm.rank*my_N:(comm.rank+1)*my_N]
        
        par_te_y = test_set[0][comm.rank*te_N:(comm.rank+1)*te_N]
        par_te_l = test_set[1][comm.rank*te_N:(comm.rank+1)*te_N]
        
        par_va_y = valid_set[0][comm.rank*va_N:(comm.rank+1)*va_N]
        par_va_l = valid_set[1][comm.rank*va_N:(comm.rank+1)*va_N]
        
        if zca_flag:
            zca.fit(train_set[0],.95)
            
            my_data = MNISTprep([par_tr_y,par_tr_l], my_N, SCALE=SCALE, zca_flag=1, zca=zca)
            te_data = MNISTprep([par_te_y,par_te_l], te_N, SCALE=SCALE, zca_flag=1, zca=zca)
            va_data = MNISTprep([par_va_y,par_va_l], va_N, SCALE=SCALE, zca_flag=1, zca=zca)
        else:
            my_data = MNISTprep([par_tr_y,par_tr_l], my_N, SCALE=SCALE, zca_flag=0)
            te_data = MNISTprep([par_te_y,par_te_l], te_N, SCALE=SCALE, zca_flag=0)
            va_data = MNISTprep([par_va_y,par_va_l], va_N, SCALE=SCALE, zca_flag=0)
        
        train_set = None
        test_set  = None
        valid_set = None
    
    if datatype=='MNIST_patch': ###############################################
        
        PeakAmp += 255
        
        # init data
        if zca_flag:
            zca = ZCA()
        
        # load in some MNIST data
        import cPickle, gzip
        
        sys.path.append('data/')
        try:
            f = open('data/mnist.pkl', 'rb')
            train_set, valid_set, test_set = cPickle.load(f)
            f.close()
        except:
            f = gzip.open('data/mnist.pkl.gz', 'rb')
            train_set, valid_set, test_set = cPickle.load(f)
            f.close()
        
        my_N = N    // comm.size
        te_N = Nte  // comm.size
        va_N = Nva  // comm.size
        
        p = np.sqrt(D).astype(int)
        
        ims = train_set[0]
        nms = train_set[1]
        
        ims = np.reshape(
                ims,
                [
                        ims.shape[0],
                        np.sqrt(ims.shape[1]).astype(int),
                        np.sqrt(ims.shape[1]).astype(int)
                 ]
                )
        
        #crop patches
        data = np.zeros((my_N+te_N+va_N,D),dtype='float64')
        name = np.zeros((my_N+te_N+va_N),dtype='int16')
        indw = np.random.randint(0,ims.shape[2]-p,N+Nte+Nva)
        indh = np.random.randint(0,ims.shape[1]-p,N+Nte+Nva)
        indi = np.random.randint(0,ims.shape[0],N+Nte+Nva)
        for i,ind in enumerate(indi):
         data[i] = ims[ind,indh[i]:indh[i]+p,indw[i]:indw[i]+p].reshape(D)
         name[i] = nms[ind]
        
        #whiten files
        if zca_flag:
            zca = ZCA()
            zca.fit(data[:1000],var=var/100.)
            wdata = zca.transform(data)
            pprint(wdata.shape)
        else:
            wdata = data.copy()
        
        my_data = {}
        te_data = {}
        va_data = {}
        my_data['S'] = wdata[:my_N]
        my_data['l'] =  name[:my_N]
        my_data['y'] = my_data['S']
        te_data['S'] = wdata[my_N:te_N+my_N]
        te_data['l'] =  name[my_N:te_N+my_N]
        te_data['y'] = te_data['S']
        va_data['S'] = wdata[my_N+te_N:my_N+te_N+va_N]
        va_data['l'] =  name[my_N+te_N:my_N+te_N+va_N]
        va_data['y'] = va_data['S']
        
        
        train_set = None
        test_set  = None
        valid_set = None
                    
    if datatype=='IMG' or datatype[0:2]=='ima':
        
        PeakAmp += 255
        
        my_N = N   // comm.size # fractions of full dataset
        te_N = Nte // comm.size
        va_N = Nva // comm.size
        
        # init data
        my_data = {}
        my_data['y'] = np.empty([comm.size,my_N,D])
        my_data['S'] = np.empty([comm.size,my_N,D])
#        my_data['s'] = np.empty([comm.size,my_N,H])
#        my_data['l'] = np.empty([comm.size,my_N])
        te_data = {}
        te_data['y'] = np.empty([comm.size,te_N,D])
        te_data['S'] = np.empty([comm.size,te_N,D])
#        te_data['s'] = np.empty([comm.size,te_N,H])
#        te_data['l'] = np.empty([comm.size,te_N])
        va_data = {}
        va_data['y'] = np.empty([comm.size,va_N,D])
        va_data['S'] = np.empty([comm.size,va_N,D])
#        va_data['s'] = np.empty([comm.size,va_N,H])
#        va_data['l'] = np.empty([comm.size,va_N])

        if comm.rank == 0:

            sys.path.append('data/')

            var = 95 # variance kept in percent
            p = int(np.sqrt(D)) #patch size
            ifname = 'data/natims_conv_1700.npy'   #input file
            ofname = 'data/natims_conv_1700_p{}_{}var.npz'.format(p,var) #output file

            #read in data
            ims = np.load(ifname)

            #crop patches
            data = np.zeros((N+Nte+Nva,D),dtype='float64')
            indw = np.random.randint(0,ims.shape[2]-p,N+Nte+Nva)
            indh = np.random.randint(0,ims.shape[1]-p,N+Nte+Nva)
            indi = np.random.randint(0,ims.shape[0],N+Nte+Nva)
            for i,ind in enumerate(indi):
            	data[i] = ims[ind,indh[i]:indh[i]+p,indw[i]:indw[i]+p].reshape(D)
            
            #whiten files
            zca = ZCA()
            zca.fit(data[:1000],var=var/100.)
            wdata = zca.transform(data)
            pprint(wdata.shape)
            
            my_data['S'] = wdata[:N].reshape(comm.size,my_N,D)
            my_data['y'] = my_data['S'] #+ (params.get('dnoise',0)) * np.random.randn(my_data['S'].shape)
            te_data['S'] = wdata[N:Nte+N].reshape(comm.size,te_N,D)
            te_data['y'] = te_data['S'] #+ (params.get('dnoise',0)) * np.random.randn(te_data['S'].shape)
            va_data['S'] = wdata[N+Nte:N+Nte+Nva].reshape(comm.size,va_N,D)
            va_data['y'] = va_data['S'] #+ (params.get('dnoise',0)) * np.random.randn(va_data['S'].shape)

        else:

            my_data['y'] = np.empty([comm.size,my_N,D])
            my_data['S'] = np.empty([comm.size,my_N,D])
            te_data['y'] = np.empty([comm.size,te_N,D])
            te_data['S'] = np.empty([comm.size,te_N,D])
            va_data['y'] = np.empty([comm.size,va_N,D])
            va_data['S'] = np.empty([comm.size,va_N,D])            
        
        my_data['y'] = comm.scatter(my_data['y'], root=0)
        my_data['S'] = comm.scatter(my_data['S'], root=0)
        my_data['s'] = np.zeros([my_N,H])
        te_data['y'] = comm.scatter(te_data['y'], root=0)
        te_data['S'] = comm.scatter(te_data['S'], root=0)
        te_data['s'] = np.zeros([te_N,H])
        va_data['y'] = comm.scatter(va_data['y'], root=0)
        va_data['S'] = comm.scatter(va_data['S'], root=0)
        va_data['s'] = np.zeros([te_N,H])
        
        pprint(my_data['y'].shape)

    if 'anneal' in params:
        anneal = params.get('anneal')
    else:
        if 'anneal' not in locals():
            anneal = LinearAnnealing(50)
            anneal['T'] = [(0, (1+1.6)), (.8, 1.)]
            anneal['Ncut_factor'] = [(0,0.),(.9,1.)]
            anneal['anneal_prior'] = True
    
    pprint(' --> Done! ')
    
    #%% -- BSC ET: Init

    # Configure DataLogger
    print_list = ('T', 'Q', 'pi', 'sigma', 'N_use', 'MAE')
    store_list = ('*')
    dlog.set_handler(print_list, TextPrinter)
    dlog.set_handler(print_list, StoreToTxt, output_path +'/'+ params.get('txt_nm','terminal.txt'))
    dlog.set_handler(store_list, StoreToH5,  output_path +'/'+ params.get('h5f_nm','result.h5'))
    
    if mod_fun == 'MoG' or mod_fun == 'MoP':
        model.to_learn = ['pies','W','sigmas_sq']
    
    model_params = model.standard_init(my_data)
    
    if mod_fun == 'MoG' or mod_fun == 'MoP':
        assert np.isfinite(model_params['sigmas_sq']).all()
    
    if sigma_init != None:
        model_params['sigma'] = sigma_init*SCALE
        pprint("sigma_init: %f" %sigma_init)
    

    if pi_init != None:
        model_params['pi'] = pi_init    
        pprint("pi_init: %f" %pi_init)
    
    
    if W_init != None:
        model_params['W'] = W_init 
        pprint("W_init: %f" %W_init)
    
    # Create and start EM
    em = EM(model=model, anneal=anneal)
    em.data = my_data
    em.lparams = model_params
    
    #learn param check
    sd_learn = model.to_learn
    my_learn = []
    #gflg=False
    for k in sd_learn:
        gflg=params.get('%s_fix' % k, False)
        if not gflg:
            my_learn.append(k)
            print('EM learns %s' % k)
    
    #%% top-layer
    
#    if top_layer != None:
#        # set top-layer as attribute of the model
#        em.model.top_layer = top_layer
#        # reset the classification for hidden layer
#        if bsc_classify or mlp_classify:
#            pprint('WARNING: No classification test for hidden layers!\n')
#            if bsc_classify:
#                pprint('BSC has no direct labels of the data!\n')
#            if mlp_classify:
#                pprint('MLP has no direct labels of the data!\n')
#                
#        bsc_classify = 1
#        mlp_classify = 1
    
    #%% -- BSC ET: Run --
    
    #for fix params
    model.to_learn = my_learn
    
    dlog_flg = 0
    if bsc_classify or gen_datas:
        em.run()
        dlog_flg = 1
    
    new_model_params = em.lparams
    model.to_learn = []
    
    
    h5path = output_path +'/'+ params.get('h5f_nm','result.h5')
    
    #%% -- New Seeds (desynced) --
    
    np.random.seed(np.random.get_state()[1][0] + 1)
    mlp.random_state = np.random.get_state()[1][0] + 1
    if hasattr(em.model,'mlp'):
        em.model.mlp.random_state = np.random.get_state()[1][0] + 1

    #%% -- BSC ET: Classify --
    test_data_complete = True
    try:
        a = te_data
        b = va_data
        a = None
        b = None
    except:
        test_data_complete = False
    
    if (bsc_classify or hasattr(em.model,'mlp')) and not test_data_complete:
        
        te_data = em.model.generate_data(params_gt,Nte//comm.size)
        
        va_data = em.model.generate_data(params_gt,Nva//comm.size)
        va_data['y'] = va_data['S']
    
    if bsc_classify:

        ann = LinearAnnealing(1)
        ann['T'] = [(0., 1.), (1., 1.)]
        ann['Ncut_factor'] = [(0.,0.),(1.,0.)]
        ann['anneal_prior'] = anneal['anneal_prior']
        ann['E_s_Flag'] = True
        ann['mlp_ratio'] = 1

        em.anneal   = ann
        
        if full_em_classify:
            em.model.Hprime = em.model.H
            em.model.gamma = em.model.H
            em.model.new_state_matrix(em.model.gamma,em.model.Hprime)
        
        #----train----
        em.model.new_step = 0

        #tr_data = em.model.generate_data(params_gt,N//comm.size)

        em.data = my_data
        
        em.model.eflg = 1
        em.step()
        
        #expect = em.expect()
        #print(expect.shape)
        
        #----test----
        em.model.new_step = 1

        em.data     = te_data
        
        BSCtetime = time.clock()
        em.step()
        BSCtetime = time.clock() - BSCtetime
        pprint('BSC test reconstruction time: %fs\n' %BSCtetime)

        #----valid----
        em.model.new_step = 2
        
        em.data     = va_data
        
        em.step()

    #%% -- BSC ET: Results --

    if comm.rank == 0 and dlog_flg:
        with tables.open_file(h5path, 'r+') as data_h5:
            W = data_h5.root.W.read()
            pi = data_h5.root.pi.read()
            sigma = data_h5.root.sigma.read()
            #mu = data_h5.root.mu.read()
            Ny = data_h5.root.N_use.read()
            try:
                L = data_h5.root.L.read()
            except:
                L = data_h5.root.Q.read()
            T = data_h5.root.T.read()
            n = data_h5.root.max_step.read()
            Ncut_factor = data_h5.root.Ncut_factor.read()
            pos = data_h5.root.position.read()
            # if expectation value get successfully integrated
            # ..one day ;)
            try:
                e_s = data_h5.root.E_s.read()
            except:
                impaint = 0


            if bsc_classify:
                e_tr = data_h5.root.E_s_0.read()[-1]
                #W_tr = data_h5.root.W.read()[-1]
                e_te = data_h5.root.E_s_1.read()[-1]
                #W_te = data_h5.root.W.read()[-2]
                e_va = data_h5.root.E_s_2.read()[-1]
                #W_va = data_h5.root.W.read()[-3]
            
            u = W[Iters-1]
            p = pi[Iters-1]
            s = sigma[Iters-1]
            llh = L[Iters-1]
            
        data_h5.close()
        print(W.shape)
        for j in range(1,4):
            if (W[-j] == new_model_params['W']).all():
                print('Check W: W[-%d]'%j)
    
    [D, H, pi_, sigma_init, N, hi,h] = (
    op.itemgetter('D', 'H', 'pi', 'sigma', 'N', 'gamma','Hprime')(params_gt)
    )

    if hasattr(em.model,'mlp'):
        vali_frac = mlp.validation_fraction
        mlp = em.model.mlp
        mlp.validation_fraction = vali_frac
        mlp.early_stopping = True

   # [W] = op.itemgetter('W')(em.lparams)
   # [T, L, pi, sigma, N, MAE] = op.itemgetter('T', 'Q', 'pi', 'sigma', 'N', 'MAE')(dlog)
    
#%% -- BSC Classify Scores --
    
    if bsc_classify or hasattr(em.model,'mlp'):
        try:
            S_tr = np.array(comm.allgather(my_data['S']))
        except:
            S_tr = my_data['S']
            pprint('AAAAHHH! . . . training data unsynced!! ;(')
        S_te = np.array(comm.allgather(te_data['S']))
        S_va = np.array(comm.allgather(va_data['S']))
        
        S_tr = np.reshape(S_tr,[comm.size*S_tr.shape[1],S_tr.shape[2]])
        S_te = np.reshape(S_te,[comm.size*S_te.shape[1],S_te.shape[2]])
        S_va = np.reshape(S_va,[comm.size*S_va.shape[1],S_va.shape[2]])
        
        #
        y_tr = np.array(comm.allgather(my_data['y']))
        y_tr = np.reshape(y_tr,[comm.size*y_tr.shape[1],y_tr.shape[2]])
        y_te = np.array(comm.allgather(te_data['y']))
        y_te = np.reshape(y_te,[comm.size*y_te.shape[1],y_te.shape[2]])
        y_va = np.array(comm.allgather(va_data['y']))
        y_tr = np.reshape(y_va,[comm.size*y_va.shape[1],y_va.shape[2]])
        
        #s_tr = np.array(my_data['s'].astype(int))
        s_tr = np.array(comm.allgather(my_data['s'].astype(int)))
        s_tr = np.reshape(s_tr,[comm.size*s_tr.shape[1],s_tr.shape[2]])
        
        #s_te = np.array(te_data['s'].astype(int))
        s_te = np.array(comm.allgather(te_data['s'].astype(int)))
        s_te = np.reshape(s_te,[comm.size*s_te.shape[1],s_te.shape[2]])
        
        #s_va = np.array(va_data['s'].astype(int))
        s_va = np.array(comm.allgather(va_data['s'].astype(int)))
        s_va = np.reshape(s_va,[comm.size*s_va.shape[1],s_va.shape[2]])
        
        pprint(S_tr.shape)
        pprint(np.max(S_te))
    
    #%% RF regression greedy
    
    if sc_predict:
        if not bsc_classify:
            x_pred = em.model.mlp.predict(my_data['y'])
        else:
            x_pred = \
            em.model.generate_from_hidden(new_model_params,{'s':e_tr})['S']
        y_pred = x_pred[:,-len(targets)]
        
        tr_RMSE = np.sqrt(np.mean(
                ( y_pred - my_data['l'] )**2
                ))
        
        pprint('done!')
        pprint('ff-SC tr_RMSE : %0.4f' %tr_RMSE)
        
        if not bsc_classify:
            x_pred = em.model.mlp.predict(te_data['y'])
        else:
            x_pred = \
            em.model.generate_from_hidden(new_model_params,{'s':e_te})['S']
        y_pred = x_pred[:,-len(targets)]
        
        te_RMSE = np.sqrt(np.mean(
                ( y_pred - te_data['l'] )**2
                ))
        
        pprint('done!')
        pprint('ff-SC te_RMSE : %0.4f' %te_RMSE)
    
    if params.get('RF_classify',0):
        
        from sklearn.ensemble import RandomForestRegressor
        
        pprint('building RF . . .')
        
        flat_rf = RandomForestRegressor(
                n_estimators = 200,
                criterion = 'mse',
                oob_score = True,
                min_samples_leaf = 3,
                max_features = 'auto',
                n_jobs = 4,
                )
        
        pprint('training. . .')
        
        if not bsc_classify:
            x_pred = em.model.mlp.predict(my_data['y'])
        else:
            x_pred = \
            em.model.generate_from_hidden(new_model_params,{'s':e_tr})['S']
        
        flat_rf.fit(x_pred,my_data['l'])
        
        y_pred = flat_rf.predict(x_pred)
        
        tr_RMSE = np.sqrt(np.mean(
                ( y_pred - my_data['l'] )**2
                ))
        
        pprint('done!')
        pprint('tr_RMSE : %0.4f' %tr_RMSE)
        
        pprint('testing- - -')
        
        if not bsc_classify:
            x_pred = em.model.mlp.predict(te_data['y'])
        else:
            x_pred = \
            em.model.generate_from_hidden(new_model_params,{'s':e_te})['S']
        
        y_pred = flat_rf.predict(x_pred)
        
        te_RMSE = np.sqrt(np.mean(
                ( y_pred - te_data['l'] )**2
                ))
        
        pprint('done!')
        pprint('te_RMSE : %0.4f' %te_RMSE)
    
    #%% MLP regression greedy
    if mlp_classify:
        
        from sklearn.neural_network.multilayer_perceptron import MLPRegressor
        
        pprint('building MLP . . .')
        
        mlp = MLPRegressor(
                    activation='relu', 
                    alpha=1e-06, 
                    batch_size='auto',
                    beta_1=0.8, beta_2=0.9, 
                    early_stopping=True,
                    epsilon=1e-08, 
                    hidden_layer_sizes=(128,),
                    learning_rate_init=.001,
                    max_iter=150, 
                    momentum=0.9, 
                    power_t=0.5, 
                    shuffle=True, 
                    solver='adam',
                    tol=0.0001, 
                    validation_fraction=.15,
                    verbose=False, 
                    warm_start=False,
                    )
        pprint('training. . .')
        
        if not bsc_classify:
            x_pred = em.model.mlp.predict(te_data['y'])
        else:
            x_pred = \
            em.model.generate_from_hidden(new_model_params,{'s':e_te})['S']
        
        mlp.fit(x_pred,my_data['l'])
        
        y_pred = mlp.predict(x_pred)
        
        tr_RMSE = np.sqrt(np.mean(
                ( y_pred - my_data['l'] )**2
                ))
        
        pprint('done!')
        pprint('tr_RMSE : %0.4f' %tr_RMSE)
        
        pprint('testing- - -')
        
        if not bsc_classify:
            x_pred = em.model.mlp.predict(te_data['y'])
        else:
            x_pred = \
            em.model.generate_from_hidden(new_model_params,{'s':e_te})['S']
        
        y_pred = mlp.predict(x_pred)
        
        te_RMSE = np.sqrt(np.mean(
                ( y_pred - te_data['l'] )**2
                ))
        
        pprint('done!')
        pprint('te_RMSE : %0.4f' %te_RMSE)
    
    #%% bsc reconstruction scores
    if comm.rank == 0 and bsc_classify:
        
        #print(e_tr.shape)
        pprint(np.max(e_te))
        pprint(np.min(e_te))
        pprint('BSC reconstruction performances:')
        
        
#       etpred_tr = np.inner(e_tr,new_model_params['W'])
        etpred_tr = \
        em.model.generate_from_hidden(\
                                      new_model_params,\
                                      {'s':e_tr}\
                                      )['S']
        
#       etpred_te = np.inner(e_te,new_model_params['W'])
        #.astype(int)
        etpred_te = \
        em.model.generate_from_hidden(\
                                      new_model_params,\
                                      {'s':e_te}\
                                      )['S']
        
#       etpred_va =  np.inner(e_va,new_model_params['W'])
        etpred_va = \
        em.model.generate_from_hidden(\
                                      new_model_params,\
                                      {'s':e_va}\
                                      )['S']
        
        #pprint(etpred_tr.max(axis=0)/SCALE)
        #pprint(S_tr.max(axis=0))
        
        bsc_tr_PSNRlog = 20*np.log10(PeakAmp/(np.sqrt(np.mean((S_tr-etpred_tr)**2))))
        bsc_tr_negRMSE = RecNegRMSE(S_tr/PeakAmp,etpred_tr/PeakAmp)
        bsc_tr_negMSAE = RecNegMSAE(S_tr/PeakAmp,etpred_tr/PeakAmp)
        bsc_tr_percent = RecPerCent(S_tr/PeakAmp,etpred_tr/PeakAmp)
        bsc_tr_accscre = RecAccScre(S_tr/PeakAmp,etpred_tr/PeakAmp)
        
        bsc_te_PSNRlog = 20*np.log10(PeakAmp/(np.sqrt(np.mean((S_te-etpred_te)**2))))
        bsc_te_negRMSE = RecNegRMSE(S_te/PeakAmp,etpred_te/PeakAmp)
        bsc_te_negMSAE = RecNegMSAE(S_te/PeakAmp,etpred_te/PeakAmp)
        bsc_te_percent = RecPerCent(S_te/PeakAmp,etpred_te/PeakAmp)
        bsc_te_accscre = RecAccScre(S_te/PeakAmp,etpred_te/PeakAmp)
        
        bsc_va_PSNRlog = 20*np.log10(PeakAmp/(np.sqrt(np.mean((S_va-etpred_va)**2))))
        bsc_va_negRMSE = RecNegRMSE(S_va/PeakAmp,etpred_va/PeakAmp)
        bsc_va_negMSAE = RecNegMSAE(S_va/PeakAmp,etpred_va/PeakAmp)
        bsc_va_percent = RecPerCent(S_va/PeakAmp,etpred_va/PeakAmp)
        bsc_va_accscre = RecAccScre(S_va/PeakAmp,etpred_va/PeakAmp)
        
        if pronoise:
            sigma_new = new_model_params['sigma']
            
            dim = list(y_tr.shape)
            neta = np.random.normal(loc=0.0,scale=1.0,size=dim) * sigma
            bsc_tr_etaPSNR = (
            20*np.log10(
                    PeakAmp / (
                            np.sqrt(np.mean(
                                    ( y_tr - (etpred_tr+neta) )**2
                                    )) 
                            )
                    )
            )
            dim = list(y_te.shape)
            neta = np.random.normal(loc=0.0,scale=1.0,size=dim) * sigma
            bsc_te_etaPSNR = (
            20*np.log10(
                    PeakAmp / (
                            np.sqrt(np.mean(
                                    ( y_te - (etpred_te+neta) )**2
                                    )) 
                            )
                    )
            )
            dim = list(y_va.shape)
            neta = np.random.normal(loc=0.0,scale=1.0,size=dim) * sigma
            bsc_va_etaPSNR = (
            20*np.log10(
                    PeakAmp / (
                            np.sqrt(np.mean(
                                    ( y_va - (etpred_va+neta) )**2
                                    )) 
                            )
                    )
            )
            
            dlog.append('bsc_tr_etaPSNR',bsc_tr_etaPSNR)
            dlog.append('bsc_te_etaPSNR',bsc_te_etaPSNR)
            dlog.append('bsc_va_etaPSNR',bsc_va_etaPSNR)
        
        if save_recon:
            dlog.append('ET_tr_recon',etpred_tr)
            dlog.append('ET_te_recon',etpred_te)
            dlog.append('ET_va_recon',etpred_va)
        
        dscrs = [\
                 'bsc_tr_PSNRlog',\
                 'bsc_tr_negRMSE',\
                 'bsc_tr_negMSAE',\
                 'bsc_tr_percent',\
                 'bsc_tr_accscre',\
                 'bsc_te_PSNRlog',\
                 'bsc_te_negRMSE',\
                 'bsc_te_negMSAE',\
                 'bsc_te_percent',\
                 'bsc_te_accscre',\
                 'bsc_va_PSNRlog',\
                 'bsc_va_negRMSE',\
                 'bsc_va_negMSAE',\
                 'bsc_va_percent',\
                 'bsc_va_accscre',\
                 ]
        
        for itm in dscrs:
            dlog.append(itm,eval(itm))
        
        bsc_tr_scr = bsc_tr_accscre
        bsc_te_scr = bsc_te_accscre
        bsc_va_scr = bsc_va_accscre
        
        
        try:
            bsc_tr_scr = eval('bsc_tr_%s' %(params.get('txt_scr','negRMSE')))
            bsc_te_scr = eval('bsc_te_%s' %(params.get('txt_scr','negRMSE')))
            bsc_va_scr = eval('bsc_va_%s' %(params.get('txt_scr','negRMSE')))
        except:
            bsc_tr_scr = eval('bsc_tr_negRMSE')
            bsc_te_scr = eval('bsc_te_negRMSE')
            bsc_va_scr = eval('bsc_va_negRMSE')
        
        
        pprint('training:')
        pprint('PSNRlog : %0.4f; 1-RMSE : %0.4f; MAP : %0.4f; AccScr : %0.4f'
               %(bsc_tr_PSNRlog,bsc_tr_negRMSE,bsc_tr_percent,bsc_tr_accscre)
               )
        
        pprint('test:')
        pprint('PSNRlog : %0.4f; 1-RMSE : %0.4f; MAP : %0.4f; AccScr : %0.4f'
               %(bsc_te_PSNRlog,bsc_te_negRMSE,bsc_te_percent,bsc_te_accscre)
               )
        
        pprint('noiseless:')
        pprint('PSNRlog : %0.4f; 1-RMSE : %0.4f; MAP : %0.4f; AccScr : %0.4f;'
               %(bsc_va_PSNRlog,bsc_va_negRMSE,bsc_va_percent,bsc_va_accscre)
               )
        
        pprint()

#%% -- BSC ET: MLP score --

    if hasattr(em.model,'mlp'):# and comm.rank==0:
        if em.model.mlp.gen_train:
            
            paramz = params_gt if not 'top_layer' in locals() else em.lparams
            
            ann = LinearAnnealing(1)
            ann['T'] = [(0., 1.), (1., 1.)]
            ann['Ncut_factor'] = [(0.,0.),(1.,0.)]
            ann['anneal_prior'] = anneal['anneal_prior']
            ann['E_s_Flag'] = True
    
            em.anneal   = ann
    
            #----train----
            em.model.new_step = 0
            
            pprint(np.max(S_tr))
            
            em.model.mlp.max_iter = em.model.mlp.Itr
            
            #--scroes--
            etmlp = em.model.mlp
            
            pprint('ANN reconstruction performances:')
            
            etpred_tr = em.model.generate_from_hidden(new_model_params,{'s':etmlp.predict(my_data['y'])})['S']
            ET_MLP_tr_PSNRlog = 20*np.log10(PeakAmp/(np.sqrt(np.mean((my_data['S']-etpred_tr)**2))))
            ET_MLP_tr_negRMSE = RecNegRMSE(my_data['S']/PeakAmp,etpred_tr/PeakAmp)
            ET_MLP_tr_negMSAE = RecNegMSAE(my_data['S']/PeakAmp,etpred_tr/PeakAmp)
            ET_MLP_tr_percent = RecPerCent(my_data['S']/PeakAmp,etpred_tr/PeakAmp)
            ET_MLP_tr_accscre = RecAccScre(my_data['S']/PeakAmp,etpred_tr/PeakAmp)
            
            etpred_te = em.model.generate_from_hidden(new_model_params,{'s':etmlp.predict(te_data['y'])})['S']
            ET_MLP_te_PSNRlog = 20*np.log10(PeakAmp/(np.sqrt(np.mean((te_data['S']-etpred_te)**2))))
            ET_MLP_te_negRMSE = RecNegRMSE(te_data['S']/PeakAmp,etpred_te/PeakAmp)
            ET_MLP_te_negMSAE = RecNegMSAE(te_data['S']/PeakAmp,etpred_te/PeakAmp)
            ET_MLP_te_percent = RecPerCent(te_data['S']/PeakAmp,etpred_te/PeakAmp)
            ET_MLP_te_accscre = RecAccScre(te_data['S']/PeakAmp,etpred_te/PeakAmp)
            
            etpred_va = em.model.generate_from_hidden(new_model_params,{'s':etmlp.predict(va_data['y'])})['S']
            ET_MLP_va_PSNRlog = 20*np.log10(PeakAmp/(np.sqrt(np.mean((va_data['S']-etpred_va)**2))))
            ET_MLP_va_negRMSE = RecNegRMSE(va_data['S']/PeakAmp,etpred_va/PeakAmp)
            ET_MLP_va_negMSAE = RecNegMSAE(va_data['S']/PeakAmp,etpred_va/PeakAmp)
            ET_MLP_va_percent = RecPerCent(va_data['S']/PeakAmp,etpred_va/PeakAmp)
            ET_MLP_va_accscre = RecAccScre(va_data['S']/PeakAmp,etpred_va/PeakAmp)
            
           # additional noise for reconstruction if desired
        if pronoise:
            neta = np.random.randn() * new_model_params['sigma']
            ET_MLP_tr_etaPSNR = (
            20*np.log10(
                    PeakAmp / (
                            np.sqrt(np.mean(
                                    ( my_data['y'] - (etpred_tr+neta) )**2
                                    )) 
                            )
                    )
            )
            ET_MLP_te_etaPSNR = (
            20*np.log10(
                    PeakAmp / (
                            np.sqrt(np.mean(
                                    ( te_data['y'] - (etpred_te+neta) )**2
                                    )) 
                            )
                    )
            )
            ET_MLP_va_etaPSNR = (
            20*np.log10(
                    PeakAmp / (
                            np.sqrt(np.mean(
                                    ( va_data['y'] - (etpred_va+neta) )**2
                                    )) 
                            )
                    )
            )
            
            dlog.append('ET_MLP_tr_etaPSNR',ET_MLP_tr_etaPSNR)
            dlog.append('ET_MLP_te_etaPSNR',ET_MLP_te_etaPSNR)
            dlog.append('ET_MLP_va_etaPSNR',ET_MLP_va_etaPSNR)
            
            if save_recon:
                dlog.append('ET_MLP_tr_recon',etpred_tr)
                dlog.append('ET_MLP_te_recon',etpred_te)
                dlog.append('ET_MLP_va_recon',etpred_va)
            
            dscrs = [\
                     'ET_MLP_tr_PSNRlog',\
                     'ET_MLP_tr_negRMSE',\
                     'ET_MLP_tr_negMSAE',\
                     'ET_MLP_tr_percent',\
                     'ET_MLP_tr_accscre',\
                     'ET_MLP_te_PSNRlog',\
                     'ET_MLP_te_negRMSE',\
                     'ET_MLP_te_negMSAE',\
                     'ET_MLP_te_percent',\
                     'ET_MLP_te_accscre',\
                     'ET_MLP_va_PSNRlog',\
                     'ET_MLP_va_negRMSE',\
                     'ET_MLP_va_negMSAE',\
                     'ET_MLP_va_percent',\
                     'ET_MLP_va_accscre',\
                     ]
            
            for itm in dscrs:
                exec("my_%s = comm.allreduce(%s)/comm.size" %(itm,itm))
                exec("%s = my_%s" %(itm,itm))
                dlog.append(itm,eval(itm))
            
            try:
                ETMLP_tr = eval('ET_MLP_tr_%s' %(params.get('txt_scr','negRMSE')))
                ETMLP_te = eval('ET_MLP_te_%s' %(params.get('txt_scr','negRMSE')))
                ETMLP_va = eval('ET_MLP_va_%s' %(params.get('txt_scr','negRMSE')))
            except:
                ETMLP_tr = eval('ET_MLP_tr_negRMSE')
                ETMLP_te = eval('ET_MLP_te_negRMSE')
                ETMLP_va = eval('ET_MLP_va_negRMSE')
            
            pprint('training ;')
            pprint('PSNRlog : %0.4f; 1-RMSE : %0.4f; MAP : %0.4f; AccScr : %0.4f'
                   %(ET_MLP_tr_PSNRlog,ET_MLP_tr_negRMSE,ET_MLP_tr_percent,ET_MLP_tr_accscre)
                   )
            
            pprint('test ;')
            pprint('PSNRlog : %0.4f; 1-RMSE : %0.4f; MAP : %0.4f; AccScr : %0.4f'
                   %(ET_MLP_te_PSNRlog,ET_MLP_te_negRMSE,ET_MLP_te_percent,ET_MLP_te_accscre)
                   )
            
            pprint('noiseless ;')
            pprint('PSNRlog : %0.4f; 1-RMSE : %0.4f; MAP : %0.4f; AccScr : %0.4f'
                   %(ET_MLP_va_PSNRlog,ET_MLP_va_negRMSE,ET_MLP_va_percent,ET_MLP_va_accscre)
                   )
            
            
            
            if hasattr(em.model,'top_layer'):
                top_tr_mlp = em.model.top_layer.predict(etmlp.predict(my_data['y']))
                this_err = np.mean( np.abs(top_tr_mlp - my_data['l']) )
                top_tr_mlp_err = comm.allreduce(this_err)
                pprint(
                        '\nMLP to Top-Layer Error on training data:%.4f'
                        %(top_tr_mlp_err)
                        )
                num_data = np.floor(int(my_data['y'].shape[0])/2)
                pprint(' ')
                for i_rnd in (np.random.randn(10)*num_data+num_data):
                    pprint(top_tr_mlp[i_rnd])
        
#%% -- top-layer training score --

    if hasattr(em.model,'top_layer') and comm.rank==0:
                top_tr_bsc = em.model.top_layer.predict(e_s[-1])
                this_err = np.mean( np.abs(top_tr_bsc['tk'] - my_data['l']) )
                top_tr_bsc_err = comm.allreduce(this_err)
                pprint(
                        '\nBSC to Top-Layer Error on training data:%.4f'
                        %(top_tr_bsc_err)
                        )
                num_data = np.floor(int(my_data['y'].shape[0])/2)
                pprint(' ')
                for i_rnd in (np.random.randn(10)*num_data+num_data):
                    pprint(top_tr_bsc[i_rnd])
    
    #%%###########--Data-Log--###################################################

    if comm.rank == 0:

        # params
        D = int(W.shape[1])
        D = np.sqrt(D)
        n_hidden = np.empty([0,1])
        n_hidden = np.append(n_hidden[:,0],np.array(mlp.hidden_layer_sizes))

        selection = getSelection(H,pi[-1]);

        # path
        path = 'data/'+selection+'/'
        dxt = selection+'_'+time.strftime("%d.%m.%Y_%I-%M-%S")
        dnm = 'mlp_data__'+dxt

        # pkl-file (outdated)
        pprint()
        pprint('saving.. ')
        #--mlp_classify--------------------------------------------------------
        if mlp_classify or hasattr(em.model,'mlp'):
            if K_mlp and mlp_classify:
                mlps = 2
            else:
                mlps = 1
            mlp_ = mlp
            for oz in range(mlps):
                if K_mlp and oz < 1:
                    mlp = em.model.mlp
                    scores = np.array([ETMLP_tr,ETMLP_te,ETMLP_va])
                ## logging the mlp_data
                # score
                dlog.append('mlp_data.score', scores)
                # in-/output sizes
                dlog.append('mlp_data.n_in', np.array(D**2))
                dlog.append('mlp_data.n_out', np.array(H))
                # n iterations
                dlog.append('mlp_data.nIer', np.array(mlp.n_iter_))
                dlog.append('mlp_data.maxIter', mlp.max_iter)
                # gradients and coeffs
                if hasattr(mlp,'loss_curve_'):
                    dlog.append('mlp_data.Grad', np.array(mlp.loss_curve_))
                for j in xrange(int(max(n_hidden.shape)+1)):
                    dlog.append('mlp_data.W'+str(j), mlp.coefs_[j])
                    dlog.append('mlp_data.Intercepts'+str(j), mlp.intercepts_[j])
                # some params
                dlog.append('mlp_data.beta', np.array([mlp.beta_1,mlp.beta_2]))
                dlog.append('mlp_data.alpha', np.array(mlp.alpha))
                dlog.append('mlp_data.activation', mlp.activation)
                dlog.append('mlp_data.solver', mlp.solver)
                # N data points
                dlog.append('mlp_data.Ntr', Ntr)
                dlog.append('mlp_data.Nte', Nte)
                dlog.append('mlp_data.Nva', Nva)
                # Data type
                dlog.append('mlp_data.Data_Type', selection)
                # all params
                mlp_hiddens = (300)
                if len(mlp.hidden_layer_sizes) == 0:
                    mlp_hiddens = 0
                else:
                    mlp_hiddens = np.array(mlp.hidden_layer_sizes)
                dlog.append('mlp_data.params.activation',mlp.activation)
                dlog.append('mlp_data.params.alpha',np.array(mlp.alpha))
                dlog.append('mlp_data.params.batch_size',mlp.batch_size)
                dlog.append('mlp_data.params.beta_1',np.array(mlp.beta_1))
                dlog.append('mlp_data.params.beta_2',np.array(mlp.beta_2))
                dlog.append('mlp_data.params.early_stopping',np.array(mlp.early_stopping))
                dlog.append('mlp_data.params.epsilon',np.array(mlp.epsilon))
                dlog.append('mlp_data.params.hidden_layer_sizes',mlp_hiddens)
                dlog.append('mlp_data.params.learning_rate',mlp.learning_rate_init)
                dlog.append('mlp_data.params.learning_rate_init',np.array(mlp.learning_rate_init))
                dlog.append('mlp_data.params.max_iter',np.array(mlp.max_iter))
                dlog.append('mlp_data.params.momentum',np.array(mlp.momentum))
                dlog.append('mlp_data.params.nesterovs_momentum',np.array(mlp.nesterovs_momentum))
                dlog.append('mlp_data.params.power_t',np.array(mlp.power_t))
                dlog.append('mlp_data.params.random_state',np.array(mlp.random_state))
                dlog.append('mlp_data.params.shuffle',np.array(mlp.shuffle))
                dlog.append('mlp_data.params.solver',mlp.solver)
                dlog.append('mlp_data.params.tol',np.array(mlp.tol))
                dlog.append('mlp_data.params.validation_fraction',np.array(mlp.validation_fraction))
                dlog.append('mlp_data.params.verbose',np.array(mlp.verbose))
                dlog.append('mlp_data.params.warm_start',np.array(mlp.warm_start))
        
    #------bsc_classify--------------------------------------------------------
        if bsc_classify:
            dlog.append('score', np.array([bsc_tr_scr,bsc_te_scr,bsc_va_scr]))

    #--------------------------------------------------------------------------
    
    dlog.close(True)
    pprint("Done")
    
    if scorstor or iterstor:
        R1 = params.get('txtx_scr','negRMSE')+'.Scores/'
        try:
            os.stat(R1)
        except:
            os.mkdir(R1)

    #%% score save (extra compilation)

    if comm.rank == 0:
        
        #--mlp_classify-------------------------------------------------------
        if mlp_classify:
            path = '../mlp_scores'
    
            print('saving scores in %s' % path)
    
            score = np.array([tr_scr,te_scr,va_scr])
    
            scoresX = {}
            try:
                    os.stat(path+'.pkl')
                    scoresX = lod(scoresX,path)
            except:
                    fictionary_number=90
            
            if gen_datas:
                preff = 'fuse'
            else:
                preff = 'norm'
            
            scoresX['%s %dx%d N%d H%d pi%0.2f sigma%0.2f %s'
                    % (preff,D,D,N,H,pi_,sigma_init,str(mlp.hidden_layer_sizes))] = score
    
            sav(scoresX,path)
    
            pprint('done..')
            
            if scorstor or iterstor:
                
                print('Trying to print in text...')
                if 'Run' in locals():
                    if Run != None:
                        Run0 = '_%04d' % Run
                    else:
                        Run0 = ''
                else:
                    Run0 = ''
        
                if 'R' in locals():
                    if R != None:
                        R0 = '%s.Scores/%04d_s%0.2f_p%0.3f_H%02d/' \
                        % (\
                           params.get('txt_scr','negRMSE'),R,\
                           params_gt['sigma'],\
                           params_gt['pi'],\
                           params_gt['H']\
                           )
                        try:
                            os.stat(R0)
                        except:
                            os.mkdir(R0)
                    else:
                        R0 = ''
                else:
                    R0 = ''
                
                if K_mlp:
                    R0 = (
                    R0[:-1] 
                    + '_MLPRatio' + str(anneal['mlp_ratio']) 
                    + '_MLPiters' + str(anneal['mlp_iters'])
                    + '_MLPN_gen' + str(em.model.mlp.Ntr)
                    )
                
                if scorstor:
                    f = open(
                            '%sscores%s_s%0.2f_p%0.2f_H%02d.txt'
                            %(R0,Run0,params_gt['sigma'],params_gt['pi'],params_gt['H']),
                            'a')
                    f.write('%d %f %f %f\n' % (Ntr,tr_scr,te_scr,va_scr))
                    f.close()
                    print('print scores...')
        
                if iterstor:
                    f = open(
                            '%snoiter%s_s%0.2f_p%0.2f_H%02d.txt'
                            %(R0,Run0,params_gt['sigma'],params_gt['pi'],params_gt['H']),
                            'a')
                    f.write('%d %f\n' % (Ntr,mlp.n_iter_))
                    f.close()
                    print('print iterations...')
        
        #--bsc_classify--------------------------------------------------------
        if bsc_classify:
            path = '../bsc_scores'

            print('saving scores in %s' % path)

            score = np.array([bsc_tr_scr,bsc_te_scr,bsc_va_scr])

            scoresX = {}
            try:
                os.stat(path+'.pkl')
                scoresX = lod(scoresX,path)
            except:
                fictionary_number=90

            scoresX['bsc %dx%d N%d H%d pi%0.2f sigma%0.2f %s'
                    % (D,D,N,H,pi_,sigma_init,str(mlp.hidden_layer_sizes))] = score

            sav(scoresX,path)

            pprint('done..')
            
            if scorstor or iterstor:
                
                print('Trying to print in text...')
                if 'Run' in locals():
                    if Run != None:
                        Run1 = '_%04d' % Run
                    else:
                        Run1 = ''
                else:
                    Run1 = ''
    
                if 'R' in locals():
                    if R != None:
                        R1 = '%s.Scores/%04d_BSC_s%0.2f_p%0.3f_H%02d/' \
                        % (\
                           params.get('txt_scr','negRMSE'),R,\
                           params_gt['sigma'],\
                           params_gt['pi'],\
                           params_gt['H']\
                           )
                        try:
                            os.stat(R1)
                        except:
                            os.mkdir(R1)
                    else:
                        R1 = ''
                else:
                    R1 = ''
    
                if scorstor:
                    f = open(
                            '%sBSC_scores%s_s%0.2f_p%0.2f_H%02d.txt'
                            %(R1,Run1,params_gt['sigma'],params_gt['pi'],params_gt['H']),
                            'a')
                    f.write('%d %f %f %f\n' % (N,bsc_tr_scr,bsc_te_scr,bsc_va_scr))
                    f.close()
                    print('print scores...')
    
                if iterstor:
                    f = open(
                            '%sBSC_noiter%s_s%0.2f_p%0.2f_H%02d.txt'
                            %(R1,Run1,params_gt['sigma'],params_gt['pi'],params_gt['H']),
                            'a')
                    f.write('%d %f\n' % (N,Iters))
                    f.close()
                    print('print iterations...')
        
        #-Kmlp-----------------------------------------------------------------
        if K_mlp:
            path = '../ETMLP_scores'

            print('saving scores in %s' % path)

            score = np.array([ETMLP_tr,ETMLP_te,ETMLP_va])

            scoresX = {}
            try:
                os.stat(path+'.pkl')
                scoresX = lod(scoresX,path)
            except:
                fictionary_number=90

            scoresX['ETMLP %dx%d N%d H%d pi%0.2f sigma%0.2f Kuse%0.2f Ktrn%0.2f Kitr%0.2f Kdat%0.2f %s'
                    % (D,D,Ntr,H,pi_,sigma_init,mlp_Ks_ratio,mlp_et_train,mlp_et_fullI,mlp_et_fullN,str(mlp.hidden_layer_sizes))] = score

            sav(scoresX,path)

            pprint('done..')
            
            if scorstor or iterstor:
                
                print('Trying to print in text...')
                if 'Run' in locals():
                    if Run != None:
                        Run1 = '_%04d' % Run
                    else:
                        Run1 = ''
                else:
                    Run1 = ''
    
                if 'R' in locals():
                    if R != None:
                        R1 = '%s.Scores/%04d_ETMLP_s%0.2f_p%0.3f_H%02d_Kuse%0.2f_Ktr%0.2f_Kitr%0.2f_Kdat%0.2f/' \
                        % (params.get('txt_scr','negRMSE'),R,\
                           params_gt['sigma'],params_gt['pi'],params_gt['H'],\
                           mlp_Ks_ratio,mlp_et_train,\
                           mlp_et_fullI,mlp_et_fullN\
                           )
                        try:
                            os.stat(R1)
                        except:
                            os.mkdir(R1)
                    else:
                        R1 = ''
                else:
                    R1 = ''
    
                if scorstor:
                    f = open(
                            '%sETMLP_scores%s_s%0.2f_p%0.2f_H%02d.txt'
                            %(R1,Run1,params_gt['sigma'],params_gt['pi'],params_gt['H']),
                            'a')
                    f.write('%d %f %f %f\n' % (N,ETMLP_tr,ETMLP_te,ETMLP_va))
                    f.close()
                    print('print scores...')
    
                if iterstor:
                    f = open(
                            '%sETMLP_noiter%s_s%0.2f_p%0.2f_H%02d.txt'
                            %(R1,Run1,params_gt['sigma'],params_gt['pi'],params_gt['H']),
                            'a')
                    f.write('%d %f\n' % (N,Iters))
                    f.close()
                    print('print iterations...')
                
        #----------------------------------------------------------------------
    
    #%% rmv dir if desired
    
    if del_h5 and comm.rank==0:
        os.remove(output_path +'/result.h5')
        os.remove(output_path +'/terminal.txt')
        try:
            os.rmdir(output_path)
        except:
            print(output_path + ' still remained!')
    
    #%% check phrase if we ran trough
    pprint('%i is greater than %i but smaller than %i' %(1,0,2))

#%%___END.___fin.py
