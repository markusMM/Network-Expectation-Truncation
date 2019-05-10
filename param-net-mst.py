import numpy as np
import sys
import os
os.stat('pylib')
sys.path.append("pylib")
sys.path.append("pylib/utils")
import argparse
import tables
from pulp.utils.parallel import pprint
from pulp.utils.barstest import generate_bars_dict
from sklearn.neural_network.multilayer_perceptron import MLPClassifier

#%% -- General params --

seed = 0

#%% -- BSC parameters --

# datatype... possible values: 'MNIST', 'BARSTEST' or 'BARS'
datatype = 'BARS'
neg_bars = False

# function of the model
mod_fun = 'BSC_ET'

# max iterations
Iters   = 50

# Scale of the bars
SCALE   = 10

# normalized gen. sigma from gaussian
sigma   = .35

# pi number of causes
pi      = .2

# Number of datapoints to generate
N       = int(2000)

Nc_lo 	= (0.0,0.0)
Nc_hi 	= (.9,1.0)

# Dimensionality of the model

size    = 5 # D = size*size

H       = 2*size     # number of latents

D       = size**2    # dimensionality of observed data

force_H = 0          # if H is fixed given by input or default

# Approximation parameters for Expectation Truncation
Hprime  = 5
gamma   = 3

# annealing params
Tmax    = 1.2
T_iter  = .8
priT    = True

#%% -- memory heavy options --

# if (try) doing an impainting test
impaint = 0

# if doing a prediction of the binary latents (s) from the MLP
predict = 0;

#%% -- (ML)P-params --

# enabling co-training
K_mlp = False

# data set size
Ntr =   10000
Nte =    7500
Nva =    4500

# after training max consecutive iterations
mlp_max_iter = 50

# generative max iteration fraction
mlp_gen_iter = 1

# generative max/min training data fraction
mlp_gen_data = 1
mlp_min_data = 1/Ntr

# if the validation fraction in training shall be noiseless
noiseless_valid = False

# no. batches for the gen. mlp data
batches = 1

mlp_lrate = .001
mlp_lmomm = 'constant'

early_stopping = True
warm_start = True
nesterovs_momentum = True
# hidden layer sizes
n_hidden=np.array([]) # here no hidden layers
validation_fraction=.1
batch_size=200
eta = .001
# experimental ... 
# if modifying sklearn/neural_network/MLP.py 
# and adding max option 
# for maximum combination instead of linear
mlp_maxpool = False

gen_datas = False

gen_train = False
mlp_ratio = .0

#%% top-layer variables

top_train = .5          #when does the training start
top_iters = (.8,50)     #when does how many iterations happen per step
top_mulIt = 0           #should there even be more top-layer iterations each step


#%% -- imput parsing 2.0 xD --
n = 2
while n < len(sys.argv):
    n2 = 1
    try:
        a = float(sys.argv[n+1])
        try:
             exec('%s = %f' % (sys.argv[n],a))
             pprint('%s = %f' % (sys.argv[n],a))
        except:
            pprint('Error: Bad argument name!')

    except:
        if      sys.argv[n] == 'outpath' or sys.argv[n] == 'h5path' or sys.argv[n] == 'h5_path':
            try:
                exec('h5path = "%s"' % (sys.argv[n+1].strip()))
                pprint('h5-path = "%s"' % sys.argv[n+1].strip())
            except:
                pprint('Error: Bad  h5-path name!')
            n2 = 1
        elif    sys.argv[n] == 'mlppath' or sys.argv[n] == 'h5_mlp':
            try:
                exec('mlppath = "%s"' % (sys.argv[n+1].strip()))
                pprint('mlp-path = "%s"' % sys.argv[n+1].strip())
            except:
                pprint('Error: Bad mlp path string: %s' %sys.argv[n])
            n2 = 1
        elif    sys.argv[n] == 'batch_size'     or sys.argv[n] == 'mlp_lmomm' \
        or      sys.argv[n] == 'txt_nm'         or sys.argv[n] == 'h5f_nm' \
        or      sys.argv[n] == 'actfun'         or sys.argv[n] == 'mod_fun'\
        or      sys.argv[n] == 'txt_scr':
            try:
                exec('%s = "%s"' % (sys.argv[n], sys.argv[n+1].strip()))
                pprint('%s: "%s"' % (sys.argv[n], sys.argv[n+1].strip()))
                exec('print(type(%s))' %sys.argv[n])
            except:
                pprint('Error: Bad strg arg: %s' %sys.argv[n])
            n2 = 1
        elif    sys.argv[n] == 'top_iters' or sys.argv[n] == 'Nc_lo' or sys.argv[n] == 'Nc_hi' or sys.argv[n] == 'Nc_lo':
            try:
                exec('%s = %s' % (sys.argv[n], sys.argv[n+1].strip()))
                pprint('%s: %s' % (sys.argv[n], sys.argv[n+1].strip()))
                exec('print(type(%s))' %sys.argv[n])
            except:
                pprint('Error: Bad strg arg: %s' %sys.argv[n])
            n2 = 1
        else: 
            try:
                exec('%s = %s' % ( sys.argv[n], sys.argv[n+1].strip() ))
                pprint('%s = %s' %( sys.argv[n], sys.argv[n+1].strip() ))
                n2 = 1
            except:
                try:
                    exec('%s = 1' % sys.argv[n].strip())
                    pprint('%s: 1' % sys.argv[n].strip())
                except:
                    pprint('Error: Bad argument name!')
                n2 = 0
            
    n += 1 + n2
    
# -- integers --
#size = int(size)
D = int(D) # things that cannot be changed
size = int(size)
H = int(H)
Hprime = int(Hprime)
gamma  = int(gamma)
N = int(N)
Ntr = int(Ntr)
Nte = int(Nte)
Nva = int(Nva)
Iters = int(Iters)

# -- flags --
mlp_gen_iter = int(mlp_gen_iter) #or 100*int(mlp_gen_iter<1)
mlp_gen_data = int(mlp_gen_data) # no. data points else Ntr or N_bsc

# -- params --
if 'piH' in locals():
    pi = piH/H

# fit truncation params to barstest
if 'truncfit' in locals():
    if truncfit:
        if not force_H:
            H = int(np.sqrt(D)) * 2
        gamma = int(pi*H+1)
        Hprime = gamma + 2
        
# -- n_hidden --
n_hidden = np.round(np.array(n_hidden))
n_hidden = tuple(n_hidden[n_hidden!=0])

# -- data type --
dtypes = [
        'BARS',
        'MNIST',
        'IMG',
        'MNIST_patch'
        ]

if 'datype' in locals():
    datatype = dtypes[int(datype)]

# decide the ET gen model
if "mod_op" in locals():
    if mod_op == 1:
        mod_fun = 'BSC_ET'
    if mod_op == 2:
        mod_fun = 'MCA_ET'
        
#%% top-layer

if 'top_layer' in locals():

    execfile('../utils/%s.py' %(top_layer))
    
    print(top_layer)
    
    if not top_mulIt:
        top_iters = (.8,1)

#%% random seed

if locals().get('Run',False) and locals().get('fix_seed',False):
    np.random.seed(int(Run))
    seed = int(Run)
else:
    if seed > 0:
        np.random.seed(seed)
    else:
        seed = np.random.get_state()[1][0]

#%% mlp related

solvers = [
	'adam',
	'sgd',
	'lbfgs'
	]

actfuns = [
	'relu',
	'logistic',
	'tanh'
	]

if 'solver' not in locals():
	solver = 'adam'

if 'actfun' not in locals():
	actfun = 'relu'

if not isinstance(solver,str):
	solver = solvers[int(solver)]

if not isinstance(actfun,str):
	actfun = actfuns[int(actfun)]
    
warm_start = bool(warm_start)

mlp_par_sync = False

#%% -- MLP-Model --

pprint()

#--- loading existing mlp from h5 (if desired) ---#
if 'data_txt' in locals() or 'mlppath' in locals():
    
    if      'data_txt' in locals():
        f = open('../data_list.txt','r')
        for d in range(int(data_txt)):
            h5 = f.readline()
        
        f.close()
    elif    'mlppath'  in locals():
        h5 = mlppath
                    
    h5 = 'output/'+h5.strip()+'/result.h5'
    
    pprint('loading data from:\n%s' %h5)
    
    try:
        mlp_params = {}
        with tables.open_file(h5, 'r') as data_h5:
            W = data_h5.root.W.read()[0]
            pprint(W.shape)
            solver = data_h5.root.mlp_data.params.solver.read()[0]
            pprint(solver)
            actfun = data_h5.root.mlp_data.params.activation.read()[0]
            nesterovs_momentum = data_h5.root.mlp_data.params.nesterovs_momentum.read()[0]
            validation_fraction = data_h5.root.mlp_data.params.validation_fraction.read()[0]
            early_stopping = data_h5.root.mlp_data.params.early_stopping.read()[0]
            batch_size = data_h5.root.mlp_data.params.batch_size.read()[0]
            n_hidden = data_h5.root.mlp_data.params.hidden_layer_sizes.read()[0]
            pprint(n_hidden.shape[0])
            eta = data_h5.root.mlp_data.params.learning_rate_init.read()[0]

            inter = {}
            W_mlp = {}
            
            for h in xrange(int(n_hidden.shape[0])+1):
                exec(
                    'inter[%d] = data_h5.root.mlp_data.intercepts%d.read()'
                    %(h,h)
                    )
                exec(
                    'W_mlp[%d] = data_h5.root.mlp_data.W%d.read()'
                    %(h,h)
                    )
            nIter = data_h5.root.mlp_data.nIter.read()[0]
            
        data_h5.close()
        
    except:
        
        pprint('ERROR: Data lost: Building standard MLP!')
    
pprint('Building MLP classifier..  ')

early_stopping = bool(early_stopping)

mlp = MLPClassifier(activation=actfun, alpha=1e-06, batch_size=batch_size,
                    beta_1=0.8, beta_2=0.9, early_stopping=early_stopping,
                    epsilon=1e-08, hidden_layer_sizes=n_hidden,
                    learning_rate=mlp_lmomm, learning_rate_init=mlp_lrate,
                    max_iter=mlp_max_iter, momentum=0.9, 
                    nesterovs_momentum=nesterovs_momentum, power_t=0.5, 
                    random_state=seed, shuffle=True, solver=solver,
                    tol=0.0001, validation_fraction=validation_fraction,
                    verbose=False, warm_start=warm_start
                    )

mlp.gen_train = 1-gen_train > 0

pprint(     'done.. with '
          +str(mlp.hidden_layer_sizes)
          +' hidden units, '
          +str(int(D))
          +' input nodes and '
          +str(H)
          +' classes.'
)

if 'inter' in locals():
    mlp.intercepts_ = inter
if 'nIter' in locals():
    mlp.n_iter_ = nIter
    mlp.max_iter = nIter*2 + mlp.max_iter + Iters*(1-gen_train)
if 'W_mlp' in locals():
    mlp.coefs_ = W_mlp
    mlp.n_layers_ = len(n_hidden)
if 'N_mlp_et' in locals():
    if N_mlp_et:
        mlp_gen_data = 1
        mlp.Ntr = N_mlp_et

if 'mlp_gen_data' in locals():
    if mlp_gen_data:
        mlp.Ntr = Ntr		
		
if 'mlp_gen_iter' in locals():
    if mlp_gen_iter:
        mlp.Itr = mlp.max_iter

if 'mlp_stoch_reinit' in locals():
    if mlp_stoch_reinit:
        mlp._stoch_reinit = mlp_stoch_reinit
        
if 'ann_sgl_fit' in locals():
    mlp.sgl_fit = ann_sgl_fit

pprint(mlp)

#%% -- BSC Ground truth parameters. Only used to generate training data. --

if datatype == 'BARS':
    Wgt = SCALE*generate_bars_dict(H=H,R=int(np.sqrt(D)),neg_bars=bool(neg_bars))
else:
    Wgt = SCALE*np.random.randn(D,H)

params_gt = {
        'W'     :  Wgt,
        'pi'    :  pi,
        'sigma' :  sigma*SCALE,
        'SCALE' :  SCALE,
        'N'     :  N,
        'H'     :  H,
        'D'     :  D,
        'gamma' :  gamma,
        'Hprime':  Hprime
    }


#%% -- Binary Sparse Coding model


# Import and instantiate a model
mod_fun     = mod_fun.upper()
libstr      = mod_fun.lower()
exec('from pulp.em.camodels.%s import %s' % (libstr,mod_fun))
exec('model = %s(%d,%d,%d,%d)' % (mod_fun,D, H, Hprime, gamma))
model.fun_name = libstr
if 'top_layer' in locals():
    if type(top_layer).__name__ != 'str':
        model.top_layer = top_layer

#%% -- additional tuning parameters --


if 'mlp_maxpool' in locals():
    mlp.max_pool = mlp_maxpool

if 'K_mlp' in locals():
    
    if K_mlp:
        #model.select_Hprimes = select_Hprimes
        model.mlp = mlp
        model.mlp.max_iter = mlp_gen_iter
        if not(batch_size == 'auto'):
            model.mlp.batch_size=1
            model.mlp.batches_cn=Ntr//batch_size
        if 'mlp_par_sync' in locals():
            model.mlp.par_sync = mlp_par_sync
        if 'mlp_n_consec' in locals():
            model.mlp.n_consec = mlp_n_consec
        if 'mlp_sup_data' in locals():
            model.mlp.sup_data = mlp_sup_data

#%% -- BSC Choose annealing schedule --

from pulp.em.annealing import LinearAnnealing

if 'anneal' not in locals():
    anneal = LinearAnnealing(Iters)
    anneal['T'] = [(0, (1+Tmax)), (T_iter, 1.)]
    #anneal['Ncut_factor'] = [(0,0.),(.9,1.)]
    anneal['Ncut_factor'] = [Nc_lo,Nc_hi]
    anneal['anneal_prior'] = priT
    if hasattr(model,'mlp') or hasattr(model,'top_layer'):
        anneal['E_s_Flag'] = 1
    #Data noise
    if 'dnoise' in locals():
        anneal['data_noise'] = dnoise
    if 'W_noise' in locals():
        anneal['W_noise'] = W_noise
    # MLPs
    anneal['mlp_train'] = [(0,0),(gen_train,.5),(1,.5)]
    anneal['mlp_ratio'] = [(0,0),(mlp_ratio,.5),(1,.5)]
    if 'mlp_lin_n' in locals():
        anneal['mlp_datas'] = [(0,0),(gen_train,mlp_min_data/Ntr or 1/Ntr),(mlp_lin_n,mlp_gen_data)]
    if 'mlp_lin_i' in locals():
        anneal['mlp_iters'] = [(0,0),(gen_train,0),(mlp_lin_i,mlp_gen_iter)]
    if 'ann_data_scheme' in locals():
        anneal['mlp_datas'] = ann_data_scheme
    if 'ann_iter_scheme' in locals():
        anneal['mlp_iters'] = ann_iter_scheme
    if 'ann_pseudo' in locals():
        anneal['ann_pseudo'] = ann_pseudo
    if 'mlp_unicause' in locals():
        anneal['mlp_unicause'] = mlp_unicause
    # top-layer
    if 'top_Ngen' in locals():
        anneal['top_Ngen'] = top_Ngen
    if 'top_layer' in locals():
        anneal['top_train'] = [(0,0),(1,top_train)]
        anneal['top_iters'] = [(0,1),top_iters]
    if 'singletons' in locals():
        anneal['singletons'] = singletons
