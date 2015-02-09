# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 18:54:21 2015

@author: agrigori
"""

from elm import *

def test_elms():
    """
    Test incremental ELM
    """
    import sklearn.datasets as ds
    
    (X,Y) = ds.make_regression(n_samples = 1000, n_features = 7,n_informative=5, \
                       n_targets=2, bias = 2.0, effective_rank = 2)
    
    neurons_dict = {'linear': True, 'sigmoid': 500 }
    
    model = Inc_TROP_ELM( neurons=neurons_dict, reg_par_optim_type='press', loo_evaluation_step=5,
                             svd_reorth_step=10)
       
    X_train = X[0:900,:]; Y_train = Y[0:900,:]
    X_pred = X[900:,:]; Y_pred = Y[900:,:]     
    
    model.set_data(X_train,Y_train)
    model.train()
    res = model.predict(X_pred, Y_pred )       
      
    return res
    
if __name__ == '__main__':
    #d_inp = np.array([[1., .9],[.3, .7],[ 1.5, 2.0]])
    #d_out = np.array([1., .4, .8])
    #ELM(d_inp, d_out, {'sigmoid': 5, 'rbf':2, 'lin': 0})
#        
#    from my_startup import *
#    import ls_test
#    
#    r = io.loadmat( '/users/agrigori/Temp/10_7.mat' )
#    A = r['A'];    b = r['b']
#    
#    A1 = A.copy(); b1 = b.copy()
#    
#    res1 = ls_test.ls_svd(A,b,cond=1e-6, overwrite_a=True, overwrite_b=True)
#
#    res2 = ls_test.ls_cof(A1,b1,cond=1e-6, overwrite_a=True, overwrite_b=True)    
#    
#    res3 = ls_test.ls_svdb(A,b,cond=1e-6, overwrite_a=True, overwrite_b=True)
    
    res = test_elms()
    