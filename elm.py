# -*- coding: utf-8 -*-
"""
Created on Mon May 20 12:45:17 2013

@author: agrigori
"""

import numpy as np
import numpy.random as rnd
from matplotlib import mlab # for percentile calculation

from scipy.spatial import distance as dist
import scipy as sp
import scipy.linalg as la
from scipy.optimize import minimize_scalar # Find optimal lambda for
import itertools
import warnings

# custom imports ->
from extra_ls_solvers import extra_ls_solvers as ls_solve # to solve regular ELM least square problem
from utils import data_utils as du # my normalization module
from utils import numeric_utils as nu
import mrsr
from svd_update import svd_update
# custom imports <-

def loo_error(X,Y,loo_type='press',U=None,S=None,lamda=0.0):
    """
    Computes leave-one-out error for system X*beta = Y.
    There are two ways to compute loo_error: regular, by SVD.
    
    SVD method has an advantage that it is easy to evaluate loo error
    for multiple lambda parameters. SVD needed to be computed only once.
    If the SVD method is used then SVD decomposition is computed before the
    function is called with U and S parts of SVD. (X = USVt )

    Input:
        X - matrix X. This can be none of SVD method is used.        
        Y - matrix(or vector) of right hand side
        loo_type - 'press' or 'gcv'
        U - matrix U in SVD (thin SVD is enough)
        S - vector (not a matrix) of singular values as returned by SVD 
            routine.
            If both matrices U and S are provided the SVD method is used,
            otherwise regular methd is used.            
        lamda - regularization parameter    
    Output:
        loo error
    """
    
    if  not loo_type.lower() in ('press', 'gcv'):
        raise ValueError("Function loo_error: parameter loo_type must be either 'press' or 'gcv'. ")
        
    SVD_method = False;
    if ( (not U is None) and (not S is None) ):
        SVD_method = True
    elif (X is None):
        raise ValueError("Function loo_error: For regular method you need to provide X matrix.")
    
    if SVD_method:
        n_samples = U.shape[0]
        
        if (lamda != 0.0):                       
            S = np.power(S,2)
            Sl = S + n_samples*lamda # parameter lambda normalized with respect to number of points !!!
            S = S / Sl
            
            
            (S, orig_S_shape) =  du.ensure_column(S);
            
            #!
            DUt = np.multiply( S, U.T) # elementwise multiplication, 
            
        else:
            orig_S_shape = None
            DUt = U.T
        
        Mii = np.multiply(U, DUt.T).sum(1) # compute only diagonal of the matrix M        
        D = 1.0/(1.0 - Mii);
        
        MY = np.dot( U ,np.dot(DUt,Y) )
        
    
    else: # regular method        
        (n_samples,X_dim) = X.shape 
        
        XtX = np.dot( X.T, X)
        
        if (lamda != 0.0):   
            XtX = XtX + n_samples*lamda*np.eye(X_dim) # parameter lambda normalized with respect to number of points !!!
            
        chol = sp.linalg.cho_factor(XtX, overwrite_a = True, check_finite=False)
        
        XtX_inv = sp.linalg.cho_solve(chol, np.eye(X_dim) )
        
        M1 = XtX_inv * X.T        
        
        Mii = np.multiply(X, M1.T).sum(1) # compute only diagonal of the matrix M         
        D = 1.0/(1.0 - Mii);
                
        MY = np.dot(X, np.dot( XtX_inv, np.dot(X.T, Y) ) )
    
    
    if loo_type.lower() == 'press':
        (D_col,tmp) = du.ensure_column(D)
            
        res = np.multiply( D_col, (Y - MY) ) # need vectors are columns in this matrix
        res = np.power(res,2).mean(axis=0) # mean wrt columns 
    else: # GCV
        trace_P = np.sum( 1.0 - Mii )        
                
        res = np.sum(np.power((Y - MY),2), axis=0 ) # sum wrt columns 
        res = (n_samples / trace_P**2) * res       
        
    if orig_S_shape is not None:
        S = du.rev_ensure_column(S,orig_S_shape)    
        
    return  res  # Sum of LOO MSE for each dimension of Y
    
   
class Struct(object): # Needed for emulation of structures
    pass    


epsilon2 = 1e12
class LinearModel(object):
    """
    Linear model 
    """

    def __init__(self,**kwargs):
        """
        Constructor, set necessary parameters of the model which are used further.
        
        Input:
            The the dict kwargs these parameters are currently supported.
            
            {'lamda': values} predefined value of regularization parameter
                Note that regularization parameter is multiplied by the number of samples
                in computations.
                
            {'reg_par_optim_type': ('press','gcv','cv','none') } - if lamda is not
                given these optionas are available for searching regularization
                parameter.            
        """
        
        self.model = Struct()
        self.model.orig_kwargs = kwargs        
        
        self.model.lamda = None        
        if 'lamda' in kwargs:
            lamda = kwargs['lamda']
            if lamda > 0:
                self.model.lamda = lamda
        
        self.model.reg_par_optim_type = None
        if ('reg_par_optim_type' in kwargs) and (self.model.lamda is None):
            reg_par_optim_type = kwargs['reg_par_optim_type']
            if (not reg_par_optim_type is None) and (reg_par_optim_type.lower() != 'none'):
                if not reg_par_optim_type in ('press','gcv','cv'):
                    raise ValueError("LinearModel.__init__: unknown values of reg_par_optim_type '%s' " % reg_par_optim_type )
                else:
                    self.model.reg_par_optim_type = reg_par_optim_type                        
                                        
        self.model_built = False
        self.data_set = False
        self.model.type = 'linear'
        
    def __repr__(self):
        """
        Text representation (detailed) of an object.
        """        
        
        return "%s, optim way: %s" % (self.model.type, self.model.reg_par_optim_type )
    
    def set_data(self,X,Y,normalize=True):
        
        """
        Sets data to the model.
        Input:
            X - training features rowwise
            Y - training targets rowwise
            normalize - whether or not normalize training data
         """       
         
        self.data = Struct()
        
        if normalize:
            X_norm,x_means,x_stds = du.normalize(X,ntype=0) # zero mean unit variance    
            Y_norm,y_means,y_stds = du.normalize(Y,ntype=0) # zero mean unit variance                
                
            self.data.normalized = True   
             
            self.data.X = X_norm
            self.data.Y = Y_norm
            
            self.data.x_means = x_means
            self.data.x_stds = x_stds
            
            self.data.y_means = y_means
            self.data.y_stds = y_stds
        else:
            self.data.normalized = False
            
            self.data.X = X
            self.data.Y = Y

        self.model_built = False
        self.data_set = True
        
    def train(self):   
        """
        Training
         
        Input:
           reg_optim_type - type of regularization parameter optimization             
                            Possible values:
                            'none','loo','cv'.
        """
        
        if not self.data_set:
            raise ValueError("LinearModel.train: Data is not set. Training is impossible!")
        
        n_points = self.data.X.shape[0]
        X_d = np.hstack( (np.ones( (n_points ,1)) ,self.data.X) ) # add the column of ones
        Y_d = self.data.Y 
        
        res = self.solve_lin_ols( X_d, Y_d, U=None,S=None,Vh=None )        
        
        self.model.coeffs = res[0]
        self.model.num_rank = res[1]        
        lamda = res[2]
        
        if (lamda is None) or ( lamda > 1.0) or ( lamda < 0.0) : # no lamda optimization happened or wrong optim results
            self.model.lamda = None
        else:
            self.model.lamda = lamda  
                    
        self.model.optim_output = res[3]
        self.model_built = True
             
    def solve_lin_ols(self, X, Y, U=None,S=None,Vh=None):
        """
        Method solves the regularized ols problem, given the SVD decomposition of X.

        Input:
            X - regressor variables
            Y - dependent variables
            U,S,Vh - SVD of X. Actually original X is used only in the cv regularization 
                     method.
                     
        Output:
            res - solution returned by solve_ols_svd.
        """
        
        reg_optim_type = self.model.reg_par_optim_type        
        
        if  not reg_optim_type in (None,'press','gcv','cv'):
            raise ValueError("Linear Model: wrong regularization optimization type %s" % reg_optim_type) 
        
        if U is None: # Perform SVD if it is not done before
            (U,S,Vh) = sp.linalg.svd(X, full_matrices=False, overwrite_a=False, check_finite=False)
       
        n_points = self.data.X.shape[0]
        lamda = None; optim_output = None
        if reg_optim_type in ('press', 'gcv'):
            optim_output = minimize_scalar(lambda lamda: np.sum( loo_error(None,Y, reg_optim_type, U,S,lamda) ), 
                              bounds= (0.0,1.0), method='Bounded')
            lamda = optim_output.x
            
        elif (reg_optim_type == 'cv'):
            
            import sklearn.cross_validation as cv
            
            def cv_optim(par_lamda):
                cv_error = 0
                for cv_train_indeces, cv_test_indeces in cv.KFold(n_points, indices=True, n_folds=10,shuffle = True):
                    X_cv_train = X[cv_train_indeces,:]; Y_cv_train = Y[cv_train_indeces,:]
                    X_cv_test = X[cv_test_indeces,:]; Y_cv_test = Y[cv_test_indeces,:]
                    
                    (U1,S1,Vh1) = sp.linalg.svd(X_cv_train, full_matrices=False, overwrite_a=False, check_finite=False)
                    
                    res = solve_ols_svd(U1,S1,Vh1, Y_cv_train, par_lamda )
                    coeffs = res[0] 
                    
                    Y_predict = np.dot( X_cv_test, coeffs )
                    
                    # compute MSE. This makes sence only if the Y data is normalized 
                    # i.e. different Y dimensions have the same scale.
                    cv_error += np.mean( np.power( (Y_predict - Y_cv_test), 2 ), axis=0 )
         
                return np.sum( cv_error )
         
            optim_output = minimize_scalar(cv_optim, bounds= (0.0,1.0), method='Bounded')
            lamda = optim_output.x
             
        else:  # None             
            if ( S[0]/S[-1] > epsilon2 ): 
                raise ValueError("LinearModel: too large condition number %f and no regularization" % S[0]/S[-1] )
        
        res = solve_ols_svd(U,S,Vh, Y, lamda)
        return res + (lamda,optim_output)
                        
    def predict(self,X_pred,Y_known = None):
        """        
        Predict method.
        
        Input:
            X_pred - data (rowwise) to which prediction to be made
            Y_known - known predictions to compute an error.
            
        """
        
        if not self.model_built:
            raise ValueError("Linear model: Prediction is impossible model is not trained.")
        
        if self.data.normalized:            
            (X_d,tmp1,tmp2) = du.normalize( X_pred, None, self.data.x_means,self.data.x_stds )
        else:
            X_d = X_pred
        
        X_d = np.hstack( ( np.ones( (X_d.shape[0] ,1)) ,X_d ) ) # add the column of ones
        Y_pred = np.dot( X_d, self.model.coeffs)
        if self.data.normalized: 
            Y_pred = du.denormalize( Y_pred, self.data.y_means, self.data.y_stds )
                
        if Y_known is None:
            return (Y_pred, None)
        else:
            return (Y_pred,  np.mean( np.power( Y_pred - Y_known, 2), axis=0  ) )
         
         
    def copy(self):         
        """
            Function which creates a copy of current model, preserving
            the model parameters
        """

        this_class = type(self)     
        new_instance = this_class(**self.model.orig_kwargs)
        
        return new_instance

class ELM(object):
    
    def __init__(self,**kwargs):
        """
        Constructor.
        
        input:
            X - input data (X) (rowwise). Input normalization is done
                    automatically.
            Y - output data (Y) (rowwise). Might be vectorial output 
                    (Y-matrix)
            kwargs 
            ('neurons':  neurons_dict) - how many neurons to take. There are 3 types
                           of keys possible: sigmoid, rbf, linear. The value is how many neurons
                           of this type to take. For linear neurons the dictionary value does
                           not matter.
                           It is possible to pass also neurons structure, generated
                           somewhere else.
            
        """
        
        self.model = Struct() # model data
        self.model.orig_kwargs = kwargs 
        
        if not 'neurons' in kwargs:
            raise ValueError( "ELM.__init__: neurons are missing" )
               
        self.model.neurons_dict = kwargs['neurons']          
        
        self.model_built = False 
        self.data_set = False
        self.model.type = 'elm' # type of the model        
    
    def __repr__(self):
        """
        Text representation (detailed) of an object.
        """        
        
        return "%s, optim way: %s" % (self.model.type, self.model.reg_par_optim_type )
    
    def set_data(self,X,Y,normalize=True):
        """
        Set data to ELM
        """

        X_norm,x_means,x_stds = du.normalize(X,ntype=0) # zero mean unit variance    
        Y_norm,y_means,y_stds = du.normalize(Y,ntype=0) # zero mean unit variance
        
        if (len(Y_norm.shape) == 1):
            Y_norm.shape = (Y_norm.shape[0],1)
            
        if (len(X_norm.shape) == 1):
            X_norm.shape = (X_norm.shape[0],1)
            
        self.data = Struct() # summarizes trainig data
        
        self.data.d_inp = X # original training data
        self.data.d_inp_n = X_norm # normalized training data
        self.data.d_out = Y # output training data
        self.data.d_out_n = Y_norm # normalized training data
        
        self.data.X_dim = X.shape[1] # input train dimensionality 
        self.data.n_samples = X.shape[0] # Number of samples in training
        
        self.data.x_means = x_means # means of normalization of train data
        self.data.x_stds = x_stds # scale of normalization of train data
        self.data.y_means = y_means # means of normalization of train data
        self.data.y_stds = y_stds # scale of normalization of train data        
            
        if isinstance(self.model.neurons_dict, dict):
            self.neurons = ELM._generate_neurons(self.model.neurons_dict, self.data.d_inp_n)
        else: 
            self.neurons = self.model.neurons_dict # already neurons
        
        H = self._apply_neurons(self.neurons, self.data.d_inp_n)
        
        self.H = H # Matrix of neurons. Columns - neurons
        
        self.data_set = True
    
    def get_generated_neurons(self):
        """
        Return neurons generated in the function set_data
        """        
        
        if not getattr(self, "neurons"):
            raise ValueError("ELM.get_generated_neurons: Neurons have not been generated by the get_generated_neurons function.")
        else:
            return self.neurons
        
        
        
    def train(self):
        """
        Train ELM. (solve the minimum-norm least squares)        
        """
        
        if not self.data_set:
            raise ValueError("ELM.train: Data is not set. Training is impossible!")        
        
        H = np.hstack( (np.ones( (self.H.shape[0],1) ) ,self.H) ) # include column of ones
        
        
        #res = ls_solve.ls_cof( H, self.data.d_out_n, overwrite_a = False,
        #                 overwrite_b = False, check_finite = True)
                         
        res = self.solve_elm_ols(H, self.data.d_out_n)
        
        self.model.coeffs = res[0] # model coefficients
        self.model.num_rank =res[1] # numerical rank of a neuron matrix
                                    # determined by least square solver
        self.model_built = True
    
    
    def is_stable(self, coeffs=None):
        """
        Is the model stable if the input is time series embedding matrix?
        """
        
        if coeffs is None:
            coeffs = self.model.coeffs
        
        if not self.model_built:
            raise ValueError("Stability can't be checked. The model is not \
                              trainded yet.")
        
        test_matrix = np.vstack( ( np.zeros((1, coeffs.shape[0]-1 )  ), 
                                      np.eye( coeffs.shape[0]-1 ) ) )
        test_matrix = np.hstack( (test_matrix, coeffs) )
        
        ss = sp.linalg.svd(test_matrix, full_matrices=False, compute_uv=False, overwrite_a=False, check_finite=False )
        
        if np.any( (ss > 1) ):
            return False
        else:
            return True
    
    
    def solve_elm_ols(self,H, Y):
        """
        This method solves the OLS problem after neurons have been build
        This method is substituded in OP-ELM and TROP-ELM class where 
        the additional regularization is applied.
        
        This method is done to uniformly use all Elm based models.     
        """        
        
        res = ls_solve.ls_cof( H, Y, overwrite_a = False,
                         overwrite_b = False, check_finite = False)
        
        lamda = None 
        return (res[0], res[2], lamda, None) # coefficients, numerical rank, lamda, optimization params
    
    
    def predict(self, X, Y = None, normalize=True):     
        """
        Give predictions for new values provided we have a trained model.
        
        Input:
            X - row wise matrix of inputs
            Y - known outputs for X. Needed for computation of error.
            normalize - whether X should be normalized according to 
                        the training data.
        """
        
        if not self.model_built:
            raise ValueError("Predictions can't be made. The model is not \
                              trainded yet.")
        
        #!!! TODO: Wrong method no call to appy neurons 
        X_norm,t1,t2 = du.normalize(X, means = self.data.x_means,\
                                                 stds = self.data.x_stds)                                                 
        del t1, t2                                             
        
        H = self._apply_neurons( self.neurons, X_norm )                
        H = np.hstack( (np.ones( (H.shape[0],1) ) ,H) ) # include column of ones        
        
        
        Y_pred = np.dot(H, self.model.coeffs )
        
        Y_pred = du.denormalize( Y_pred, self.data.y_means, self.data.y_stds )        
        
        if not Y is None:
            nmse = nu.sqr_euclid( Y- Y_pred ) / Y_pred.shape[0]
        else:
            nmse = None            
        
        return Y_pred, nmse
    
    def copy(self):         
        """
            Function which creates a copy of current model, preserving
            the model parameters
        """

        this_class = type(self)     
        new_instance = this_class(**self.model.orig_kwargs)
        
        return new_instance
                
    @staticmethod
    def _generate_neurons(neurons_dict, X_data):
        """
        Generate neurons from neurons_dict
        
        Input:
            neurons_dict - how many neurons to take. There are 3 types
                            of keys possible: sigmoid, rbf, linear. The value is how many neurons
                            of this type to take. For linear neurons the dictionary value does
                            not matter. 
            X_data - data, used only for generation of rbf neurons, to determine
                     neurons centers.
        """

        X_dim = X_data.shape[1]        
        neurons  = Struct() # summarizes generated neurons
        
        neurons.types = [] # list with types of neurons
     
        if 'linear' in  neurons_dict:
            neurons.linear = True                 
            neurons.types.extend(  [ ('l_%i' % (i+1)) for i in xrange( X_dim ) ]  )            
        
        if 'sigmoid' in neurons_dict:
            num = neurons_dict['sigmoid']
            (W,b) = ELM._init_sigmoid_weights_def(num, X_dim )
                              
            neurons.sigmoid = (W,b)
            neurons.types.extend( [('s_%i' % (i+1)) for i in xrange(num)] )            
            
        if 'rbf' in neurons_dict:
            num = neurons_dict['rbf']
            
            data_samples = X_data.shape[0] # how many samples
            if data_samples > 5000:
                num_in_dist = 5000
            else:
                num_in_dist = data_samples
                
            (centers, stds) = ELM._init_rbf_weights_def(num, X_data, 
                                                           num_in_dist, 
                                                           'euclidean')            
            neurons.rbf = ( centers, stds )
            neurons.types.extend( [('r_%i' % (i+1)) for i in xrange( centers.shape[0] )] )
        
        return neurons
        
    @staticmethod    
    def _apply_neurons(neurons, X_data):
        """
        Apply neuron weight to the data

        Input:
            X_data - data matrix, samples are rows
        """
        
    
        H1 = None; H2 = None; H3 = None
        if hasattr(neurons,'linear'):
            H1 = X_data
        
        if hasattr(neurons,'sigmoid'):
            (W,b) = neurons.sigmoid
            H2 =   ELM._create_sigmoid_neurons(X_data, W ,b)
         
        if hasattr(neurons,'rbf'):
            (centers, stds) = neurons.rbf
            H3 =  ELM._create_rbf_neurons(X_data, centers, stds, 'euclidean')   
        
        H = np.hstack( [ i for i in (H1,H2,H3) if i is not None] )
        return H
    
    @staticmethod
    def _init_sigmoid_weights_def(num, dim):
        """ 
        Computes weights and additive terms of sigmoidal neurons.
        Static method.
    
        Input:
            dim - dimensionality of input data
            num - number of neurons required
            
        Output:
            W - matrix with neuron weights (column-wise)
            b - vector with additive terms
        """
        
        #We want that Var[ x_1^T * d + 1] = 2.5.
        # Variance of uniform distribution is 1/12 * (b-a)^2 = {b = -a} = a^2 /3
        # hance we scale uniform distribution appropriately:
        
        target_variance = 2.5 / ( 0.25 * dim + 1)
        #uniform_limit = np.sqrt( 3* target_variance )        
            
        # W = uniform_limit*rnd.rand(dim,num) - uniform_limit # uniformly from [-1, 1] interval
        # b = uniform_limit*rnd.rand(num) - uniform_limit     # uniformly from from [-1, 1] interval
        
        W = rnd.randn(dim,num) * np.sqrt( target_variance )  # Gaussian distribution with var - target_variance
        b = rnd.randn(num) * np.sqrt( target_variance )     # Gaussian distribution with var - target_variance
        
        
        return (W,b)


    @staticmethod
    def _init_rbf_weights_def(num, data_n, num_in_dist=None, 
                              metric_name='euclidean'):
        """ 
        Initialize sigmoidal weights in a default way
        Static method.
    
        Input:
            num - number of neurons required
            data_n - normalized data
            num_in_dist -number of data points to use in distance calulations
            metric_name - metric name in function pdist.
        Output:
            centers - matrix of neuron centers (rowwise)
            stds - scale parameters (stds in case of euclidean distance)
        """
        
        
        n,d = data_n.shape # number of datapoints and dimensionality        
        
        if not num_in_dist is None:
            distances = dist.pdist(data_n[rnd.choice(n, size=num_in_dist, 
                                        replace=False), : ],  metric_name )        
        else:
            distances = dist.pdist(data_n)                                             
                                                     
        a_low = mlab.prctile(distances,20)                                             
        a_high = mlab.prctile(distances,80)          
        
        stds = rnd.rand( np.min( (num,n)) )*(a_high - a_low) + a_low # stds of gaussian neurons
        
        centers = data_n[ rnd.choice(n, size=np.min( (num,n)) , replace=False) ,: ]
                
        return (centers,stds)
        
        
    @staticmethod
    def _create_sigmoid_neurons(data_n,W,b):
        """
        Produce matrix of neurons from the given weights
        
        Input:
            data_n - normalized data (rowwise). Training or test
            W - neuron weights (columnwise)
            b - array of additive terms
        Output:
            Neuron Matrix - column correspond to neurons, - rows to samples 
        """
    
    # 2 / (1 + exp(-2x)) - 1 = tanh(x) - this form is equivalent to tanh     
        #return (sp.dot(data_n, W) + b)
        #return 2. / ( np.exp( -2. * (np.dot(data_n, W) + b) ) + 1. ) - 1.
        return np.tanh( np.dot(data_n, W) + b )
    
    @staticmethod
    def _create_rbf_neurons(data_n,W,stds, metric_name='euclidean'):
        """
        Produce matrix of neurons from the given weights
        
        Input:
            data_n - normalized data (rowwise). Training or test
            W - neuron centers(rowwise)
            stds - scale parameters (stds in case of euclidean distance)
            metric_name - metric name in function pdist.
        Output:
            Neuron Matrix - column correspond to neurons, - rows to samples
        """
    
        return np.exp( dist.cdist(data_n,W, metric_name) / (2. * stds**2 ) )
        
 
class TikhELM(ELM):
    def __init__(self, X, Y, neurons_dict):
        """
        Constructor is the same as for ELM.
        """
        super(TikhELM,self).__init__(X, Y, neurons_dict)
        
        self.model.tikh = True # Indicator of Tikhonov regularization
        self.model.lamda = None # value of regularization parameter
        self.model.type = 'tikh-elm'
        
    def train(self, lamda=None):
        """
        Train the regularized ELM
        """
        
        H = np.hstack( (np.oneas( (self.H.shape[0],1) ) ,self.H) ) # include column of ones
        
        U, S, Vt = la.svd(H, full_matrices = False, check_finite = True)
                

        res = minimize_scalar(lambda lamda: loo_error(None,self.data.d_out_n,'press', U,S,lamda), 
                              bounds= (0.0,1.0), method='Brent')
        # Use these bounds because error functions are normalized with respect
        # to number of points.
      
        if not res.success:
            raise ValueError("Thikh ELM optimization failure: %s" % 
                            res.message)
        else:
            self.lamda = res.x
            
            n = y.shape[0] # total number of points    
    
            
            S2 = np.power(S,2)
            Sl = S2 + n*self.lamda # parameter lambda normalized with respect
                                  # to number of of points                             
            S = S2 / Sl
            
            S = np.diag(S) # construct the matrix from the diagonal 
            
            A = np.dot( np.dot(V,S), U.T)            
        
            self.model.coeffs = np.dot( A, self.train.d_out )
            self.model.num_rank = min(self.train.d_inp.shape(0),
                                        self.train.d_inp.shape(1))

            self.model.trained = True

            
epsilon1 = 1e12
class OP_ELM(ELM):
    def __init__(self,**kwargs):
        """
        Input:
            {'neurons':  neurons_dict} - how many neurons to take. There are 3 types
                           of keys possible: sigmoid, rbf, linear. The value is how many neurons
                           of this type to take. For linear neurons the dictionary value does
                           not matter.
                           It is possible to pass also neurons structure, generated
                           somewhere else.
            {'loo_evaluation_step': int } - how often perform loo validation
                
                
        Also, see description in ELM class.
        """
        super(OP_ELM,self).__init__(**kwargs)
        
        self.model.type = 'op-elm' # type of the model 
        self.model.reg_par_optim_type = None
        
        if 'loo_evaluation_step' in kwargs:
            self.model.loo_evaluation_step = kwargs['loo_evaluation_step']
        else:
            self.model.loo_evaluation_step = 5
            
    def train(self):
        """
        Training of OP-ELM Model
        """
        # This part is supposed to be similar for OP-ELM, TROP-ELM ->
        if not self.data_set:
            raise ValueError("OP_ELM.train: Data is not set. Training is impossible!")        
        
        # normalize neurons matrix        
        Y = self.data.d_out_n         
        Y_dim = Y.shape[1]        
        
        mrsr_obj = mrsr.Mrsr(self.H, Y,normalize = True) # Only Y is already normalized, neurons are not normalized
        res = mrsr_obj.run_mrsr()
        neurons_order = res[1]
     
        H = np.hstack( (np.ones( (self.H.shape[0],1) ) ,self.H[:, neurons_order]) ) # include column of ones
        
        total_neurons_num = H.shape[1] - 1 # total number of neurons (excluding column of ones)
        
        if total_neurons_num <= (self.data.X_dim + 1):
            # Ensure that the test number of neurons is not higher than the number of neurons.
            # This might happen if too few iterations are done by MRSR and linear neurons are included.
            test_neurons_num = range(1, total_neurons_num + 1)                
        else:    
            if hasattr(self.neurons, 'linear'): # there are linear neurons
                # In linear model assume that the first (self.train.dim + 1) neurons
                # can be all valid                                          
                nonlin_neurons_num = total_neurons_num - (self.data.X_dim + 1) # Greater than zero
                assert( nonlin_neurons_num > 0)
                loo_test_gap =  int( np.ceil(nonlin_neurons_num/10.0) )
                if (loo_test_gap > self.model.loo_evaluation_step):
                    loo_test_gap = self.model.loo_evaluation_step 
                    
                test_neurons_num = range(1,self.data.X_dim + 1) + range(self.data.X_dim + 1, total_neurons_num+1, loo_test_gap)  
            else:
                nonlin_neurons_num = total_neurons_num - 1 # Greater than zero
                assert( nonlin_neurons_num > 0)
                loo_test_gap =  int( np.ceil(nonlin_neurons_num/10.0) )
                if (loo_test_gap > self.model.loo_evaluation_step):
                    loo_test_gap = self.model.loo_evaluation_step  
                    
                test_neurons_num =  [1,] + range( 2, total_neurons_num+1, loo_test_gap )               
          
        # This part is supposed to be similar for OP-ELM, TROP-ELM <-          
          
        
        loo_growing = 0; current_min = np.Inf
        loo_errors = np.empty( (len(test_neurons_num), Y_dim+1) ) # in the first dimension the mean of all dimensions is stored
        for (ind,nn) in enumerate(test_neurons_num):
            H1 = H[:,0:nn]
            
            (U,S,Vh) = sp.linalg.svd(H1, full_matrices=False, overwrite_a=False, check_finite=False)
            if not getattr(self.model, 'reg_par_optim_type', False): # OP-ELM part
                if ( S[0]/S[-1] > epsilon1): 
                    loo_errors[ind,:] = np.Inf
                else:
                    responce_loo = loo_error(None,Y,'press', U,S,0.0) # loo for each column of output Y
                    loo_errors[ind,0] = np.mean( responce_loo )
                    loo_errors[ind,1:] = responce_loo
                    
                    
            else: # TROP-ELM part
                        
                optim_output = minimize_scalar(lambda lamda: np.sum( loo_error(None,Y,self.model.reg_par_optim_type, U,S,lamda) ), 
                              bounds= (0.0,1.0), method='Bounded')
                              
                lamda = optim_output.x
                responce_loo = loo_error(None,Y,self.model.reg_par_optim_type, U, S, lamda) # loo for each column of output Y
                loo_errors[ind,0] = np.mean( responce_loo)
                loo_errors[ind,1:] = responce_loo
                
            if ((ind > 0) and (loo_errors[ind,0] > loo_errors[ind-1,0] )) or (loo_errors[ind,0] == np.Inf):
                loo_growing += 1
            else:
                loo_growing = 0
                if loo_errors[ind,0] <  current_min:
                     current_min = loo_errors[ind,0]
                
            if (loo_growing > test_neurons_num[-1] * 0.1/ (test_neurons_num[-1] - test_neurons_num[-2]) ): # too many consequtive increases
                break # exit from iterations
                                        
            # if ((i>1) && ( (min(errloo(i,:)>var(y)*1.5) ) || ((min(errloo(i,:)>min(errloo)*1.5))))) - originally
            if (ind > 1) and ( (np.any( loo_errors[ind,:] > 1 *1.5) ) or (loo_errors[ind,0] >  current_min*1.5 )  ):
                break
            
        del loo_growing, current_min    
               
        
    
        min_ind = np.argmin( np.mean( loo_errors[0:ind+1,:], axis=1 ) ) # index of minimum number of neurons
        # TODO: warn if the last index is minimum - increase number of neurons in this case?
        
        # Having the right number of newrons train the final model 
        H1 = H[:,0:test_neurons_num[min_ind]]
        #res = ls_solve.ls_cof( H1, self.data.d_out_n, overwrite_a = False,
        #                 overwrite_b = False, check_finite = False)
        
        self.model.neurons_order = neurons_order
        self.model.neurons_num = test_neurons_num[min_ind] #  number of neurons ( including artificial zero column neuron )        
        self.model.neurons_names = [ self.neurons.types[i] for i in neurons_order ][ 0:(test_neurons_num[min_ind]-1) ]        
        
        res = self.solve_elm_ols(H1, Y)
                
        self.model.coeffs = res[0] # model coefficients
        self.model.num_rank = res[1]
        self.model.lamda = res[2]
        self.model.optim_output = res[3] # output of lamda optimization, used by TROP-ELM        
        
        self.model_built = True 
      
    def solve_elm_ols(self,H, Y):
        """
        This method solves the OLS problem after neurons have been sorted
        and the optimal number of neurons have been selected.
        This method works for both OP-ELM and TROP-ELM where additional regularization is applied
        
        """        
        
        if getattr(self.model, 'reg_par_optim_type', False): # TROP-ELM regularization is applied
            optim_type = self.model.reg_par_optim_type 
            
            (U,S,Vh) = sp.linalg.svd(H, full_matrices=False, overwrite_a=False, check_finite=False)
            
            lamda = None; optim_output = None 
    
            optim_type = self.model.reg_par_optim_type
            optim_output = minimize_scalar(lambda lamda: np.sum( loo_error(None,Y, optim_type, U,S,lamda) ), 
                              bounds= (0.0,1.0), method='Bounded')
            lamda = optim_output.x
           
            res = solve_ols_svd(U,S,Vh, Y, lamda )
            
            return (res[0], res[1], lamda, optim_output) # coefficients, numerical rank, lamda, optimization outputs
            
        else: # OP-ELM no regularization is applied
            res = ls_solve.ls_cof( H, Y, overwrite_a = False,
                             overwrite_b = False, check_finite = False)
            
            lamda = None 
            return (res[0], res[2], lamda, None) # coefficients, numerical rank, lamda, optimization params

    def predict(self, X_pred, Y_true=None):
        """
        Predict the values using
        
        Input:
            X_pred - data for which predictions are needed
            Y_true - true values if mse computation is needed
        """
       
        if not self.model_built:
            raise ValueError("Predictions can't be made. The model is not \
                              trainded yet.")       
      
        # Normalization with known stds and means
        res = du.normalize( X_pred, None, self.data.x_means,self.data.x_stds ) 
        X_pred_n = res[0]
        
        H = self._apply_neurons(self.neurons, X_pred_n )
        H = np.hstack( (np.ones( (H.shape[0],1) ),\
            H[:, self.model.neurons_order[ 0:(self.model.neurons_num - 1) ] ] ) ) # include column of ones
        
              
        Y_pred  = np.dot( H, self.model.coeffs )        
        
        Y_pred = du.denormalize( Y_pred,  self.data.y_means,self.data.y_stds )
               
      
        if not Y_true is None:
            mse = np.mean( np.power( (Y_true - Y_pred) , 2 ), axis = 0 ) # mse for each dimension of Y
        else:
            mse = None            
            
        return (Y_pred, mse)  
        

def solve_ols_svd(U,S,Vh, Y, lamda = 0.0 ):
    """
    Solve OLS problem given the SVD decomposition
    
    Input:
        ! Note X= U*S*Vh and Y are assumed to be normalized, hence lamda is between 0.0 and 1.0.
    
        U, S, Vh - SVD decomposition
        Y - target variables
        lamda - regularization parameter. Lamda must be normalized with respect
                                          to number of samples. Data is assumed
                                          to be normalized, so lamda is between 0.0 and 1.0.
    """
    
    n_points = U.shape[0]
    machine_epsilon = np.finfo(np.float64).eps
    if (lamda is None) or ( lamda > 1.0) or ( lamda < 0.0) : # no lamda optimization happened or wrong optim results
        num_rank = np.count_nonzero(S > S[0] * machine_epsilon)  # numerical rank  
    
        S.shape = (S.shape[0],1)
        coeffs = np.dot( Vh.T, np.multiply( 1.0/S ,np.dot( U.T, Y ) ) )
        
    else:
        S2 = np.power(S,2)
        S2 = S2 + n_points*lamda # parameter lambda normalized with respect to number of points !!!
        S = S / S2
        S.shape = (S.shape[0],1)
        coeffs = np.dot( Vh.T, np.multiply( S ,np.dot( U.T, Y ) ) )
        
        num_rank = None # numerical rank is None because regularization is used
    
    return coeffs, num_rank

class Inc_TROP_ELM(OP_ELM):
    def __init__(self,**kwargs):
        """
        Input:
            {'neurons':  neurons_dict} - how many neurons to take. There are 3 types
                           of keys possible: sigmoid, rbf, linear. The value is how many neurons
                           of this type to take. For linear neurons the dictionary value does
                           not matter.
                           It is possible to pass also neurons structure, generated
                           somewhere else.
            {'loo_evaluation_step': int } - how often perform loo validation
              
                
        Also, see description in ELM class.
        """
        super(Inc_TROP_ELM,self).__init__(**kwargs)
        
        self.model.type = 'inc-trop-elm' # type of the model 
        self.model.reg_par_optim_type = None
        
        if 'reg_par_optim_type' in kwargs:
            if kwargs['reg_par_optim_type'] in ('press','gcv'):
                self.model.reg_par_optim_type = kwargs['reg_par_optim_type']
        else:
            self.model.reg_par_optim_type = None
            
        if 'svd_reorth_step' in kwargs:
            self.model.svd_reorth_step = kwargs['svd_reorth_step']
        else:
            self.model.svd_reorth_step = 5
            
    def train(self):
        """
        Training of Inc-OP-ELM Model
        """
        # This part is supposed to be similar for OP-ELM, TROP-ELM ->
        if not self.data_set:
            raise ValueError("Inc_OP_ELM.train: Data is not set. Training is impossible!")        
        
        # normalize neurons matrix        
        Y = self.data.d_out_n         

        total_neurons_num = self.H.shape[1] # total number of neurons
        
        if total_neurons_num <= (self.data.X_dim + 1):
            # Ensure that the test number of neurons is not higher than the number of neurons.
            # This might happen if too few iterations are done by MRSR and linear neurons are included.
            test_neurons_num = range(1, total_neurons_num + 1)                
        else:    
            if hasattr(self.neurons, 'linear'): # there are linear neurons
                # In linear model assume that the first (self.train.dim + 1) neurons
                # can be all valid                                          
                nonlin_neurons_num = total_neurons_num - (self.data.X_dim + 1) # Greater than zero
                assert( nonlin_neurons_num > 0)
                loo_test_gap =  int( np.ceil(nonlin_neurons_num/10.0) )
                if (loo_test_gap > self.model.loo_evaluation_step):
                    loo_test_gap = self.model.loo_evaluation_step 
                    
                test_neurons_num = range(1,self.data.X_dim + 1) + range(self.data.X_dim + 1, total_neurons_num+1, loo_test_gap)
                
            else:
                nonlin_neurons_num = total_neurons_num - 1 # Greater than zero
                assert( nonlin_neurons_num > 0)
                loo_test_gap =  int( np.ceil(nonlin_neurons_num/10.0) )
                if (loo_test_gap > self.model.loo_evaluation_step):
                    loo_test_gap = self.model.loo_evaluation_step  
                    
                test_neurons_num =  [1,] + range( 2, total_neurons_num+1, loo_test_gap )
                
            if (test_neurons_num[-1] != total_neurons_num):
                test_neurons_num = test_neurons_num + [total_neurons_num,]
                    
        # This part is supposed to be similar for OP-ELM, TROP-ELM <-          
          
        elm_mrsr = Elm_mrsr(self.H, Y,normalize = True)
     
        iteration_no, neurons_order, loo_errors = elm_mrsr.run_mrsr(test_neurons_num,  
            self.model.svd_reorth_step, self.model.reg_par_optim_type )
            
        del iteration_no
        
        min_ind = np.argmin( loo_errors[:,0] ) # index of minimum number of neurons
        # TODO: warn if the last index is minimum - increase number of neurons in this case?
        
        # Having the right number of newrons train the final model 
        H1 = np.hstack( (np.ones( (self.H.shape[0],1) ) ,self.H[:,neurons_order[0: test_neurons_num[min_ind]]]) ) # include column of ones
    
        self.model.neurons_order = neurons_order
        self.model.neurons_num = test_neurons_num[min_ind] + 1#  number of neurons ( +1 because we include artificial zero column neuron )        
        self.model.neurons_names = [ self.neurons.types[i] for i in neurons_order ][ 0:test_neurons_num[min_ind] ] # -1 is not used because 1-column is not counted       
        
        res = self.solve_elm_ols(H1, Y)
                
        self.model.coeffs = res[0] # model coefficients
        self.model.num_rank = res[1]
        self.model.lamda = res[2]
        self.model.optim_output = res[3] # output of lamda optimization, used by TROP-ELM        
        
        self.model_built = True 
      
class Elm_mrsr(object):
    """
    Class which implements multiresponce sparse regression.
    Input data X and Y are not changed instead new varibales are allocated in class.
    
    It is used in the Incremental OP-ELM. It is assumed that Y is already normalized.    
    
    Reference: 
       Timo Simila, Jarkko Tikka. Multiresponse sparse regression with
       application to multidimensional scaling. International Conference
       on Artificial Neural Networks (ICANN). Warsaw, Poland. September
       11-15, 2005. LNCS 3697, pp. 97-102.
    """
    
    epsilon1 = 1e-7
    epsilon2 = 1e-10
    epsilon3 = 1e-8
    
    def __init__(self, X, Y, normalize = True):
        """
        Constructor.
        
        Inputs:
            X - input variables data matrix (rows are samples)
            Y - output variables data matrix ( rows are samples, columns - variables )
            normalize - should the input data be normlized or not. Do not normalize
                        if the data is already normalized.
        """
    
        # with shape is required to guarantee that Y is a column vector ( until matrix implementation is done)
        if Y.shape[0] != X.shape[0]:
            raise AssertionError("MRSR. Number of samples of regressors and responces is different")
        
        # First need to normalize input data. Original X is not changed. It is copied instead.
        # Y must be normalized already and it is not chanched in this class.
        if normalize:
            X_norm, X_means, X_stds = du.normalize(X, ntype=0)  # zero means unit variance
            self.data_normalized = True
            self.X_means = X_means
            self.X_stds = X_stds
        
        else:
            self.data_normalized = False
            
            X_norm = X.copy()
            
        self.X = np.matrix( X_norm, copy = False )
        self.Y = np.matrix( Y, copy=False )
     
    def run_mrsr(self, test_neurons_num, reorth_step, reg_par_optim_type, max_iter_no=None):
        """
        Input:
            test_neurons_num - at which iteration numbers perform 
            reorth_step - how often perform reorthogonalization of SVD update. Usually should be equal 
            max_iter_no - maximal number of iterations
                      (Iteration count starts from 1)
        Output:
            iteration_no, active_set_indices, beta_matr, beta, mu
            
            iteration_no - actual number of performed iterations. It can be less than
                           max_iter_no because of inappropriate relations between
                           regressors and responces. For instance, some regressors
                           can be very correlated or responses can be orthogonal to some
                           regressors. In these cases determination of the next entering variable
                           becomes impossible and iterations stop.
                           
                           
            active_set_indices - indices of the selected regressors in the correct order. 
                                 If the iterations stops earlier (look above). Number
                                 of active_set_indices is usually (iteration_no+1),
                                 because it includes the last variable which is determined,
                                 but beta and mu is impossible to compute for it, because
                                 even sebsequent variables can't be determined.
            
            beta_matr - matrix with MRSR parameters for all iterations. Size is
                        (m * ((iteration_no-1)*p + p ), where m -number of regressors,
                        p - number of responses. To obtain coefficients on the i-th
                        iteration take the section beta_matr[:, (i-1)*p: ((i-1)*p + p) ]
                        
            beta - MRSR parameters of the last iteration, which are             
                   beta_matr[:, (li-1)*p: ((li-1)*p + p) ], where li is the number of last
                   iteration.
                   
            excluded_indices - indices excluded from consideration because of their 
                               correlation with other variables.
                               
            mu - MRSR projection of responses onto the space of regressors.
                        
        """
        
        T = self.Y # target responses. T - in the article
        X = self.X # This varible might be modified in the code (removing some variables)
                        
        (n,m) = X.shape # n - munber of samples, m - number of dimensions  
        (n,p) = T.shape # number of samples, p - responce dimensionality
        if max_iter_no is None:
            max_iter_no = m                

        # set of all sign combinations of order p, each combination in column of S_sign_set
        S_sign_set = np.asmatrix( [ v for v in itertools.product((-1,1), repeat=p) ] ).T
        zero_m_p_matrix = np.asmatrix( np.zeros( (m,p) ) ) # zero matrix needed for copying it, to increase speed. 
        
        mu = np.asmatrix( np.zeros( (n,p) ) ) # Current solution in the responce space. Y - in the article
        beta = zero_m_p_matrix.copy() # weights of the model. W - in the article         
        beta_matr = np.asmatrix( np.zeros( (m, max_iter_no*p ) ) )# Matrix of beta coefficients which
                                                                  # collects beta for different iterations.        
        XT = X.T * T # temporary matrix. needed to reduce computationss
        
        loo_ind = 0 # index of loo iteration
        loo_errors = np.empty( (len(test_neurons_num), p+1) ) # storage for loo errors
        loo_growing = 0 # how many points loo_errors are growing        
        current_loo_min = np.Inf        
        
        active_set_indices = [] # indices of variables in the active set
        excluded_indices = [] # indeces excluded from iteration
        new_active_ind = None # it is none on the first iteration
        iteration_no = 0 # iterations counter        
        exit_loop = False
        no_active_variables = m # number of variables to be considered. Decreased when 
                                # correlations are detected i. e. excluded_indices is modified
        while not exit_loop:
            
            if ( max_iter_no == iteration_no ):
                break
            
            if ( no_active_variables == iteration_no ):
                break            
            
            iteration_no += 1            
            
            # new_active_ind  go from previous iteration unless it is a first iteration !!!
            if iteration_no == 1:
                
                A_k = XT
                C = np.sum( np.abs( A_k ), axis=1 ) # current correlations
                C_max_corr = C.max()
                new_active_ind = np.nonzero( C == C_max_corr) # take only first element of a tuple

                max_corr_inds = np.nonzero( np.abs(C - C_max_corr) < self.epsilon1*n ) 
                max_corr_inds =  max_corr_inds[0].tolist()[0] # convert to list                     
                if ( len(new_active_ind[0]) > 1) or  len(max_corr_inds) > 1:
                    # There are corelated variables !!!                    
                    new_active_ind = new_active_ind[0][0,0] # from numpy array to integer
                    assert new_active_ind in max_corr_inds, "Consistency check"                   
                    if len(max_corr_inds) > 1:
                                          
                        # Indices of correlated variables
                        excluded_indices = list( set( max_corr_inds ).difference([new_active_ind,]) )
                                            
                        no_active_variables -= len(excluded_indices) # reduce maximum number of iteration 
                        
                   
                        
                        warnings.warn("""MRSR. Input variables are correlated.  
                                    Variables %s are not considered further.""" % excluded_indices,\
                                    RuntimeWarning)
                else:
                    new_active_ind = new_active_ind[0][0,0] # from numpy array to integer

                        
            # new_active_ind  go from previous iteration unless it is a first iteration !!!
            
            # update active variables ->                       
            active_set_indices.append(new_active_ind)     
            # update active variables <-
            
            not_active_inds =   np.setdiff1d(range(m), active_set_indices )# indices not in the active set
            not_active_inds =   np.setdiff1d(not_active_inds, excluded_indices )
            assert len(np.intersect1d(active_set_indices,excluded_indices )) == 0, "Excluded indices and active indeces overlap" # consistency check
            
             # For iteration we need to have: new_active_ind, mu                                                                                                             
            if not locals().get('inv_upd'):                
                inv_upd = nu.Inverse_Update()
                XTX = inv_upd.new_column( X[:, new_active_ind ] )
            else:    
                XTX = inv_upd.new_column( X[:, new_active_ind ], X=X[:,active_set_indices[0:-1] ] ) # New inverse matrix
            
            if not locals().get('svd_upd'):       
                tmp_matr = np.hstack( (np.ones( (X.shape[0] ,1) ), X[:, new_active_ind ] ) )                
                (Ut,St,Vt) = sp.linalg.svd(tmp_matr, full_matrices=False, overwrite_a=False, check_finite=False)
                
                svd_upd = svd_update.SVD_updater( Ut, St, Vt, update_V = True, reorth_step=reorth_step )
                
            else:    
                svd_upd.add_column( np.asarray(X[:, new_active_ind ]) )
                
            # Compute OLS solution with selected the regressors ->
            Xa = X[:, active_set_indices ] # active variables of X            
            W_dense_next = XTX * XT[active_set_indices,:] # W_{\bar}_{k+1} in the article but dense
            
            W_sparse_next = zero_m_p_matrix.copy() # W_{\bar}_{k+1} in the article but sparse
            W_sparse_next[ active_set_indices, :] = W_dense_next # W_{\bar}_{k+1} in the article but sparse
            
            Y_ols = Xa * W_dense_next            
            # Compute OLS solution with selected the regressors <-
                                  
            # Compute step gamma for mu update and simultaneously new_active_ind for next iteration ->
             
            if iteration_no != no_active_variables: # not the last variable
            
                if iteration_no != 1: # for the first iteration this has been done
                # Compute current correlations (without new variables) and do some consistency checks -> 
                # mu - current respoce solution produced by current active variables: Y_{k} in the article
                    
                    A_k = X.T * (T - mu)            # temp variable needed to compute next entering regressor
                                                    # a_j^{k} in the article as rows in A_k 
                                                    # coincide with computation of correlations                       
                    C = np.sum( np.abs( A_k ), axis=1 ) # current correlations
             
                    C_max_corr = C.max()
        
                    
                    max_corr_inds = np.nonzero( np.abs(C - C_max_corr) < self.epsilon1*n ) 
                    max_corr_inds =  max_corr_inds[0].tolist()[0] # convert to list
                # max_corr_inds[0] - beause it is obtained from matrix, we need to discard of the second tuple
               
                if  (C_max_corr < self.epsilon2):
                    #raise AssertionError("Lars. Some regressors are completely orthogonal to the response.")
                    warnings.warn("MRSR. Some regressors are completely orthogonal to the response.",\
                                    RuntimeWarning)
                    iteration_no -= 1
                    break
                
                # active set indices and indices of maximum correlations must coincide, but in practice they
                # sometimes do not. Probably numerical effect.
                active_but_not_max = np.setdiff1d(active_set_indices, max_corr_inds )
                if (len( active_but_not_max ) != 0):
                    warnings.warn("MRSR. Some active indices are not maximal correlation indices.\
                    This is due to numerical reasons, further processing doesn't make sence. epsilon1 affects this.",RuntimeWarning)
                    iteration_no -= 1
                    break
                    
    #                if (len( active_bit_not_max ) != 0):
    #                     import scipy.io as io
    #                     file_name = './mrsr_error.mat'
    #                     dct = {}; dct['X'] = self.X; dct['Y'] = self.Y
                         #io.savemat(file_name, dct)
                         #raise ValueError("MRSR error caught, see 'mrsr_error.mat' ")
                
                # Compute current correlations (without new variables) and do some consistency checks <- 
                Xb = X[:, not_active_inds]
                B_k = Xb.T * (Y_ols - mu)     # temp variable needed to compute next entering regressor
                                                  # b_j^{k} in the article as rows in B_k 
                                                  # resembles the computation of A_k
                
                A_k = A_k[ not_active_inds ,:] # This is equivalent to: A_k = Xb.T * (T - mu), but A_k already computed                  
                    
                search_matr = np.divide( C_max_corr + A_k * S_sign_set, C_max_corr + B_k * S_sign_set )
                search_matr_max = np.nanmax( search_matr )
                search_matr[ search_matr <= 0] = search_matr_max # remove negative values
                del search_matr_max
                
                gamma_min = search_matr.min()
                gamma_ind = np.nonzero(  (np.abs( search_matr - gamma_min ) < self.epsilon3)  )[0].tolist()[0] # convert to list
                if len(gamma_ind) > 1: # Check that minimum is reached at only one point                   
                   
                    # We need set here because there can be more than one max in a row, hence indices can be not unique
                    gamma_ind = list( set( gamma_ind ) ) # unique indices
                    if len(gamma_ind) < not_active_inds.size: # not all inds have the same gamma
                        
                        current_exclude = not_active_inds[ gamma_ind[1:] ]
                        excluded_indices.extend( current_exclude.tolist() )                        
                        gamma_ind =  gamma_ind[0] # convert to integer
                        
                        warnings.warn("""MRSR. Several gammas have the same value. 
                                       Variables %s are exluded from consideration.""" % current_exclude,
                                       RuntimeWarning)
                        
                        no_active_variables -= len(current_exclude)
                    else: # all the rest indices have the same gamma
                        warnings.warn("MRSR. All the gammas have the same value. Badly conditioned matrix, or some regressors are orthogonal to respose.",\
                                    RuntimeWarning)
                        #iteration_no -= 1
                        #break
                        gamma_ind =  gamma_ind[0] # convert to integer
                        exit_loop = True # exit after considering the last variable
                else:
                    gamma_ind = gamma_ind[0] # convert to integer
                
                # row of gamma_ind  determines entering variable
                new_active_ind = not_active_inds[gamma_ind]# new_active_ind for next iteration    
                
                
                mu = mu + gamma_min * ( Y_ols - mu ) # update mu on this iteration
                #mu = (1.0 - gamma_min) * mu + gamma_min *  Y_ols  # update mu on this iteration
                beta = (1.0 - gamma_min) * beta + gamma_min * W_sparse_next
                # Compute step gamma for mu update and simultaneously new_active_ind for next iteration <-
                
            else:
                mu = Y_ols # last value of mu equal Y_ols
                beta = W_sparse_next
                 
            beta_matr[:, (iteration_no-1)*p: ((iteration_no-1)*p + p) ] = beta
            
            
            # loo error computation ->            
            if (iteration_no == test_neurons_num[loo_ind]):
                
                
                Ut,St,Vt = svd_upd.get_current_svd() # Vt is none here because it is not updated
                            
                if reg_par_optim_type is None:
                    #(U,S,Vh) = sp.linalg.svd(H1, full_matrices=False, overwrite_a=False, check_finite=False)
                    if ( St[0]/St[-1] > epsilon1): 
                        loo_errors[loo_ind,:] = np.Inf
                    else:
                        responce_loo = loo_error(None,T,'press', Ut,St,0.0) # loo for each column of output Y
                        loo_errors[loo_ind,0] = np.mean( responce_loo )
                        loo_errors[loo_ind,1:] = responce_loo
                else:
                    optim_output = minimize_scalar(lambda lamda: np.sum( loo_error(None,T,reg_par_optim_type, Ut,St,lamda) ), 
                              bounds= (0.0,1.0), method='Bounded')
                    lamda = optim_output.x
                    responce_loo = loo_error( None,T,reg_par_optim_type, Ut,St,lamda ) # loo for each column of output Y
                    loo_errors[loo_ind,0] = np.mean( responce_loo )
                    loo_errors[loo_ind,1:] = responce_loo
                    
                # Discuss exit criteria with Yoan
                #if (loo_ind > 1) and ( (np.any( loo_errors[loo_ind,:] > 1 *1.5) )    or  (np.any( loo_errors[loo_ind,:] > np.min(loo_errors[0:loo_ind,:], axis=0) *1.5) )  ):
                #    break
                if ((loo_ind > 0) and (loo_errors[loo_ind,0] > loo_errors[loo_ind-1,0] )) or (loo_errors[loo_ind,0] == np.Inf):
                    loo_growing += 1
                else:
                    loo_growing = 0
                    if loo_errors[loo_ind,0] <  current_loo_min:
                        current_loo_min = loo_errors[loo_ind,0]
                        
                if ( loo_growing > test_neurons_num[-1] * 0.1/ (test_neurons_num[-1] - test_neurons_num[-2]) ):
                    exit_loop = True # exit from mrsr iterations
                    
                if (loo_ind > 1) and ( (np.any( loo_errors[loo_ind,:] > 1 *1.5) ) or (loo_errors[loo_ind,0] >  current_loo_min*1.5 )  ):
                    exit_loop = True # exit from mrsr iterations
            
                loo_ind += 1
            # loo error computation <-   
            
        if (iteration_no != m): # if the function quits earlier reduce also the beta_matr
            beta_matr = beta_matr[:,0:((iteration_no-1)*p + p) ]
        
        loo_errors = loo_errors[0:loo_ind,:]            
            
        return iteration_no, active_set_indices, loo_errors            



class TROP_ELM(OP_ELM):
    """
    Implements TROP-ELM
    """
    def __init__(self,**kwargs):
        """
        Input:
            {'neurons':  neurons_dict} - how many neurons to take. There are 3 types
                           of keys possible: sigmoid, rbf, linear. The value is how many neurons
                           of this type to take. For linear neurons the dictionary value does
                           not matter.
                           It is possible to pass also neurons structure, generated
                           somewhere else.
            {'reg_par_optim_type': ('press','gcv') } - type of searching of regularization
                parameter.
        """
        
        super(TROP_ELM,self).__init__(**kwargs)
        
        self.model.reg_par_optim_type = None
        if 'reg_par_optim_type' in kwargs:
            if kwargs['reg_par_optim_type'] in ('press','gcv'):
                self.model.reg_par_optim_type = kwargs['reg_par_optim_type']
        
        if self.model.reg_par_optim_type is None:
            raise ValueError("TROP_ELM.__init__: regularizarion parameter optimization must be set")
            
        self.model.type = 'trop-elm' # type of the model 
         

class ELM_TS(ELM):
    """
    Class provides additional functionality for solving least-squares problem for
    time series i.e iterations to make algorithm stable with respect to errors.
    Similar to the paper of Alexander Dyakonov.
    """
    def set_data(self,X,Y,normalize=True):
        """        
        Function shadows the original function in the ELM class. The main
        goal of this function is to check that regressor size and prediction
        dim are suitable to time series iterations.
        """
        
        if (X.shape[1] != Y.shape[1]):
            if (Y.shape[1] != 1):                        
                raise ValueError("""Time series iterations can be performed only
                                    if regressor size equal prediction dim., or 
                                    prediction dim. equals one""")
                                    
        if (X.shape[1] == 1):                                     
              raise ValueError("Code might work improperly when X_dim = 1" )
                                                                
        super(self.__class__,self).set_data(X,Y,normalize)
    
    def set_ts_pred_iteration( self, ts_prediction_iteration ):
        self.model.ts_prediction_iteration = ts_prediction_iteration
    
    def solve_elm_ols(self, H, Y):
        
        kwargs = self.model.orig_kwargs
        
        if not 'ts_iter_params' in kwargs:
            raise ValueError("ELM_TS.__init__: Time series optimization params are missing")
        else:
            ts_iter_params = kwargs['ts_iter_params']
            if not 'iter_num' in ts_iter_params:
                # Number of iterations to be performed
                raise ValueError("ELM_TS.__init__: Time series optimization number of iterations is missing")
            else:
                iter_num = ts_iter_params['iter_num']
                self.model.ts_iter_num = iter_num
                assert (iter_num >= 0), " This parameter must be positive"
                
            if hasattr( self.model, 'ts_prediction_iteration'):
                prediction_iteration = self.model.ts_prediction_iteration
            else:
                prediction_iteration = 1                     
            
            self.set_ts_pred_iteration( prediction_iteration )
                        
        res = self.perform_iterations(H,Y,prediction_iteration, iter_num )
                        
        return res # coefficients, numerical rank, lamda, optimization params


    def  perform_iterations(self, H,Y,prediction_iteration, iter_num ):
        """
        Function which actually peform iterations for time series.
        """
        
         # first iteration
        res = super(self.__class__,self).solve_elm_ols(H,Y) # might call form ELM, OP-ELM, TROP-ELM
                         
        predictions = np.dot( H, res[0])
        
        X_dim = self.data.X_dim
        Y_dim = Y.shape[1]

        # index of the position of the first element in Y in the array X        
        first_elem_shift = X_dim + (prediction_iteration - 1) * Y_dim 
                
        if (first_elem_shift >= H.shape[0] ): # iterations are impossibe, not enough elements in H
            pass                            
        else:
            Y_shifted = Y[first_elem_shift:,:]
            for it_no in xrange(1,(iter_num+1)):
                
                if (X_dim != Y_dim) and (Y_dim == 1): # Case when output is one dimensional
                    if (predictions.size >= X_dim+ Y_dim*prediction_iteration):
                        import time_series as ts 
                        tmp_ts = ts.TimeSeries( predictions )
                        predictions, tmp_y = tmp_ts.ts_matrix(X_dim, Y_dim*prediction_iteration,\
                                                only_b = False, num_b_col = Y_dim, with_nans = False )
                    else:
                        break
                    assert (predictions.shape[0] == Y_shifted.shape[0]), "Consinstency check"                                              
                    del tmp_ts, tmp_y                     
                    
                H1 = self._apply_neurons( self.neurons, predictions )
                
                if (self.model.type == 'trop-elm') or (self.model.type == 'op-elm'):                  
                    assert (hasattr(self.model,'neurons_order') and hasattr(self.model,'neurons_num')), "These attributes \
                    must be present"
                    
                    H1 = H1[:, self.model.neurons_order[ 0:(self.model.neurons_num - 1) ] ]
                    
                    # From OP-ELM predict:
                    # H = np.hstack( (np.ones( (H.shape[0],1) ),\
                    # H[:, self.model.neurons_order[ 0:(self.model.neurons_num - 1) ] ] ) ) # include column of ones
                
                H1 = np.hstack( (np.ones( (H1.shape[0],1) ) ,H1) ) # include column of ones
                H1 = H1[0:Y_shifted.shape[0],:]
                
                H2 = np.vstack( (H,H1) )
                Y2 = np.vstack( (Y, Y_shifted) )
                
                assert (H2.shape[0] == Y2.shape[0]), "Sizes of a matrix must be equal here"
                
                res = super(self.__class__,self).solve_elm_ols(H2,Y2)            
                
                predictions = np.dot( H2, res[0])           
                predictions = predictions[0:H.shape[0],:] # return to original predictions
        return res

class LinearModel_TS(LinearModel,ELM_TS):
    
    def solve_lin_ols(self, X, Y, U=None, S=None, Vh=None ):
        return ELM_TS.solve_elm_ols(self,X,Y)
        
    def set_data(self,X,Y,normalize=True): 
        ELM_TS.set_data(self,X,Y,normalize)
    
    def perform_iterations( self, X,Y,prediction_iteration, iter_num ):
        """
        Function which actually peform iterations for time series.
        """        
        # First iteration
        (U,S,Vh) = sp.linalg.svd(X, full_matrices=False, overwrite_a=False, check_finite=False)
        res = LinearModel.solve_lin_ols( self, X, Y, U,S,Vh ) 
        
        predictions = np.dot( X, res[0])
        
        X_dim = X.shape[1] - 1 # (X_dim-1) - subtract coulmn of ones from the dimensionality
        Y_dim = Y.shape[1]
        # index of the position of the first element in Y in the array X
       
        first_elem_shift = X_dim + (prediction_iteration - 1) * Y_dim         
        
        if ( first_elem_shift >= X.shape[0] ):  # iterations are impossibe, not enough elements in H
            pass                            
        else:
            Y_shifted = Y[first_elem_shift:,:]
            for it_no in xrange(1,(iter_num+1)):
                
                if (X_dim != Y_dim) and (Y_dim == 1): # Case when output is one dimensional
                    if (predictions.size >= X_dim+ Y_dim*prediction_iteration):
                        import time_series as ts 
                        tmp_ts = ts.TimeSeries( predictions )
                        predictions, tmp_y = tmp_ts.ts_matrix(X_dim, Y_dim*prediction_iteration,\
                                                only_b = False, num_b_col = Y_dim, with_nans = False )
                    else:
                        break
                    assert (predictions.shape[0] == Y_shifted.shape[0]), "Consinstency check"                                              
                    del tmp_ts, tmp_y                
                
                X1 = predictions[0:Y_shifted.shape[0],: ]                
                X1 = np.hstack( (np.ones( (X1.shape[0] ,1)) ,X1) ) # add columns of ones                
                
                X2 = np.vstack( (X,X1) )
                Y2 = np.vstack( (Y, Y_shifted) )
                
                assert (X2.shape[0] == Y2.shape[0]), "Sizes of a matrix must be equal here"

                (U,S,Vh) = sp.linalg.svd(X2, full_matrices=False, overwrite_a=False, check_finite=False)
                res = LinearModel.solve_lin_ols(self, X2, Y2, U,S,Vh)                   
                
                predictions = np.dot( X2, res[0])           
                predictions = predictions[0:X.shape[0],:] # return to original predictions
                
        return res
             
class OP_ELM_TS(OP_ELM, ELM_TS):
    """
    Class provides additional functionality for solving least-squares problem for
    time series i.e iterations to make algorithm stable with respect to errors.
    As in the paper of Alexander Dyakonov.
    """
    def set_data(self,X,Y,normalize=True): 
        
        if (X.shape[1] != Y.shape[1]):
            if (Y.shape[1] != 1):                        
                raise ValueError("""Time series iterations can be performed only
                                    if regressor size equal prediction dim., or 
                                    prediction dim. equals one""")                                    
        if (X.shape[1] == 1):                                     
              raise ValueError("Code might work improperly when X_dim = 1" )
              
        OP_ELM.set_data(self,X,Y,normalize)    
    
    def solve_elm_ols(self, H, Y):
        return ELM_TS.solve_elm_ols(self, H, Y)
            
class TROP_ELM_TS(TROP_ELM, ELM_TS):
    """
    Class provides additional functionality for solving least-squares problem for
    time series i.e iterations to make algorithm stable with respect to errors.
    As in the paper of Alexander Dyakonov.
    """
    def set_data(self,X,Y,normalize=True): 
        
        if (X.shape[1] != Y.shape[1]):
            if (Y.shape[1] != 1):                        
                raise ValueError("""Time series iterations can be performed only
                                    if regressor size equal prediction dim., or 
                                    prediction dim. equals one""")                                    
        if (X.shape[1] == 1):                                     
              raise ValueError("Code might work improperly when X_dim = 1" )        
        
        TROP_ELM.set_data(self,X,Y,normalize)
        
    def solve_elm_ols(self, H, Y):
        return ELM_TS.solve_elm_ols(self, H, Y)
