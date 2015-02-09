# -*- coding: utf-8 -*-
"""
Module which implements Multi-Response Sparse Regression Algorithm, and
Lars algorithm.

MRSR algorithm differs from the Lars algorithm in that it can handle
multidimensional output variables (responces).

Author: Alexander Grigorievskiy, Aalto University, 2013.
"""

import numpy as np
import itertools
import warnings

# ------------ Non standard imports ------------   ->
from utils import data_utils as du
from utils import numeric_utils as nu
# ------------ Non standard imports ------------   ->



epsilon1 = 1e-7 # This epsilon is used in Lars and Mrsr algorithms to compute indices of 
                 # current maximum correlation variables

epsilon2 = 1e-10 # This epsilon is used in Lars and Mrsr algorithms to decide that
                 # correlations are too small 

epsilon3 = 1e-8  # This epsilon is used in Lars and MRSR algorithms to check whether
                 # difference in gamma is too small, which means that next variable choise is
                 # ambiguous

def change_epsilon(eps1=None, eps2=None, eps3=None):
    """
    By this function one can change the default values of epsilon parameters.
    
    Input:
        eps1: new value of epsilon1
        eps2: new value of epsilon2
        eps3: new value of epsilon3
    
    Output:
        None
    """
    global epsilon1, epsilon2, epsilon3

    if not eps1 is None:
        epsilon1 = eps1
    
    if not eps2 is None:
        epsilon2 = eps2
    
    if not eps3 is None:
        epsilon3 = eps3


class Mrsr(object):
    """
    Class which implements multiresponce sparse regression.
    
    Input data X and Y are not changed instead new varibales are allocated in class.
    
    
    Reference: 
       Timo Simila, Jarkko Tikka. Multiresponse sparse regression with
       application to multidimensional scaling. International Conference
       on Artificial Neural Networks (ICANN). Warsaw, Poland. September
       11-15, 2005. LNCS 3697, pp. 97-102.
    """
    
    def __init__(self, X, Y, normalize = True):
        """
        Constructor.
        
        Inputs:
            X - input variables data matrix (rows are samples)
            Y - output variables data matrix ( rows are samples, columns - variables )
            normalize - should the input data be normlized or not. Do not normalize
                        if the data is already normalized.
        """
        
        # copied later
        #X = np.matrix(X,copy = True) # make copy of the input data, hence it is not changed        
        #Y = np.matrix(Y,copy = True) # make copy of the input data, hence it is not changed
        
        if (Y.shape[0] == 1) and (Y.shape[1] > 1): #  row vector
            # It is assumed that one sample regression is not possible, hence transform
            # row vector into column vector
            Y = Y.T
        
        # with shape is required to guarantee that Y is a column vector ( until matrix implementation is done)
        if Y.shape[0] != X.shape[0]:
            raise AssertionError("MRSR. Number of samples of regressors and responces is different")
        
        # First need to normalize input data 
        if normalize:
            X_norm, X_means, X_stds = du.normalize( X, ntype=0 ) # zero means unit variance
        
            Y_norm, Y_means,Y_stds = du.normalize( Y, ntype=0 ) # zero means unit variance
            
            self.data_normalized = True
            self.X_means = X_means
            self.X_stds = X_stds
            self.Y_means = Y_means
            self.Y_stds = Y_stds
        
        else:
            self.data_normalized = False
            
            X_norm = X.copy()
            Y_norm = Y.copy()
            
        self.X = np.matrix( X_norm, copy=False )
        self.Y = np.matrix( Y_norm, copy=False )
     
    def run_mrsr(self, max_iter_no=None):
        """
        Input:
            iter_no - maximal number of iterations
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
        X = self.X # This varible IS NOT modified in the code (seems sos)
                        
        (n,m) = X.shape # n - number of samples, m - number of dimensions  
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

                max_corr_inds = np.nonzero( np.abs(C - C_max_corr) < epsilon1*n ) 
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
        
                    
                    max_corr_inds = np.nonzero( np.abs(C - C_max_corr) < epsilon1*n ) 
                    max_corr_inds =  max_corr_inds[0].tolist()[0] # convert to list
                # max_corr_inds[0] - beause it is obtained from matrix, we need to discard of the second tuple
               
                if  (C_max_corr < epsilon2):
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
                gamma_ind = np.nonzero(  (np.abs( search_matr - gamma_min ) < epsilon3)  )[0].tolist()[0] # convert to list
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
            
        if (iteration_no != m): # if the function quits earlier reduce also the beta_matr
            beta_matr = beta_matr[:,0:((iteration_no-1)*p + p) ]
            
        return iteration_no, active_set_indices, beta_matr, beta, excluded_indices, mu 


class Lars(object):
    """
    This calass implements Lars algorithm.
    
    Input data X and Y are not changed, new varibales are allocated in class.
    
    
    Reference:
    Efron B., Hastie T., Tibshirani R.
    Least Angle Regression, Annals of Statistics, 2004, 32.
    """
    
    def __init__(self, X, Y):
        """
        Constructor.
        
        Inputs:
            X - input variables data matrix (rows are samples)
            Y - output variables data. ( must be a vector )
        """
        
        # First need to normalize input data
        
        X_norm, X_means, X_stds = du.normalize(X, ntype=0)  # Temp !!!
        # X_norm, X_means, X_stds = normalize.normalize(X, ntype=2)  # zero means
        # unit length
        
        # Y_norm, Y_means,temp = normalize.normalize(Y, ntype=3) # zero means no length scaling
        Y_norm, Y_means,temp = du.normalize(Y, ntype=0) # Temp      
        
        self.X = np.asmatrix(X_norm)        
        self.Y = np.asmatrix( Y_norm.reshape(len(Y_norm),1) ) # manipulation 
        # with shape is required to guarantee that Y is a column vector ( until matrix implementation is done)

        self.X_means = X_means
        self.X_stds = X_stds
        self.Y_means = Y_means
    
    
    def run(self, max_iter_no=None):
        """
        
        Input:
            iter_no - maximal number of iterations
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
            
            mu - MRSR projection of responses onto the space of regressors.
                
        """
        
        orig_X = self.X
        y = self.Y
        X = orig_X.copy() # Create a copy of the original data, because this variable is modified.
                        
        (n,m) = X.shape # n - munber of samples, m - number of dimensions  
        if max_iter_no is None:
            max_iter_no = m
        
        active_set_indices = [] # indices of variables in the active set
        active_set_signs = np.array( [1,]*m , dtype = 'i' )# signs of active set variables

        mu = np.asmatrix((0,)*n, dtype='d').T # column vector of zeros. Initial values for mu
        beta = np.asmatrix( np.zeros( (m,1) ) ) # weights of the model. W - in the article         
        beta_matr = np.asmatrix( np.zeros( (m, max_iter_no ) ) )# Matrix of beta coefficients which
                                                               # collects beta for different iterations.          
        
        iteration_no = 0        
        exit_loop = False
        num_vars_selected = 0
        while not exit_loop:
            if (max_iter_no == iteration_no):
                break
            
            iteration_no += 1            
            
            # new_active_ind and new_sign go from previous iteration !!!
            # orig_X is now computed in the function beginning
            # orig_X = np.multiply(X, active_set_signs) # restore the original signs of variables
            
            # Compute current correlations (without new variables) and do some consistency checks ->
            # mu - current respoce solution produced by current active variables
            C = orig_X.T * (y - mu) # mu - current Lars estimate 
            
            abs_C = np.abs(C)
            C_max_corr = abs_C.max()                        
            max_corr_inds = np.nonzero( np.abs(abs_C - C_max_corr) < epsilon1 ) 

            # new_active_ind and new_sign go from previous iteration !!!
            if iteration_no == 1:            
                new_active_ind = np.nonzero( abs_C == C_max_corr) # take only first element of a tuple                
                
                if ( len(new_active_ind[0]) > 1) or  np.count_nonzero(np.abs(abs_C - C_max_corr) < epsilon1 ) > 1:
                    raise AssertionError("Lars first iteration. Matrix is too badly conditioned.")
                
                new_sign = np.sign( C[new_active_ind] )[0,0] # determine sign of best variable            
                new_active_ind = new_active_ind[0][0,0] # from numpy array to integer
            
            # update active variables ->
            num_vars_selected += 1
            X[:, new_active_ind ] = new_sign * X[:, new_active_ind ]
            
            active_set_indices.append(new_active_ind)
            active_set_signs[new_active_ind] = new_sign            
            # update active variables <-
            
            if  (C_max_corr < epsilon2):
                #raise AssertionError("Lars. Some regressors are completely orthogonal to the response.")
                warnings.warn("Lars. Some regressors are completely orthogonal to the response.",\
                                RuntimeWarning)
                iteration_no -= 1
                break
            # active set indices and indices of maximum correlations must coincide
            if len( np.setxor1d(active_set_indices, max_corr_inds[0].tolist()[0] ) ) != 0:
                #raise AssertionError("Lars. Something wrong with the indices of active set.")
                warnings.warn("Lars. Indices of active set and maximum correlation indices do no match. Regressor correlation might be the reason.",\
                                RuntimeWarning)
                iteration_no -= 1
                break    
            # signs of variables in active set and correlations must be equal
            if np.any( np.sign( C[max_corr_inds] ) != active_set_signs[max_corr_inds[0]] ):
                #raise AssertionError("Lars. Something wrong with the signs of active set.")
                warnings.warn("Lars. Something wrong with the signs of active set.",\
                                RuntimeWarning)
                iteration_no -= 1 
                break
            # max_corr_inds[0] - beause it is obtained from matrix, we need to discard of the second tuple
                
            # Compute current correlations (without new variables) and do some consistency checks <-             
            
            # For iteration we need to have: new_sign, new_active_ind, mu
                                                         
            # Compute new active dataset Xa, norm. constant Aa, and vector Ua, 
            # having index of next variable to include and sign of this variable ->                                    
            
            if not locals().get('inv_upd'):                
                inv_upd = nu.Inverse_Update()
                XaTXa = inv_upd.new_column( X[:, new_active_ind ] )
            else:    
                XaTXa = inv_upd.new_column( X[:, new_active_ind ], X=X[:,active_set_indices[0:-1] ] ) # New inverse matrix
            
            column_of_ones = np.matrix((1,)*num_vars_selected).T
            XaTXa_1 = XaTXa * column_of_ones
                
            Aa = ( 1/np.sqrt(column_of_ones.T * XaTXa_1) )[0,0] # convert to a number
            
            Ua = Aa * X[:, active_set_indices] * XaTXa_1
            
            del column_of_ones, XaTXa_1            
            # Compute new active dataset Xa, norm. constant Aa, and vector Ua <-
                                               
            # Compute step gamma for mu update and simultaneously new_sign and new_active_ind for next iteration ->
            a_vec =  orig_X.T * Ua            
                
            if iteration_no != m: # not the last variable    
                not_active_inds =   np.setdiff1d(range(m), active_set_indices )# indices not in the active set
                
                
                minus_part = np.divide( (C_max_corr - C[not_active_inds]) , ( Aa - a_vec[not_active_inds]) )
                plus_part = np.divide( (C_max_corr + C[not_active_inds]) , ( Aa + a_vec[not_active_inds]) )
                
                gamma_minus_min = minus_part[ minus_part > 0].min()
                gamma_plus_min = plus_part[ plus_part > 0].min()
            
                                
                if np.abs( gamma_minus_min - gamma_plus_min ) < epsilon3:
                   #raise AssertionError(" Lars. Strage gamma_plus_min can't be equal gamma_minus_min")
                    warnings.warn("Lars. gamma_plus equal gamma_mins. Badly conditioned matrix, or some regressors are orthogonal to respose.",\
                                RuntimeWarning)
                    iteration_no -= 1
                    break 
                
                elif gamma_minus_min < gamma_plus_min:
                    new_sign = 1 # new sign for the next iteration
                    gamma_vals =  minus_part
                    gamma_min = gamma_minus_min
                    
                elif gamma_minus_min > gamma_plus_min:      
                    new_sign = -1 # new sign for the next iteration
                    gamma_vals = plus_part
                    gamma_min = gamma_plus_min
                
                gamma_ind = np.nonzero( ( np.abs( gamma_vals - gamma_min) < epsilon3) )[0]
                if gamma_ind.size > 1: # Check that minimum is reached at only one point
                    #raise AssertionError("Lars. Several gammas have the same value. Badly conditioned matrix?")
                    warnings.warn("Lars. Several gammas have the same value. Badly conditioned matrix, or some regressors are orthogonal to respose.",\
                                RuntimeWarning)
                    iteration_no -= 1
                    break
                    
                del gamma_minus_min, gamma_plus_min, minus_part, plus_part
        
                mu = mu + gamma_min * Ua # update mu on this iteration
                new_active_ind = not_active_inds[gamma_ind][0,0] # new_active_ind for next iteration            
                # Compute step gamma for mu update and simultaneously new_sign and new_active_ind for next iteration <-
                
            else:
                mu = mu + np.abs( C[new_active_ind][0,0] ) / Aa * Ua # last value of mu equal y

            # Compute model coefficients beta iteratively ->
            XaTmu = X[:, active_set_indices ].T * mu 
            
            beta_tmp = XaTXa * XaTmu
            beta[ active_set_indices,:] = np.multiply( beta_tmp, active_set_signs[active_set_indices].reshape(len(active_set_indices) ,1) )
            beta_matr[:, (iteration_no-1) ] = beta
            # Compute model coefficients beta iteratively <-
            
        if (iteration_no != m): # if the function quits earlier reduce also the beta_matr
            beta_matr = beta_matr[:,0:iteration_no ]
            
        return iteration_no, active_set_indices, beta_matr, beta, mu


if __name__ == "__main__":
    import sklearn.datasets as ds
    import bottleneck as bn    
    import scipy.io as io
    
    
#    mr = Mrsr(X,Y)
#    res = mr.run_mrsr()    
    
#    BostonHousing = ds.load_boston()
#        
#    X0 = BostonHousing.get('data')
#    y0 = BostonHousing.get('target')
#    
#    if bn.anynan(X0) or bn.anynan(y0):
#        raise AssertionError('NaNs are not allowed')        
#        
#    X1 = X0.copy(); y1 = y0.copy()
#    
#    lars = Lars(X1,y1)
#    (X_1, asgn_1, ai_1, beta_1, mu_1, y_1,b_matr_1) = lars.run(max_iter_no=4)
#    
#    mrsr = Mrsr( X0, y0)
#    (X_2, ai_2, beta_2, mu_2, y_2, b_matr_2) = mrsr.run(max_iter_no=4)
#    
#    print "Difference in mu:   ", np.linalg.norm(( mu_1 - mu_2) ,ord=2)
#    print "Difference in beta:   ", np.linalg.norm(( beta_1 - beta_2) ,ord=2)

# Test of Lars and Mrsr ->
    for i in xrange(0,5):
        (X,Y) = ds.make_regression(n_samples = 1000, n_features = 7,n_informative=5, \
                       n_targets=2, bias = 2.0, effective_rank = 2)        
#        X = np.dot(X, np.random.rand(4,6))
#        X = np.hstack((X, 2*X[:,0].reshape(X.shape[0],1 ) ) )
        
    #dct = io.loadmat('./mrsr_test_0.mat')
    #X = dct['X']; Y = dct['Y']

#        lars = Lars(X,Y)
#        (it_1,ai_1,b_matr_1, beta_1,mu_1 ) = lars.run(max_iter_no=5)

        mrsr = Mrsr( X,Y)
        (it_2,ai_2,b_matr_2, beta_2,excl_vars_2, mu_2 ) = mrsr.run_mrsr()
        
        b_matr_d = np.linalg.norm(( b_matr_1 - b_matr_2) ,ord=2 )
        
        print "Iteration %i, var no. %i,    %i,%i" % (i,it_1, len(ai_1), len(ai_2) ) 
        if it_1 == it_2:
            if (b_matr_d > 1e-13):
                raise AssertionError("MRSR Test. mu difference is too high %f.",  b_matr_d)
            else:
                print "Iteration %i is fine." % (i,) 
        else:
            raise AssertionError("MRSR Test. Lars number of iteration - %i, MRSR - %i", it_1, it_2 )
        
#        file_name = './mrsr_test_%i.mat' % (i,)
#        dct = {}; dct['X'] = mrsr.X; dct['Y'] = mrsr.Y; dct['beta'] = b_matr_2
#        dct['act_ind'] = ai_2
#        dct = io.savemat(file_name, dct)
        
# Test of Lars and Mrsr <-

    


