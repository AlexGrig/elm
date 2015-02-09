# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 17:29:42 2013

@author: agrigori
"""

import numpy as np

# ------------ Non standard imports ------------   ->
import normalize
import extra_ls_solvers as lss

# ------------ Non standard imports ------------   ->

epsilon1 = 1e-10 # This epsilon is used in Lars algorithm to compute indices of 
                 # current maximum correlation variables


class Inverse_Update(object):
    """    
    Instance of this class provide the fuctions which updates the inverse
    of a Gram matrix. Even more general than inverse because diagonal matrix
    can be added. 
    
    Function returns update to this inverse
    ( X.T * X  + Diag )^(-1)    
    
    when new columns to matrix X is added.    
    
    
    Details are given in:
    Mark JL Orr. Introduction to Radial Basis Function Networks.
    Centre for Cognitive Science, University of Edinburgh, Technical
    Report, April 1996. 
    """
    
    def  __init__(self):   
        """
        Constructor
        """
        
        self.prev_inv = None # inverse from previous iteration
        self.prev_size = None #  size of the previous inverse (symmetric)
    
    def new_column(self, new_col, new_lambda=0, X=None, prev_inverse=None):     
        """
        Returns the updated inverse matrix.        
        
        Input: (matrix type (not new_lambda))
            new_col - new colum of matrix X
            new_lambda - (optional) new diagonal value
            X - old matrix X (without new column)
            prev_inverse - if we want to start not from the first iteration.
                            We provide already computed inverse matrix
        Output:
            New inverse matrix
        """
        
        if new_col.shape[0] == 1: 
            new_col = new_col.T # make sure this is column vector
    
        if prev_inverse:
            self.prev_inv = prev_inverse
            self.prev_size = self.prev_inv.shape[0]
        
        
        if self.prev_inv == None: # first iteration
            
            self.prev_inv = 1./ ( new_col.T * new_col + new_lambda)
            self.prev_size = 1
            
            return self.prev_inv
            
        else: # not the first iteration
            n = self.prev_size
            
            
            # Part 1 of the formula: add one more zero row and zero column
            P1 = np.hstack(  ( self.prev_inv, np.matrix( (0,)* n ,dtype='d').T ) )
            P1 = np.vstack( (P1, np.matrix( (0,)* (n+1),dtype='d')) )
            
            
            T = self.prev_inv * X.T # temporary expression
            
            
            delta = new_lambda + new_col.T * ( new_col - X * ( T * new_col) )
            delta = delta[0,0] # to make constant
            
            v = np.vstack(( T * new_col, -1)) # vector for part 2        
            
            # Part 2
            P2 = np.outer(v,v)
            
            self.prev_inv = P1 + 1/delta * P2
            self.prev_size = self.prev_inv.shape[0]
            
            return self.prev_inv            
            
            

class Lars(object):
    
    
    def __init__(self, X, Y):
        """
        Constructor.
        
        Inputs:
            X - input variables data matrix (rows are samples)
            Y - output variables data matrix ( now it is assumed to be a column matrix)
        """
        
        # First need to normalize input data
        
        X_norm, X_means, X_stds = normalize.normalize(X, ntype=2)  # zero means
        # unit length
        
        Y_norm, Y_means,temp = normalize.normalize(Y, ntype=3) # zero means no length scaling
        
        self.X = np.asmatrix(X_norm)        
        self.Y = np.asmatrix( Y_norm.reshape(len(Y_norm),1) ) # manipulation 
        # with shape is required to guarantee that Y is a column vector ( until matrix implementation is done)

        self.X_means = X_means
        self.X_stds = X_stds
        self.Y_means = Y_means
    
    
    def run(self, max_iter_no=None):
        """
        Now assume that the Y is a vector(clolumn)
        
        Input:
            iter_no - maximal number of iterations
        
        Output:
            orig_X - matrix of normalized data from this class for which lars algorithm ran
            active_set_signs - signs of the data variables (columns) 
            active_set_indices - order of the variables as they enter into model 
            beta - model coefficients beta, used later for prediction
            mu - result of OLS for selected variables. 
                Equal mu = np.multiply(orig_X, active_set_signs)[:, active_set_indices] * beta
                Used also for testing, compare this with original OLS solutions.
                
        """
        
        orig_X = self.X
        y = self.Y
        X = orig_X.copy() # Create a copy of the original data
                        
        (n,m) = X.shape # n - munber of samples, m - number of dimensions  
        
        active_set_indices = [] # indices of variables in the active set
        active_set_signs = np.array( [1,]*m , dtype = 'i' )# signs of active set variables

        mu = np.matrix((0,)*n, dtype='d').T # column vector of zeros. Initial values for mu

        iteration_no = 0        
        not_exit_loop = True
        num_vars_selected = 0
        while not_exit_loop:
            if ( max_iter_no and (max_iter_no == iteration_no)) or (m == iteration_no):
                break
            
            iteration_no += 1            
            
            # new_active_ind and new_sign go from previous iteration !!!
            # orig_X is now computed in the function beginning
            # orig_X = np.multiply(X, active_set_signs) # restore the original signs of variables
            
            # Compute current correlations (without new variables) and do some consistency checks -> 
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
            
            
            # signs of variables in active set and correlations must be equal
            if np.any( np.sign( C[max_corr_inds] ) != active_set_signs[max_corr_inds[0]] ):
                raise AssertionError("Lars. Something wrong with the signs of active set.")
            # max_corr_inds[0] - beause it is obtained from matrix, we need to discard of the second tuple
                
            # active set indices and indices of maximum correlations must coincide
            if len( np.setxor1d(active_set_indices, max_corr_inds[0].tolist()[0] ) ) != 0:
                raise AssertionError("Lars. Something wrong with the indices of active set.")
            # Compute current correlations (without new variables) and do some consistency checks <-             
            
            # For iteration we need to have: new_sign, new_active_ind, mu
                                                         
            # Compute new active dataset Xa, norm. constant Aa, and vector Ua, 
            # having index of next variable to include and sign of this variable ->                                    
            
            if not locals().get('inv_upd'):                
                inv_upd = Inverse_Update()
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
            
                if gamma_minus_min < gamma_plus_min:
                    new_sign = 1 # new sign for the next iteration
                    gamma_vals =  minus_part
                    gamma_min = gamma_minus_min
                    
                elif gamma_minus_min > gamma_plus_min:      
                    new_sign = -1 # new sign for the next iteration
                    gamma_vals = plus_part
                    gamma_min = gamma_plus_min
                else:
                    raise AssertionError(" Lars. Strage gamma_plus_min can't be equal gamma_minus_min")
                    
                gamma_ind = np.nonzero(gamma_vals == gamma_min)[0]
                if gamma_ind.size > 1: # Check that minimum is reached at only one point
                    raise AssertionError("Lars. Several gammas have the same value. Badly conditioned matrix?")
                                    
                del gamma_minus_min, gamma_plus_min, minus_part, plus_part
        
                mu = mu + gamma_min * Ua # update mu on this iteration
                new_active_ind = not_active_inds[gamma_ind][0,0] # new_active_ind for next iteration            
                # Compute step gamma for mu update and simultaneously new_sign and new_active_ind for next iteration <-
                
            else:
                mu = mu + np.abs( C[new_active_ind][0,0] ) / Aa * Ua # last value of mu equal y

            # Compute model coefficients beta iteratively ->
            XaTmu = X[:, active_set_indices ].T * mu 
            
            beta = XaTXa * XaTmu
            # Compute model coefficients beta iteratively <-
            
        return orig_X, active_set_signs, active_set_indices,  beta, mu, y 

        
if __name__ == "__main__":
    import sklearn
    import sklearn.datasets as ds
    import bottleneck as bn    
    
    BostonHousing = ds.load_boston()
        
    X0 = BostonHousing.get('data')
    y0 = BostonHousing.get('target')
    
    if bn.anynan(X0) or bn.anynan(y0):
        raise AssertionError('NaNs are not allowed')        
        
    X1 = X0.copy(); y1 = y0.copy()
    
    lars = Lars(-X1,y1)
    (X_1, asgn_1, ai_1, beta_1, mu_1, y_1) = lars.run(max_iter_no=5)

            