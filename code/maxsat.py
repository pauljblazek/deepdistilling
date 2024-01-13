import numpy as np
import math
import random
import os
from copy import copy
import scipy.sparse as sparse
from multiprocessing import Pool
#from keras.models import Sequential
from functools import partial
import time

def get_all_maxsat(x_test, greedy_maxsat, enn=None, sgd=None, deliberate=False, verbose=True):
    """Compute each method's attempt at finding the MAXSAT value for all formulae"""

    if isinstance(enn, Network):
        enn.layers[0].weights = sparse.csc_matrix(enn.layers[0].weights)

    maxsat = np.zeros((x_test.shape[0], 5))
    maxsat_mean = np.zeros((x_test.shape[0], 5))
    maxsat_max = np.zeros((x_test.shape[0], 5))
    
    #Get results for each formula and print
    if verbose:
        print('GDN ENN Greedy Random Approximation')
    if 'SLURM_JOB_ID' in os.environ and x_test.shape[1]>(10**2)*2*10:
        #These are large so we will run them in parallel
        num_at_a_time = 500
        for start_ind in range(0, x_test.shape[0], num_at_a_time):
            pool = Pool()
            sample_x = x_test[start_ind:(start_ind+num_at_a_time), :]
            results = pool.map(partial(get_one_maxsat, nets=[sgd, enn, None, None, None], x_test=sample_x, verbose=verbose), range(sample_x.shape[0]))
            pool.close()
            pool.join()
            for r,result in enumerate(results):
                for j in range(5):
                    maxsat[start_ind + r , j] = result[0][j]
                    maxsat_mean[start_ind + r , j] = result[1][j]
                    maxsat_max[start_ind + r , j] = result[2][j]
    else:
        for i in range(x_test.shape[0]):
            result, result_mean, result_max = get_one_maxsat(i, [sgd, enn, None, None, None], x_test, verbose=verbose)
            print('a',end=' ')
            for j in range(5):
                maxsat[i, j] = result[j]
                maxsat_mean[i, j] = result_mean[j]
                maxsat_max[i, j] = result_max[j]
                print(round(np.mean(maxsat[:i+1,j]), 1), end=' ')
            print()
    
    #PRINT SUMMARY STATISTICS
    print()
    print('##################################################')
    print()
    print('GDN avg:', np.mean(maxsat[:, 0]))
    print('ENN avg:', np.mean(maxsat[:, 1]))
    print('Approx avg:', np.mean(maxsat[:, 2]))
    print('Rand avg:', np.mean(maxsat[:, 3]))
    print('Greedy avg:', np.mean(maxsat[:, 4]))
    print('GDN avg E:', np.mean(maxsat_mean[:, 0]))
    print('ENN avg E:', np.mean(maxsat_mean[:, 1]))
    print('Approx avg E:', np.mean(maxsat_mean[:, 2]))
    print('Rand avg E:', np.mean(maxsat_mean[:, 3]))
    print('Greedy avg E:', np.mean(maxsat_mean[:, 4]))
    print('GDN avg max:', np.mean(maxsat_max[:, 0]))
    print('ENN avg max:', np.mean(maxsat_max[:, 1]))
    print('Approx avg max:', np.mean(maxsat_max[:, 2]))
    print('Rand avg max:', np.mean(maxsat_max[:, 3]))
    print('Greedy avg max:', np.mean(maxsat_max[:, 4]))
    print()
    print('##################################################')
    print()

    enn.layers[0].weights = enn.layers[0].weights.toarray()
    return

def get_one_maxsat(test_i, nets, x_test, deliberate=False, verbose=True):
    clauses = x_test[test_i].reshape(1,-1)
    num_clause = int(clauses[0,0])
    clauses = clauses[:,1:]
    
    if np.any(clauses>1):
        #If this is the case then the clauses are stored sparsely and need to be extracted
        clauses = np.reshape(clauses[0,:3*int(round((clauses[0,0]+1)))], (3,-1)).transpose()
        num_clause = int(round(clauses[0,1]))
        num_var = int(round(clauses[0,2])/2)
        new_clauses = np.zeros((num_clause, num_var*2))
        for i in range(int(round(clauses[0, 0]))):
            val = clauses[i+1, 0]
            row = int(round(clauses[i+1, 1]-1))
            col = int(round(clauses[i+1, 2]-1))
            new_clauses[row, col] = val
        num_nonempty_clauses = np.sum(np.any(new_clauses!=0, axis=1))
        clauses = np.reshape(new_clauses.transpose().flatten(), (1,-1))
    elif num_clause==None:
        num_var = int(np.sqrt(clauses.shape[1]/2/5))
        num_clause = int(clauses.shape[1]/num_var/2)
    
    maxsat = np.zeros(len(nets))
    maxsat_mean = np.zeros(len(nets))
    maxsat_max = np.zeros(len(nets))
    
    #Because the output is probabilistic, we will perform it multiple times
    num_trials = 10
    if 'SLURM_JOB_ID' in os.environ:
        num_trials = 100
    for net_i,net in enumerate(nets):
        if (nets[1] is None) and net_i != 0:
            continue
        for trial in range(num_trials):
            temp_clauses = clauses.copy()
            #Store the clauses as a matrix to see progress
            clauses_mat = np.reshape(temp_clauses, (2*num_var, num_clause)).transpose()
            if net_i==2:
                clauses_mat = np.delete(clauses_mat, np.where(np.all(clauses_mat==0, axis=1))[0], axis=0)
            num_sat = 0

            #For each variable we will iteratively assign a TRUE or FALSE value
            for var in range(num_var):
                if net_i<2: #The GDN and the ENN here
                    new_x = get_assignment(net, temp_clauses, is_sgd=(net_i==0), deliberate=deliberate)
                elif net_i==2: #The 3/4-approximation algorithm
                    new_x = get_assignment(net, clauses_mat, greedy=True, var=var, num_sat=num_sat)
                elif net_i==3: #Random guess
                    if np.random.random()<.5:
                        new_x = 1
                    else:
                        new_x = -1
                elif net_i==4: #The pure greedy algorithm
                    t = np.sum(clauses_mat[:,0])
                    f = np.sum(clauses_mat[:,1])
                    if t>f:
                        new_x = 1
                    elif t<f:
                        new_x = -1
                    elif np.random.random() < .5:
                        new_x = 1
                    else:
                        new_x = -1
                #Based on the choice of assignment, update formula by removing it and zeroing satisfied clauses
                for c in range(clauses_mat.shape[0]):
                    if new_x==-1:
                        if clauses_mat[c,1] == 1:
                            clauses_mat[c,:] = 0
                            num_sat += 1
                    elif clauses_mat[c, 0] == 1:
                        clauses_mat[c, :] = 0
                        num_sat += 1
                clauses_mat = np.hstack((clauses_mat[:, 2:], np.zeros((clauses_mat.shape[0],2))))
                temp_clauses = clauses_mat.transpose().flatten()
            #Get statistics
            maxsat[net_i] = num_sat
            maxsat_mean[net_i] += num_sat/num_trials
            if num_sat>maxsat_max[net_i]:
                maxsat_max[net_i] = num_sat
    
    #Print results for each formula for each net
    if verbose:
        print_string = str(num_nonempty_clauses) + ' '
        for j in range(len(maxsat)):
            print_string += str(round(maxsat_mean[j], 2)) + ' '
        print(print_string)

    return maxsat, maxsat_mean, maxsat_max

def get_assignment(net, clauses, is_sgd=True, deliberate=False, greedy=False, var=0, num_sat=0, valid_ind=None, unsat=0, is_sparse=False):
    if valid_ind is None:
        valid_ind = np.arange(len(clauses))
    if greedy or net==1:
        #The 3/4-approximation algorithm
        if var==0:
            cur_B = .5*clauses.shape[0]
            cur_SAT = num_sat
            cur_UNSAT = clauses.shape[0]
        else:
            cur_SAT = num_sat
            cur_UNSAT = unsat
            cur_B = .5*(cur_SAT + clauses.shape[0] - cur_UNSAT)
        ft = []
        temp_UNSAT = [None, None]
        for s in range(2):
            if is_sparse:
                ind = clauses[valid_ind,2*var+1-s].indices
            else:
                ind = np.where(clauses[valid_ind,2*var+1-s])[0]
            temp_SAT = num_sat + len(ind)
            off_ind = np.setdiff1d(np.arange(len(valid_ind)), ind)
            if is_sparse:
                rows = np.unique(clauses[valid_ind[off_ind],2*var+2:].indices)
                temp_UNSAT[s] = len(off_ind)-len(rows)
            else:
                temp_UNSAT[s] = len(off_ind)-np.sum(np.any(clauses[valid_ind[off_ind],2*var+2:], axis=1))
            temp_B = .5*(temp_SAT + clauses.shape[0] - temp_UNSAT[s])
            ft.append(temp_B-cur_B)
        f = ft[0]
        t = ft[1]
        if f<0:
            output = [1, 0]
        elif t<0:
            output = [0, 1]
        elif t==0 and f==0:
            output = [.5, .5]
        else:
            output = [t/(t+f), f/(t+f)]
        assignment = (np.random.random() < output[0])
        return assignment*2-1, temp_UNSAT[assignment]
    elif net is None: #Random guess
        output = np.array([.5, .5])
    elif is_sgd: #Gradient descent-trained networks
        if len(clauses.shape)==1:
            clauses = clauses.reshape((1,-1))
        if isinstance(net, Network):
            output = net.compute_output(clauses, deliberate=deliberate, last_activation=False)[0]
            output = np.array([output[0], output[1]])
            output = np.exp(output)
        else:
            output = net.predict(clauses)[0]
    else: #ENN
        output = net.compute_output(clauses, deliberate=deliberate, last_activation=False)[0]
        output = np.array([output[0], output[1]])
        output = np.exp(output)
    #Normalize to get probabilities
    if np.sum(output)>0:
        output /= np.sum(output)
    else:
        output = np.array([.5, .5])
    #Return as +1 or -1
    return (np.random.random() < output[0])*2-1

def ENN_maxsat(I, n, m, ind, v, is_sparse=False):
    #I is an input of size mxn (m one-hot-encoded Boolean variables, n clauses)

    times = np.zeros(11)
    
    """
	D1 = np.zeros(n)
	for i in range(n):
		if np.any(I[2:, i]!=0):
			D1[i] = -1
		else:
			D1[i] = 1
    """
    t0 = time.time()
    I1 = I[2*v+2:, ind]
    if not is_sparse:
        D1 = 1-np.any(I1, axis=0).astype(int)
    else:
        D1 = np.ones(I1.shape[1])
        D1[np.unique(sparse.csr_matrix(I1).indices)] = 0
    times[0] = time.time()-t0

    """
    D2 = np.zeros(n)
	for i in range(n):
		if np.any(I[2:, i]!=0):
			D2[i] = 1
		else:
			D2[i] = -1
    """
    t0 = time.time()
    D2 = 1-D1
    times[1] = time.time()-t0

    """
	D3 = np.zeros(n)
	for i in range(n):
		col_mean = np.mean(I[2:, i])
		if I[1, i] + col_mean - I[0, i] > 0:
			D3[i] = 1
		else:
			D3[i] = -1
    """
    t0 = time.time()
    if I1.size>0:
        mn = np.mean(I1, axis=0)
    else:
        mn = np.array(0)
    
    if not is_sparse:
        Ia0 = I[2*v+0,ind]
        Ia1 = I[2*v+1,ind]
    else:
        Ia0 = I[2*v+0,ind].toarray().flatten()
        Ia1 = I[2*v+1,ind].toarray().flatten()
        mn = mn.flatten()
    d_I = Ia1-Ia0
    D3 = (d_I + mn > 0)
    times[2] = time.time()-t0
		
    """
	D4 = np.zeros(n)
	for i in range(n):
		col_mean = np.mean(I[2:, i])
		if I[0, i] + col_mean - I[1, i] > 0:
			D4[i] = 1
		else:
			D4[i] = -1
    """
    t0 = time.time()
    D4 = (-d_I + mn > 0)
    times[3] = time.time()-t0
	
    t0 = time.time()
    """
    D5 = np.zeros(n)
    for i in range(n):
    	if (I[0, i] and (not I[1, 0])):
    		D5[i] = 1
    	elif (not I[1, 0]) or I[0, i]:
    		D5[i] = -1
    """
    D5 = np.sign(Ia0*2+I[2*v+1,ind[0]]-1)
    times[4] = time.time()-t0
	
    t0 = time.time()
    """
    D6 = np.zeros(n)
    for i in range(n):
        if (I[1, i] and (not I[0, 0])):
            D6[i] = 1
        elif (not I[0, 0]) or I[1, i]:
        	D6[i] = -1
    """
    D6 = np.sign(Ia1*2+I[2*v+0,ind[0]]-1)
    times[5] = time.time()-t0

    """
    S1 = np.zeros(n)
    for i in range(n):
    	S1[i] = (D6[i]>0 and D1[i]>0)
    """
    t0 = time.time()
    S1 = (D6.astype(int) + D1.astype(int))==2
    times[6] = time.time()-t0
    """
    S2 = np.zeros(n)
    for i in range(n):
    	S2[i] = (D2[i]>0 and D3[i]>0)
    """
    t0 = time.time()
    S2 = (D2.astype(int) + D3.astype(int))==2
    times[7] = time.time()-t0
    """	
    S3 = np.zeros(n)
	for i in range(n):
		S3[i] = (D5[i]>0 and D1[i]>0)
    """
    t0 = time.time()
    S3 = (D5.astype(int) + D1.astype(int))==2
    times[8] = time.time()-t0
    """
	S4 = np.zeros(n)
	for i in range(n):
		S4[i] = (D2[i]>0 and D4[i]>0)
    """
    t0 = time.time()
    S4 = (D2.astype(int) + D4.astype(int))==2
    times[9] = time.time()-t0

    t0 = time.time()
    C1 = 10.0*np.sum(S3) + 2.298*np.sum(S4) - 2.298*np.sum(S2) - 10.0*np.sum(S1)
    C2 = 10.0*np.sum(S1) + 2.298*np.sum(S2) - 2.298*np.sum(S4) - 10.0*np.sum(S3)
    C = [C1, C2]
    times[10] = time.time()-t0

    return np.exp(C)/np.sum(np.exp(C)), times

if __name__ == '__main__':

    if True:
        for problem in reversed(range(2)):
            num_samples = 100
            for sample in range(num_samples):
                for variables in [10]:#, 25, 50, 100, 250, 500, 1000]:
                    num_clauses = np.unique(np.maximum(1,np.linspace(0,variables*10, 21)).astype(int))
                    for nc in num_clauses:        
                        formula = np.zeros((nc,variables*2))
                        for c in range(nc):
                            if problem==0:
                                num_var = 3
                            else:
                                num_var = 0
                                while num_var==0:
                                    num_var = np.random.binomial(variables, 3/variables)
                            indices = np.random.choice(np.arange(variables), num_var, replace=False)
                            indices = 2*indices + np.random.randint(0,2,num_var)
                            formula[c,indices] = 1
                        print(problem,variables,nc, end=' ')
                        all_times = np.zeros(14)
                        for alg in range(3):
                            num_trials = 20
                            total_sat = 0
                            for trial in range(num_trials):
                                valid_ind = np.arange(len(formula))
                                if (alg==2 and formula.size>500000) or (alg==0 and formula.size>1.5e6):
                                    temp_formula = sparse.csc_matrix(formula)
                                    is_sparse = True
                                else:
                                    temp_formula = formula.copy()
                                    is_sparse = False
                                num_sat = 0
                                unsat = 0
                                for v in range(variables):
                                    if alg==0 or alg==2:
                                        probs, times = ENN_maxsat(temp_formula.transpose(), temp_formula.shape[0], variables, valid_ind, v, is_sparse)
                                        if alg!=0:
                                            from general_maxsat import ENN_maxsat as enn
                                            probs = enn(temp_formula[valid_ind][:, 2*v:].transpose(), temp_formula[valid_ind][:, 2*v:].shape[0], variables)
                                        new_x = (np.random.random()<probs[0])*2-1
                                        all_times[:11] += times
                                    elif alg==1:
                                        t = np.sum(temp_formula[valid_ind,2*v+0])
                                        f = np.sum(temp_formula[valid_ind,2*v+1])
                                        if t>f:
                                            new_x = 1
                                        elif t<f:
                                            new_x = -1
                                        elif np.random.random() < .5:
                                            new_x = 1
                                        else:
                                            new_x = -1
                                    elif alg==2:
                                        t0 = time.time()
                                        new_x, unsat = get_assignment(None, temp_formula, greedy=True, var=v, num_sat=num_sat, valid_ind=valid_ind, unsat=unsat, is_sparse=is_sparse)
                                        all_times[11] += time.time()-t0
                                    
                                    t0 = time.time()
                                    if new_x == -1:
                                        if not is_sparse:
                                            ind = np.where(temp_formula[valid_ind,2*v+1]==0)[0]
                                        else:
                                            ind = np.setdiff1d(np.arange(len(valid_ind)), temp_formula[valid_ind,2*v+1].indices)
                                    else:
                                        if not is_sparse:
                                            ind = np.where(temp_formula[valid_ind,2*v+0]==0)[0]
                                        else:
                                            ind = np.setdiff1d(np.arange(len(valid_ind)), temp_formula[valid_ind,2*v+0].indices)
                                    num_sat += len(valid_ind)-len(ind)
                                    #temp_formula[ind,:] = 0
                                    if len(ind)==0:
                                        break
                                    valid_ind = valid_ind[ind]
                                    #temp_formula = temp_formula[ind,2:]
                                    all_times[12] += time.time()-t0
                                    t0 = time.time()
                                    #temp_formula = np.hstack((temp_formula[:, 2:], np.zeros((temp_formula.shape[0],2))))
                                    all_times[13] += time.time()-t0
                                total_sat += num_sat
                            mean_sat = total_sat/num_trials
                            print(round(mean_sat,1), end=' ')
                        print(np.round(all_times,1))
                                


    else:

        #Load pre-trained networks
        problem = 'SAT'
        jobid = '1841972' #10 variables
        #jobid = '1860095' #40 variables  
        #jobid = '2018741' #50 variables
        networkid = '0'

        enn_file = problem + '_enn_' + jobid + '_' + networkid + '.npz'
        gdn_file = problem + '_sgdnet_' + jobid + '__' + networkid + '.npz'
        enn = Network([])
        enn.load_network(enn_file)
        if os.path.isfile(gdn_file):
            sgd = Network([])
            sgd.load_sgdnet(gdn_file)
        else:
            sgd = None

        print('Using ', enn_file)
        enn.layers[0].symbolic = True
        enn.layers[1].symbolic = True
        num_var = int(np.sqrt(enn.layers[0].weights.shape[0]/10/2))
        
        #Test the networks on the two test sets
        x_test = np.loadtxt('SATrand_'+str(num_var)+'_test_samples.csv', delimiter=',')
        get_all_maxsat(x_test, None, enn, sgd, False)
        x_test = np.loadtxt('SAT3_'+str(num_var)+'_test_samples.csv', delimiter=',')
        get_all_maxsat(x_test, None, enn, sgd, False)