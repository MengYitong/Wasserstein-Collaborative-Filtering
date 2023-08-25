# function[D, lambda , objectives, HD, Hlambda] = wasserstein_DL(X, p, M, gamma, rhoL, rhoD, options, initialValues)
import tcutil as ut, tcoptimizeLambda as ol, tcoptimizeD as od
import numpy as np
import torch as tc
import evaluate as ev
from scipy.sparse import csr_matrix, lil_matrix, save_npz, load_npz,find
#train, test should be full numpy matrix
def wasserstein_DL(X,k, M, gamma, rhoL, rhoD, initD, initHD, initHlambda, options,Tensor):
    print('k:', k, '; gamma:', gamma.cpu().data.numpy(), '; rhoL:', rhoL.cpu().data.numpy(), '; rhoD:', rhoD.cpu().data.numpy(), '; lambda_step_stop:',
          options['lambda_step_stop'].cpu().data.numpy(), '; D_step_stop: ', options['D_step_stop'].cpu().data.numpy(), '; stop:', options['stop'].cpu().data.numpy())

    # [options] = checkOptions(options);

    verbose = options['verbose']
    if verbose:
        options['verbose'] = verbose - 1

    # D, HD, Hlambda = initialValue( X, p,options['gpu'])
    D, HD, Hlambda =initD, initHD, initHlambda
    biglast = Tensor([0])
    bigobj = Tensor([-1])
    if options['gpu']:
        biglast=biglast.cuda()
        bigobj= bigobj.cuda()
    objectives = []
    niter = 0

    K=tc.exp(-M / gamma)
    pX = ut.matrixEntropy(X)
    optionsD = options
    options['dual_descent_stop'] = options['lambda_step_stop']
    optionsD['dual_descent_stop'] = options['D_step_stop']

    if rhoL>0 or rhoD>0:
        optimizeLambda = lambda D,Hlambda: ol.wasserstein_NMF_coefficient_step(X, pX, K,D,gamma,rhoL,Hlambda,options)#(X, K, D, gamma, rhoL, Hlambda, options, 1)
        optimizeD = lambda lambdA, HD: od.wasserstein_NMF_dictionary_step(X,pX,K,lambdA,gamma,rhoD,HD,optionsD)#(X, K, lambda , gamma, rhoD, HD, optionsD, 1)
    else:
        optimizeLambda = lambda D, Hlambda: ol.wasserstein_DL_coefficient_step(X,pX, K,D,gamma,Hlambda,options)  # (X, K, D, gamma, rhoL, Hlambda, options, 1)
        optimizeD = lambda lambdA, HD: od.wasserstein_DL_dictionary_step(X,pX, K,lambdA,gamma,HD,options)

    recall = 0
    while tc.abs(bigobj - biglast) > options['stop'] * (1 + tc.abs(bigobj)) :#and niter<20:
    # while niter<20:
        niter = niter + 1
        print(niter)
        if verbose:
            print('Optimize with respect to lambda')

        # Optimize with respect to lambda, D is fixed

        lambdA, Hlambda, objL,_ = optimizeLambda(D, Hlambda)

        biglast = bigobj
        tmpObj = objL[-1]
        if rhoD > 0:
            tmpObj = tmpObj - rhoD * ut.matrixEntropy(D)

        objectives.append(tmpObj)

        if verbose:
            print('Optimize with respect to D')


        # % Optimize     with respect to D, lambda is fixed
        D, HD, objD,_ = optimizeD(lambdA , HD)

        tmpObj = objD[-1]
        if rhoL>0:
            tmpObj = tmpObj - rhoL * ut.matrixEntropy(lambdA)
        else:
            sumD = tc.sum(abs(D),0)
            sumD[sumD == 0] = 1
            D = D/ sumD

        objectives.append(tmpObj)

    return D, lambdA, objectives



