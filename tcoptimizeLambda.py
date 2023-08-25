import time
import numpy.matlib as nmat
import tcutil as ut
import torch as tc

def wasserstein_NMF_coefficient_step(X,pX, K,D,gamma,rho,H,options):
    # if ~exist('H', 'var') | | isempty(H):
    #     H = options.createZeroArray([n, m]);

    cObj = lambda H: computeObj(X,H,gamma,pX,K,D,rho)
    H, obj, gradient = ut.accelerated_gradient(H, cObj, options['dual_descent_stop'], options['t0'], options['alpha'], options['beta'], options['verbose'])
    # obj = -obj
    obj = [x * (-1) for x in obj]

    #Recover  lambda from the dual variable
    probtmp = -tc.transpose(D,0,1)@H/rho
    maxtmp = tc.max(probtmp, 0)[0]
    expD = tc.exp(probtmp - maxtmp.repeat(probtmp.shape[0], 1))
    sumE = expD.sum(0)
    lambdA = expD/sumE

    return lambdA, H, obj, 0

def computeObj(X,H,gamma,pX,K,D,rho):
    obj, grad = ut.computeWassersteinLegendre(X,H,gamma,pX,K)
    t = time.time()
    objE, gradE = entropyObj2(H,D,rho)
    # print('entropyObj2 costs time:', time.time() - t)
    # print('obj:',obj,'objE',objE)
    obj = obj + objE
    grad = grad + gradE

    gradNorm = ut.fro_norm(grad)
    grad = grad / gradNorm
    return obj, grad, gradNorm

def entropyObj2(H,D,rho):
    # probtmp = -D'*H/rho

    probtmp = -tc.transpose(D,0,1)@H/rho
    # probtmp = -D.transpose(0,1)@H
    expD = tc.exp(probtmp)
    # expD = probtmp

    expD_inf_idx=tc.isinf(expD)
    # # expD_inf_idx = tc.isinf(D)
    # # print(expD_inf_idx.get_device())
    # t = time.time()
    # flag=expD_inf_idx.any()
    # print('1 =time:', time.time() - t)
    # print(flag)
    # t = time.time()
    if expD_inf_idx.any():
        expD[expD_inf_idx] = tc.max(expD[~expD_inf_idx])
        print('inf in expD!')
    # print('2 time:', time.time() - t)

    # ut.memReport()

    #sum each column

    sumE = expD.sum(0)
    obj = rho * tc.sum(tc.log(sumE));del expD;del sumE;
    if tc.isinf(obj):
        obj=0

    # print('obj in entropy:',obj)
    # a = tc.cuda.DoubleTensor([3.] * 100)
    # print(0.1 * tc.sum(tc.log(a)))
    # clear expD sumE;

    # maxtmp = max(probtmp); (max of each column)
    maxtmp=tc.max(probtmp, 0)[0]
    expD = tc.exp(probtmp - maxtmp.repeat(probtmp.shape[0], 1))
    sumE = expD.sum(0)

    grad = -(D@expD)/sumE

    return obj,grad

def entropyObj3(H,D,rho):
    # probtmp = -D'*H/rho

    probtmp = tc.transpose(D,0,1)@H
    # probtmp = -D.transpose(0,1)@H
    # # expD = tc.exp(probtmp)
    # expD = probtmp

    # expD_inf_idx=tc.isinf(expD)
    expD_inf_idx = tc.isinf(H)
    print(expD_inf_idx.get_device())
    t = time.time()
    flag=expD_inf_idx.any()
    print('1 ==time:', time.time() - t)
    print(flag)
    t = time.time()
    if expD_inf_idx.any():
        expD[expD_inf_idx] = tc.max(expD[~expD_inf_idx])
        print('inf in expD!')
    print('2 time:', time.time() - t)

    # ut.memReport()

    #sum each column

    sumE = expD.sum(0)


    obj = rho * tc.sum(tc.log(sumE))

    # print('obj in entropy:',obj)
    # a = tc.cuda.DoubleTensor([3.] * 100)
    # print(0.1 * tc.sum(tc.log(a)))
    # clear expD sumE;

    # maxtmp = max(probtmp); (max of each column)
    maxtmp=tc.max(probtmp, 0)[0]
    expD = tc.exp(probtmp - maxtmp.repeat(probtmp.shape[0], 1))
    sumE = expD.sum(0)

    grad = -(D@expD)/sumE

    return obj,grad

def wasserstein_DL_coefficient_step(X,pX, K,D,gamma,H,options):
    pinvD = D @ tc.pinverse(D)
    proj=lambda H:projection(H,pinvD)
    cObj = lambda H: ut.computeObj_DL(X,H,gamma,pX,K)
    H,_=proj(H)
    H, obj, gradient = ut.linear_projected_gradient_descent(H, cObj, proj, options['dual_descent_stop'], options['t0'], options['alpha'], options['beta'], options['verbose'])
    obj = [x * (-1) for x in obj]

    pD = tc.pinverse(D)
    lambdA = pD @ gradient
    return lambdA, H, obj, gradient

def projection(gradient,pinvD):
    gradNorm = ut.fro_norm(gradient)
    if gradNorm > 0:
        projected = (gradient - pinvD @ gradient) / gradNorm
        gradNorm = tc.sqrt(tc.sum(projected* gradient))
    else:
        projected = gradient
    return projected, gradNorm

# def computeObj_DL(X,H,gamma,pX,K):
#     obj, grad=ut.computeWassersteinLegendre(X,H,gamma,pX,K)
#     gradNorm=ut.fro_norm(grad)
#     return obj, grad, gradNorm