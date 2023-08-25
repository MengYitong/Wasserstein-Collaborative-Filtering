
import numpy.matlib as nmat
import tcutil as ut
import torch as tc

def wasserstein_NMF_dictionary_step(X,pX,K,lambdA,gamma,rho,H,options):

    cObj = lambda H: computeObj(X,H,gamma,pX,K,lambdA,rho)
    H, obj, gradient = ut.accelerated_gradient(H, cObj, options['dual_descent_stop'], options['t0'], options['alpha'], options['beta'], options['verbose'])
    # obj = -obj
    obj = [x * (-1) for x in obj]

    probtmp = -H@tc.transpose(lambdA,0,1)/rho
    maxtmp = tc.max(probtmp, 0)[0] # a row vector (max in each column)
    expL = tc.exp(probtmp-maxtmp.repeat(probtmp.shape[0], 1))
    sumL = expL.sum(0)
    D=expL/sumL

    return D, H, obj, 0

def computeObj(X,H,gamma,pX,K,lambdA,rho):
    obj, grad = ut.computeWassersteinLegendre(X,H,gamma,pX,K)
    objE, gradE = entropyObj2(H,lambdA,rho)
    # if tc.isinf(grad).any():
    #     print('tcoptimizeD computeObj inf in grad!')
    # if tc.isinf(gradE).any():
    #     print('tcoptimizeD computeObj inf in gradE!')
    # if tc.isnan(grad).any():
    #     print('tcoptimizeD computeObj inf in grad!')
    # if tc.isnan(gradE).any():
    #     print('tcoptimizeD computeObj inf in gradE!')
    obj = obj + objE
    grad = grad + gradE

    gradNorm = ut.fro_norm(grad)
    grad = grad / gradNorm
    return obj, grad, gradNorm

def entropyObj2(H,lambdA,rho):
    probtmp = -H@tc.transpose(lambdA,0,1)/rho
    expL=tc.exp(probtmp)

    if tc.isinf(expL).any():
        expL[tc.isinf(expL)] = tc.max(expL[~tc.isinf(expL)])
        print('inf in expL!=')

    sumL = expL.sum(0)
    obj = rho * tc.sum(tc.log(sumL));del expL;del sumL
    if tc.isinf(obj):
        obj=0
    # del sumL
    # maxtmp = max(probtmp) # a row vector(max in each column)
    maxtmp=tc.max(probtmp, 0)[0]
    # expL = exp(probtmp - repmat(maxtmp, [size(probtmp, 1), 1]))
    expL = tc.exp(probtmp - maxtmp.repeat(probtmp.shape[0], 1))
    sumL = expL.sum(0)
    grad = -expL @(lambdA/sumL.unsqueeze_(0).transpose_(0,1)) # divide each column in lambda by the vector sumL
    return obj, grad

def wasserstein_DL_dictionary_step(X,pX,K,lambdA,gamma,H,options):
    pinvLambda = tc.pinverse(lambdA)@ lambdA
    proj=lambda H:projection(H,pinvLambda)
    cObj = lambda H: ut.computeObj_DL(X,H,gamma,pX,K)
    H, _ = proj(H)
    H, obj, gradient = ut.linear_projected_gradient_descent(H, cObj, proj, options['dual_descent_stop'], options['t0'], options['alpha'], options['beta'], options['verbose'])
    obj = [x * (-1) for x in obj]

    # D = gradient / lambda
    D= gradient @  tc.pinverse(lambdA)
    return D, H, obj, gradient
def projection(gradient,pinvLambda):
    gradNorm = ut.fro_norm(gradient)
    if gradNorm > 0:
        projected = (gradient - (gradient @ pinvLambda) ) / gradNorm
        gradNorm = tc.sqrt(tc.sum(projected * gradient))
    else:
        projected = gradient

    return projected, gradNorm

