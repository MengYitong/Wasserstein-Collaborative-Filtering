import numpy as np
import torch as tc
import time
import gc
# computing H_p^*(g) and the first order gradient.
# inputs: H is g, K is tc.exp(-M / gamma), pX is E(p), X is p.
def computeWassersteinLegendre(X,H,gamma,pX,K): #
    #the input should all be pytorch tensor
    # t = time.time()
    alphaTmp=tc.exp(H/gamma)
    alphaTmp_inf_idx=tc.isinf(alphaTmp)
    if alphaTmp_inf_idx.any():
        alphaTmp[alphaTmp_inf_idx] = tc.max(alphaTmp[~alphaTmp_inf_idx])
        # print('inf in alphaTmp!')
    del alphaTmp_inf_idx

    if tc.isnan(alphaTmp).any():
        print('nan in alphaTmp!')

    grad=K@alphaTmp

    grad_zero_idx=(grad == 0)
    if grad_zero_idx.any():
        grad[grad_zero_idx] = tc.min(grad[~grad_zero_idx])
        print('0 in grad!')
    del grad_zero_idx

    grad_inf_idx=tc.isinf(grad)
    if grad_inf_idx.any():
        grad[grad_inf_idx] = tc.max(grad[~grad_inf_idx])
        # print('inf in grad==!, max:',tc.max(grad[~grad_inf_idx]))
        print('inf in grad==!')
    del grad_inf_idx

    obj=(tc.sum(X*tc.log(grad))+pX)*gamma
    grad_=alphaTmp*(K.transpose(0,1)@(X/grad))
    if tc.isnan(grad_).any():
        print('computeWassersteinLegendre: nan in grad!')
    #     temp1=alphaTmp
    #     if tc.isnan(temp1).any():
    #         print('computeWassersteinLegendre: nan in temp1!')
    #     temp2=(K@(X/grad))
    #     if tc.isnan(temp2).any():
    #         print('computeWassersteinLegendre: nan in temp2!')
    #
    #     temp3 = alphaTmp * (K @ (X / grad))
    #     if tc.isnan(temp3).any():
    #         print('computeWassersteinLegendre: nan in temp3!')
    # print('computeWassersteinLegendre costs time:',time.time() - t)
    return obj, grad_

#input is np.array(), output is also np.array(), partially pytorch in the process
def computeWassersteinLegendre2(X,H,gamma,pX,K,Tensor):
    t = time.time()
    gpu_H=H.cuda()
    gpu_gamma=gamma.cuda()
    gpu_alphaTmp=tc.exp(gpu_H/gpu_gamma)
    gpu_K=K.cuda()
    gpu_grad=gpu_K@gpu_alphaTmp
    print(time.time()-t)
    # t = time.time()
    grad_zero_idx=(grad == 0)
    if grad_zero_idx.any():
        grad[grad_zero_idx] = tc.min(grad[~grad_zero_idx])
        print('0 in grad!')
    del grad_zero_idx

    if tc.isinf(grad).any():
        grad[tc.isinf(grad)] = tc.max(grad[~tc.isinf(grad)])
        print('inf in grad!')
    # print(time.time() - t)
#    print(tc.sum(X*tc.log(grad)).dtype)
 #   print(pX.dtype)
 #    tc.cuda.empty_cache()

    obj=(tc.sum(X*tc.log(grad))+pX)*gamma


    grad=alphaTmp*(K@(X/grad))
    return obj, grad

def accelerated_gradient(H0,computeObjGrad,stop,t0,alpha,beta,verbose):
    sumFunction=lambda x,t,y:x+t*y
    H = H0
    objective, gradient, gradNorm = computeObjGrad(H)
    obj = [objective]
    checkObj = objective
    niter = 0
    t = t0
    Glast = H
    Hlast = H
    if verbose:
        print('\tDual iteration : %d Objective : %f, Current step size %.4E' % (niter, obj[-1], t))
        print ('\t\tGradnorm=%f stop=%f'%(gradNorm, stop))
    tol = stop * (1 + fro_norm(H))
    flag=0
    while gradNorm > tol and flag<20 and niter<1500:
        niter = niter + 1
        last = obj[-1]
        # prevt = t;
        t, objective, H, _, _ = backtrack( computeObjGrad, last, H, gradient, -alpha * gradNorm, beta, t, sumFunction);
        # numberEval = numberEval + log(t / prevt) / log(beta) + 1;
        G = H
        H = sumFunction(G, (niter - 2.) / (niter + 1.), sumFunction(G, -1., Glast))
        objective, gradient, gradNorm = computeObjGrad(H)
        obj.append(objective)
        Glast = G
        # numberEval = numberEval + 1
        tol = stop * (1 + fro_norm(H))
        if (niter - 1)%20 == 0:
            if verbose:
                print('\tDual iteration : %d Objective : %.8E, Current step size %.4E' % (niter, obj[-1], t))
                print('\t\tGradnorm=%.8E tol=%.8E flag=%d' % (gradNorm, tol,flag))
            Hlast = H

        if checkObj < obj[-1]:
            niter = 0
            H = Hlast
            objective, gradient, gradNorm = computeObjGrad(H)
            obj.append(objective)
        if checkObj <= obj[-1]:
            flag=flag+1

        checkObj = obj[-1]
        t = tc.min(t / tc.sqrt(beta), t0)
    if verbose:
        print('\tDual iteration : %d Objective : %f, Current step size %.4E'% (niter, obj[-1], t))
        print('\t\tGradnorm=%f tol=%f'% (gradNorm, tol))

    return H, obj,gradient

def backtrack(phi,f,U,dir,alpha,beta,t,sumFunction):
   # print('U.dtype:',U)
   # print('t.dtype:',t)
   #  print('dir.dtype',(t*dir).is_cuda)
    H = sumFunction(U, -t, dir)
    # print(type(U.data))
    # temp=tc.norm(H-U, 'fro')
    # print (temp)
    # if temp==0:
    #     print('temp is 0!')
    obj, grad, gradNorm = phi(H)
    objs=[]
    objs.append(obj)
    test = obj + gradNorm
    while  tc.isnan(test).any() or tc.isinf(test).any() or obj > f + alpha * t:
        # print('backtrak')
        t = beta * t
        H = sumFunction(U, -t, dir)
        obj, grad, gradNorm = phi(H)
        objs.append(obj)
        test = obj + gradNorm
    return t, obj, H, grad, gradNorm

def matrixEntropy(X):
#    print('X.dtype:',X.dtype)
    x=X[X!=0]
    pX = x * tc.log(x)
    pX = -tc.sum(pX)
    return pX
def fro_norm(M):
    return tc.sqrt(tc.sum(M*M))

def initialValue(X,sizeD,gpu,Tensor):

    # HD = tc.zeros(X.shape,dtype=np.float64)
    # Hlambda = tc.zeros(X.shape,dtype=tc.float64)
    sizeH=(sizeD[0],X.shape[1])
    HD = Tensor(np.zeros(sizeH))
    Hlambda = Tensor(np.zeros(sizeH))

    D = np.random.uniform(0, 100, sizeD)
    # D = np.random.normal(1, 1, sizeD)
    # D=np.load('./data/mfD.npy')
    # D = X[:,0:k]
    sumD = D.sum(0)
    D = D/sumD
    D = np.nan_to_num(D)
    # D=np.load('initD.npy')
    D = Tensor(D)
    if gpu:
        D = D.cuda()
        HD = HD.cuda()
        Hlambda= Hlambda.cuda()

    return D, HD, Hlambda
def memReport():
    for obj in gc.get_objects():
        if tc.is_tensor(obj):
            print(type(obj), obj.size())

def computeObj_DL(X,H,gamma,pX,K):
    obj, grad=computeWassersteinLegendre(X,H,gamma,pX,K)
    gradNorm=fro_norm(grad)
    return obj, grad, gradNorm


def linear_projected_gradient_descent(H, computeObj, projection, stop, t0, alpha, beta,verbose):
    objective, gradient, _= computeObj(H)
    obj = [objective]
    last = np.inf
    niter = 0
    t = t0
    if verbose:
        print('    Dual initialization, Objective : ', obj[-1], ' Current step size ', t)
    while (abs(obj[-1]-last)>stop*(1+abs(obj[-1]))*t or abs(obj[-1]-last)>1) and niter < 30002:
        last=obj[-1]
        projected, gradNorm = projection(gradient)
        t, objective, H, gradient, _ = backtrack( computeObj, last, H, projected, -alpha * gradNorm, beta, t,   lambda a, coeff, b:a + coeff * b)
        obj.append(objective)
        if verbose and  niter%5000 == 0:
            print('    Dual iteration : ', (niter + 1), ' Objective : ', obj[-1], ' Current step size ',t)

        t = t / tc.sqrt(beta)
        niter = niter + 1

    if verbose:
        print('    Finished, number of iterations : ', (niter), ' Objective : ',  obj[-1],' Current step size ', t)
    return  H, obj, gradient

