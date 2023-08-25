import numpy as np
import bottleneck as bn
from scipy.sparse import csr_matrix, lil_matrix, save_npz, load_npz,find
def Recall_at_k_batch(X_pred, heldout_batch, k=20): #take user-item matrix as input
    m, n = X_pred.shape
    # print(k)
    if k>n:
        k=n-1
    # heldout_batch=heldout_batch.A
    batch_users = X_pred.shape[0]
    # print X_pred.shape
    # print heldout_batch.shape
    axi=1
    idx = bn.argpartition(-X_pred, k, axis=axi)
    # print idx[:, :k]
    # print X_pred[:,idx[:, :k]]
    # print np.unique(heldout_batch[:,idx[:, :k]],axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    # print X_pred_binary
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True
    # print X_pred_binary
    X_true_binary = (heldout_batch > 0)
    # print X_true_binary
    # print X_true_binary[1,:]
    # print X_pred_binary[1,:]
    tempmatrix=np.logical_and(X_true_binary, X_pred_binary)
    tmp = (tempmatrix.sum(axis=axi)).astype(np.float32)
    base=np.minimum(k, X_true_binary.sum(axis=axi))
    # print base
    recall = tmp / base
    return np.mean(recall)

def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100): #X_pred and heldout_batch should be user-item matrix and of the same size.
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    m,n=X_pred.shape

    heldout_batch=csr_matrix(heldout_batch)
    heldout_batch.data=np.ones_like(heldout_batch.data)
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)

    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]

    idx_part = np.argsort(-topk_part, axis=1)

    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))
    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].A * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    # print('DCG:',DCG,'IDCG:',IDCG)
    return np.mean(DCG / IDCG)
def NDCG_binary_at_k_batch2(X_pred, heldout_batch, k=100): #X_pred and heldout_batch should be user-item matrix and of the same size.
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    m,n=X_pred.shape
    # print(k)
    if k>n:
        k=n-1
    # print('k:',k)
    heldout_batch=csr_matrix(heldout_batch)
    heldout_batch.data=np.ones_like(heldout_batch.data)
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    # print('idx_topk_part:',idx_topk_part)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k+1]]
    # print('topk_part:',topk_part)
    idx_part = np.argsort(-topk_part, axis=1)
    # print('idx_part:',idx_part)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))
    tp=np.concatenate(([1.],tp),axis=None)
    # print(tp)
    # print('idx_topk:',idx_topk)
    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].A * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    # print('DCG:',DCG,'IDCG:',IDCG)
    return np.mean(DCG / IDCG)

def AP(pred,actual):# actual and pred should be one dimensional row vector
    idx = np.argsort(-pred)
    c = actual[idx]
    nz_idx = np.nonzero(c)
    nz_idx = nz_idx[0]
    tmp = np.array(np.arange(1, len(nz_idx) + 1),dtype=np.float32)
    base = (nz_idx + 1.0)
    return np.sum(tmp/base)/(len(nz_idx)+.0)

def MAP(Pred,Actual):#input should be user-item matrix
    return np.mean([AP(p,a) for p,a in zip(Pred,Actual)])

def eval(train,pred,test): #the 3 matrix should be item-user matrix of the same size
    train=train.transpose()#convert to user-item matrix
    pred=pred.transpose()
    test=test.transpose()
    train_user_num,train_item_num=train.shape
    test_user_num, test_item_num = test.shape
    pred_user_num, pred_item_num = pred.shape
    if train_user_num!=test_user_num or pred_user_num!=test_user_num or train_user_num!=pred_user_num:
        print('user_num inconsistant!')
        exit()
    if test_item_num!=pred_item_num:
        print('item num inconsistant!')
        exit()
    train_nnz_idx = train.nonzero()
    if train is None:
        print('train is None!')
    elif train.shape==test.shape:
        print('train.shape=test.shape')
        pred[train_nnz_idx] = -np.inf
        test[train_nnz_idx] = 0
    else:
        pass
        # print('train.shape!=test.shape')
    r, c = test.nonzero()
    idx = list(set(r))
    recall = Recall_at_k_batch(pred[idx, :], test[idx, :], 20)
    ndcg = NDCG_binary_at_k_batch((pred[idx, :]), (test[idx, :]), 20)
    map=MAP(pred[idx, :], test[idx, :])
    return map,ndcg,recall#np.mean(recall_list)


def eval2(pred,test,k=20): #the 2 matrix should be user-item dense matrix
    test_user_num, test_item_num = test.shape
    pred_user_num, pred_item_num = pred.shape
    if  pred_user_num!=test_user_num :
        print('user_num inconsistant!')
        exit()
    if test_item_num!=pred_item_num:
        print('item num inconsistant!')
        exit()
    r, c = test.nonzero()
    idx = list(set(r))
    recall = Recall_at_k_batch(pred[idx, :], test[idx, :], k)
    ndcg = NDCG_binary_at_k_batch2((pred[idx, :]), (test[idx, :]), k)
    map=MAP(pred[idx, :], test[idx, :])
    return map,ndcg,recall#np.mean(recall_list)

if __name__== "__main__":
    pred=np.array([[1,2,3,0,0]])
    test=np.array([[0,0,2,3,4]])
    print(NDCG_binary_at_k_batch(pred, test, k=3))