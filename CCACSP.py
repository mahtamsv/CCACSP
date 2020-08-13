import numpy as np 
import sklearn.discriminant_analysis as sklda
import scipy as sp 

def train(train_data_1, train_data_2, numFilt):

    numTrials_1 = np.size(train_data_1,0)
    numTrials_2 = np.size(train_data_1,0)

    # train the CCACSP filters 
    ccacsp_filts = calc_CCACSP(train_data_1, train_data_2, numFilt)

    # extract the features
    train_filt_1 = apply_CCACSP(train_data_1, ccacsp_filts, numFilt)
    train_logP_1  = np.squeeze(np.log(np.var(train_filt_1, axis=2)))

    train_filt_2 = apply_CCACSP(train_data_2, ccacsp_filts, numFilt)
    train_logP_2  = np.squeeze(np.log(np.var(train_filt_2, axis=2)))

    # define the classifier
    clf = sklda.LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    
    X = np.concatenate((train_logP_1, train_logP_2), axis=0)

    y1 = np.zeros(numTrials_1)
    y2 = np.ones(numTrials_2)
    y = np.concatenate((y1, y2))

    # train the classifier 
    clf.fit(X, y)

    return ccacsp_filts, clf


def test(test_data, ccacsp_filts, clf):

    total_filts = np.size(ccacsp_filts,1)

    # test the classifier on the test data
    test_filt = np.matmul(ccacsp_filts.transpose(), test_data)
    test_logP  = np.squeeze(np.log(np.var(test_filt, axis=1)))
    test_logP = np.reshape(test_logP,(1,total_filts))

    return clf.predict(test_logP)


def calc_CCACSP(x1,x2, numFilt):
    
    
    num_trials_1 = np.size(x1,0) 
    num_trials_2 = np.size(x2,0) 

    # number of channels and time samples should be the same between x1 and x2
    n_samps = np.size(x1,2)
    n_chans = np.size(x1,1) 

    c1_shifted = np.zeros([n_chans,n_chans])
    c2_shifted = np.zeros([n_chans,n_chans])
    c1 = np.zeros([n_chans,n_chans])
    c2 = np.zeros([n_chans,n_chans])

    range0 = range(0,n_samps-2)
    range1 = range(1,n_samps-1)
    range2 = range(2,n_samps)

    # estimate the covariances 
    for ik in range(num_trials_1):
        Samp = x1[ik]
        temp1 = 0.5*(Samp[:,range0]+Samp[:,range2])
        temp2 = Samp[:,range1]

        c1_shifted = c1_shifted+ my_cov(temp2, temp1)/np.trace(my_cov(temp2, temp1))
        c1 = c1+np.cov(x1[ik])/np.trace(np.cov(x1[ik]))

    c1_shifted = np.divide(c1_shifted,num_trials_1)
    c1 = np.divide(c1,num_trials_1)

    for ik in range(num_trials_2):
        Samp = x2[ik]
        temp1 = 0.5*(Samp[:,range0]+Samp[:,range2])
        temp2 = Samp[:,range1]
	
        c2_shifted = c2_shifted+ my_cov(temp2, temp1)/np.trace(my_cov(temp2, temp1))
        c2 = c2+np.cov(x2[ik])/np.trace(np.cov(x2[ik]))

    c2_shifted = np.divide(c2_shifted,num_trials_2)
    c2 = np.divide(c2,num_trials_2)
        

    # taking care of rank deficiency for a more robust result 
    D, V = sp.linalg.eigh(c1+c2) 
    indx = np.argsort(D)
    indx = indx[::-1]
    d = D[indx[0:np.linalg.matrix_rank(c1+c2)]]
    W = V[:,indx[0:np.linalg.matrix_rank(c1+c2)]]
    W_T = np.matmul(np.sqrt(sp.linalg.pinv(np.diag(d))),W.transpose())

    S1 = np.matmul(np.matmul(W_T,c1),W_T.transpose())
    S2 = np.matmul(np.matmul(W_T,c2),W_T.transpose())
    S1_shifted = np.matmul(np.matmul(W_T,c1_shifted),W_T.transpose())
    S2_shifted = np.matmul(np.matmul(W_T,c2_shifted),W_T.transpose())

    # find filters for class 1
    d,v = sp.linalg.eigh(S1_shifted,S1+S2)
    indx = np.argsort(d)
    indx = indx[::-1]
    filts_1 = v.take(indx, axis=1)
    filts_1 = np.matmul(filts_1.transpose(),W_T)
    filts_1 = filts_1.transpose()
    filts_1 = select_filts(filts_1, numFilt)

    # find filters for class 2
    d,v = sp.linalg.eigh(S2_shifted,S1+S2)
    indx = np.argsort(d)
    indx = indx[::-1]
    filts_2 = v.take(indx, axis=1)
    filts_2 = np.matmul(filts_2.transpose(),W_T)
    filts_2 = filts_2.transpose()
    filts_2 = select_filts(filts_2, numFilt)

    # concatenate filters for classes 1 and 2 and return 
    return np.concatenate((filts_1, filts_2), axis=1)

def select_filts(filt, col_num):

    temp = np.shape(filt)
    columns = np.arange(0,col_num)
    #print(columns)
    f = filt[:, columns]
    for ij in range(col_num):
        f[:, ij] = f[:, ij]/np.linalg.norm(f[:, ij])

    return f


def apply_CCACSP(X, f, col_num):

    f = np.transpose(f)

    temp = np.shape(X)
    num_trials = temp[0]

    #dat = np.zeros(np.shape(X), dtype = object)
    dat = np.zeros((num_trials, 2*col_num, temp[2]))
    for ik in range(num_trials):
        dat[ik,:,:] = np.matmul(f,X[ik,:,:])

    return dat

def my_cov(X, Y):
	avg_X = np.mean(X, axis=1)
	avg_Y = np.mean(Y, axis=1)

	X = X - avg_X[:,None]
	Y = Y - avg_Y[:,None]

	return np.matmul(X, Y.transpose())
