from data import chris_data as data_maker

def get_sets(tot_dataset_size=int(2**20),ndata=16,usepars=[0,1,2],sigma=0.2,seed=42,r=4):

    # get training set data
    pos_train, labels_train, x, sig_train, parnames = data_maker.generate(
                tot_dataset_size=tot_dataset_size,
                ndata=ndata,
                usepars=usepars,
                sigma=sigma,
                seed=seed
            )
    print('generated training data')

    # get test set data
    pos_test, labels_test, x, sig_test, parnames = data_maker.generate(
                tot_dataset_size=r*r,
                ndata=ndata,
                usepars=usepars,
                sigma=sigma,
                seed=seed
            )
    print('generated testing data')

    return pos_train,sig_train,labels_train,pos_test