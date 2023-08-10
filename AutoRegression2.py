import os
import sys
import numpy as np
from scipy import special, integrate
from tqdm import tqdm
import time

rng = np.random.default_rng(11)

#p_parameters = [int(sys.argv[1])]
#train_pers = int(sys.argv[2])

DIRNAME = os.path.abspath("")
PLOTS_PATH = os.path.join(DIRNAME, "Plots")
DATA_DIR = os.path.join(os.path.dirname(DIRNAME), "FINO1Data")
ARRAY_PATH = os.path.join(DIRNAME, "Arrays")

os.makedirs(ARRAY_PATH, exist_ok=True)
os.makedirs(PLOTS_PATH, exist_ok=True)
datasets = {path.replace(".npy", ""): np.load(os.path.join(DATA_DIR, path), allow_pickle=True) for path in os.listdir(DATA_DIR) if path.endswith("npy")}


# Differentiating
datasets = {key: dataset[1:] - dataset[:-1] for key, dataset in datasets.items()}  

dname = "2015-2017_100m"                                                             
dataset = datasets[dname]
dataset[dataset>5] = np.nan  


class AR():

    def __init__(self, *, p, train_per, gust_th=1.5, beta=1):

        self.p = p
        self.train_per = train_per
        self.gust_th = gust_th ** beta
        self.beta = beta
        

    def get_rolling(self, *, data, mode, show):

        # Additional chunking
        indices = np.arange(len(data)-self.train_per+1)
        n_splits = max(int(len(data) / 2000), 1)
        indices_split = np.array_split(indices, n_splits)

        rolling = []
        for indices in tqdm(indices_split, disable=not(show)):
            indices = indices[:, None] + np.arange(self.train_per)
            x = data[indices]
            if mode == "var":
                split_rolling = np.nanvar(x, axis=1)
            elif mode == "mean":
                split_rolling = np.nanmean(x, axis=1) 
            rolling.append(split_rolling)
        return np.concatenate(rolling)


    def get_phis(self, *, data):

        def get_yule_walker(*, acfs):
            matrices = acfs[matrix_mask, :].reshape(self.p,self.p,-1).T
            det_mask = np.linalg.det(matrices)
            matrices[det_mask<1e-6] = rng.normal(size=matrices[0].shape)*0.001
            return np.linalg.solve(matrices, acfs[1:].T)
        
        # TODO Implement substract mean Steinerscher Verschiebungssatz

        z = [np.array(data[lag:] * data[:len(data)-lag])[:len(data)-self.p] for lag in range(0, self.p+1)]
        mean_gammas = [self.get_rolling(data=gamma, mode="mean", show=False) for gamma in z]
        acfs = [np.divide(gamma, mean_gammas[0], out=np.zeros_like(gamma), where=mean_gammas[0]!=0) for gamma in mean_gammas]
        acfs = np.stack(acfs)
        matrix_mask = np.array([abs(-k+j) for k in range(self.p) for j in range(self.p)])
        splits = max(int(len(acfs[0]) / 2000), 1)
        phis = [get_yule_walker(acfs=acfs_split) for acfs_split in np.array_split(acfs, splits, axis=1)]
        return np.concatenate(phis), np.array(mean_gammas)
    

    def forecast(self, *, data, phis, var_w):

        indices = np.arange(self.p+self.train_per-1, len(data))[:, None] - np.arange(self.p)
        predictions = data[indices] * phis
        predictions = predictions.sum(axis=1)
        white_noise = rng.normal(scale=var_w)
        predictions += white_noise
        return predictions
    

    def get_integrals(self, *, predictions, var_w):
        
        z = (self.gust_th - predictions) / (2*var_w)**0.5
        predictions = 0.5 * (1 - special.erf(z))
        return np.where(var_w==0, 0, predictions)




def data_chunks(*, ar, data):

    chunk_size = 1000000
    n_chunks = max(int(len(data)/chunk_size), 1)
    indices = np.arange(len(data)) # t-1
    indices_splits = np.array_split(indices, n_chunks)   
    indices_splits[:-1] = [np.append(indices_split, indices_split[-1] + np.arange(1, ar.train_per+ar.p)) for indices_split in indices_splits[:-1]]

    predictions = []

    for indices_split in tqdm(indices_splits):
        data_chunk = data[indices_split]
        phis, mean_gammas = ar.get_phis(data=data_chunk)

        # course-corrected variance
        var_w_chunk = mean_gammas[0] - (phis * mean_gammas[1:].T).sum(axis=1)
        var_w_chunk[var_w_chunk<0] = 0.001
        
        prediction = ar.forecast(data=data_chunk, phis=phis, var_w=var_w_chunk)
        # predictions.append(prediction)
        predictions.append(ar.get_integrals(predictions=prediction, var_w=var_w_chunk)) 
    return np.concatenate(predictions)


def get_performances(*, data, predictions, p, train_per, gust_th):


    quantiles = np.append([0], 10**np.arange(-9, 0.1, 0.1))
    # quantiles = np.linspace(0,0.5,50)
    wind_nans = np.isnan(data[train_per+p:])
    wind_gusts = np.where(data[train_per+p:] >= gust_th, True, False) # G(t)
    
    wind_gusts[wind_nans]          = np.nan 
    predictions_nanidx             = np.where(np.isnan(predictions))
    wind_nans_idx                  = np.where(wind_nans)

    all_nan_idx = np.unique(np.append(wind_nans_idx, predictions_nanidx))
    predictions = np.delete(predictions, all_nan_idx)
    wind_gusts = np.delete(wind_gusts, all_nan_idx)

    tpr, fpr, prob_threshold, distances, auc = ROC(gustprob=predictions, prob_threshold=quantiles, isgustbool=wind_gusts, printing=True)
    return tpr, fpr, prob_threshold, distances, auc




def save_performances(*, ar, tpr, fpr, prob_threshold, distances, auc, mode):
    """
    This is my first doc string! 
    """
    PERFORMANCE_PATH = os.path.join(ARRAY_PATH, f"{dname}", f"AutoRegression2", f"AR({ar.p})", f"GustTh={ar.gust_th}", f"Beta={ar.beta}")
    os.makedirs(PERFORMANCE_PATH, exist_ok=True)
    ROCsave = np.column_stack([tpr, fpr]) 
    np.savetxt(os.path.join(PERFORMANCE_PATH, f"{dname}-p={ar.p}-training_per={ar.train_per}-gust_th={ar.gust_th}-beta={ar.beta}-mode={mode}.dat"), ROCsave)
    return


def distanceNorm(x, y):  # normiert auf 1
    d       = (y - x)
    return d 


def ROC(gustprob, prob_threshold, isgustbool, printing=True):
### iterate through all discrimination thresholds (probability thresholds)
    tpr          = np.zeros(len(prob_threshold), dtype=float)   # initialize array for sensitivity [same as true positive rate (TPR)]; number of elements equal to number of discrimination thresholds
    fpr          = np.zeros(len(prob_threshold), dtype=float)   # and for specifity [false positive rate (FPR) = 1 - specifity]
    for ithreshold in range(len(prob_threshold)):
    ### Indices of gust alarms: Where do the gust probabilities "gustprob" exceed the probability threshold "prob_threshold"?
        alert_idx                  = np.ravel(np.where(gustprob >= prob_threshold[ithreshold]))
    ### Create boolean array with true, if gust alarm
        predictionalert            = np.array(np.zeros(len(isgustbool)), dtype=bool)     
        predictionalert[alert_idx] = True 
    ### Indices, where no gust alarm is
        nonalert_idx               = np.ravel(np.where(~predictionalert))
    ### Calculate the number of true positives (there is a gust, and also a gust is predicted)
        right_alert                = np.count_nonzero(isgustbool[alert_idx])  # count the number of true positives
    ### Calculate the number of true negatives (there is no gust, and no gust has been predicted)
        right_nonalert             = np.count_nonzero(~isgustbool[nonalert_idx])
    ### Total number of all gust events in "dv" ("gustpositive") and of no-gust-events ("gustnegative")
        gustpositive               = np.count_nonzero(isgustbool) 
        gustnegative               = np.count_nonzero(~isgustbool)
        tpr[ithreshold]    = right_alert/gustpositive            # TPR = True positive rate  = TP/P     =     (True positive)/(Total gust number)
        fpr[ithreshold]    = 1 - right_nonalert/gustnegative     # FPR = False positive rate = 1 - TN/N = 1 - (True negative)/(Total not-gust number)
    # printing is optional of course
        if printing:
            print("\nProbability threshold: " + str(prob_threshold[ithreshold]))  
            print("True positive rate: " +  str(100*np.round(tpr[ithreshold], 2)) + "%")
            print("False positive rate: " + str(100*np.round(fpr[ithreshold], 2)) + "%")  
    # Estimate the distance between the linear line and all [TPR, FPR] points in the ROC plot. There are as much [TPR, FPR] points as "prob_threshold" values exist
    distances            = np.zeros(len(prob_threshold), dtype=float)
    for idist in range(len(distances)):
        distances[idist]      = distanceNorm(fpr[idist], tpr[idist])
    idxmaxdist           = np.ravel(np.where(distances == np.max(distances)))       # find the index of the maximum distance "idxmaxdist" (according to the probability threshold with best predictive power)
    if printing:
        print('\nBest probability threshold: ' + str(prob_threshold[idxmaxdist]) + '\n')
    integy  = np.flip(tpr)
    integx  = np.flip(fpr)
    # auc     = 2*integrate.trapezoid(y=integy-integx, x=integx)#[-1]
    auc     = integrate.trapezoid(y=integy, x=integx)#[-1]
    print(f"AUC: {auc}")
    return tpr, fpr, prob_threshold[idxmaxdist], distances[idxmaxdist], auc


#----------------------------------------------------------------------------------------------------------------#

gust_ths = [1,1.5,2]
# train_pers = np.unique(np.round(10**np.linspace(np.log10(5), np.log10(10*24*60*60), 200))).astype(int)
train_pers = [1242]
p_parameters = [1,2,4,10]
betas = [1] # 0.644
mode = "c"

n = len(gust_ths) * len(train_pers) * len(p_parameters) * len(betas)
i = 0

for beta in betas:
    dataset_gauss = np.sign(dataset) * np.abs(dataset)**beta

    for train_per in train_pers:

        for p in p_parameters:
            ar = AR(p=p, train_per=train_per, gust_th=1, beta=beta)
            predictions = data_chunks(ar=ar, data=dataset_gauss[:-1])

            for gust_th in gust_ths:
                ar.gust_th = gust_th
                
                i += 1
                start = time.time()
                print(f"\n({i}/{n}): Getting AR({vars(ar)}) performances:")
                tpr, fpr, prob_threshold, distances, auc = get_performances(data=dataset, predictions=predictions, p=ar.p, train_per=ar.train_per, gust_th=ar.gust_th)
                
                save_performances(ar=ar, tpr=tpr, fpr=fpr, prob_threshold=prob_threshold, distances=distances, auc=auc, mode=mode)
                print(f"AR({ar.p}), {vars(ar)}, took {np.round(time.time()-start, 1)} secs.")