import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy import integrate, special
# from tqdm import tqdm  

rng = np.random.default_rng(11)


DIRNAME = os.path.abspath("")
PLOTS_PATH = os.path.join(DIRNAME, "Plots")
DATA_DIR = os.path.join(os.path.dirname(DIRNAME), "FINO1Data")
ARRAY_PATH = os.path.join(DIRNAME, "Arrays")

os.makedirs(PLOTS_PATH, exist_ok=True)
datasets = {path.replace(".npy", ""): np.load(os.path.join(DATA_DIR, path), allow_pickle=True) for path in os.listdir(DATA_DIR) if path.endswith("npy")}


def lerp(data):
    ok = ~np.isnan(data)
    xp = ok.nonzero()[0]
    fp = data[ok]
    x = np.isnan(data).nonzero()[0]
    data[~ok] = np.interp(x, xp, fp)
    return data

def autocovariance(data, lag):
    cov = data[lag:] * data[:-lag]
    return np.nanmean(cov)


# datasets = {key: lerp(value) for key, value in datasets.items()}                    # linear interpolation
datasets = {key: dataset[1:] - dataset[:-1] for key, dataset in datasets.items()}     # Differentiation to make data stationary
dname = "2015-2017_100m"                                                              # Active dataset
dataset = datasets[dname]
dataset[dataset>5] = np.nan                                                           # Exclude Helicopter noise # 194 values



class AR():

    def __init__(self, p):

        self.p = p
        self.lags = range(1,p+1)
        self.phis = None
        self.var_w = None

    def get_phis(self, train_data):

        z = [autocovariance(train_data, lag) for lag in self.lags]      # Autocorrelation function
        acf = [gamma / np.nanvar(train_data) for gamma in z]
        acf.insert(0, 1.)   
        r = [acf[abs(-k+j)] for k in range(self.p) for j in range(self.p)]  # Yule-Walker Estimation with autocorrelation function
        acf.pop(0)
        r_matrix = (np.array(r).reshape(self.p, self.p))
        r_inv_matrix = np.linalg.inv(r_matrix)
        phi_hats = r_inv_matrix @ np.array(acf).reshape(self.p, 1)
        phi_hats = [phi[0] for phi in phi_hats]
        self.phis = phi_hats
        self.var_w = acf[0] - sum([phi*gamma for phi, gamma in zip(self.phis, z[1:])])


    def predict(self, test_data):
        predictions = np.convolve(self.phis, test_data, "valid")
        predictions += rng.normal(scale=self.var_w, size=len(predictions))
        return predictions
  
    def get_integrals(self, *, predictions, gust_th):
        
        z = (gust_th - predictions) / (2*self.var_w)**0.5
        predictions = 0.5 * (1 - special.erf(z))
        return np.where(self.var_w==0, 0, predictions)

    
    def get_performances(self, *, data, predictions, gust_th):

        quantiles = np.append([0], 10**np.arange(-9, 0.1, 0.1))
        # quantiles = np.linspace(0,0.2,50)
        wind_nans = np.isnan(data)
        wind_gusts = np.where(data >= gust_th, True, False) # G(t)
        
        wind_gusts[wind_nans]          = np.nan 
        predictions_nanidx             = np.where(np.isnan(predictions))
        wind_nans_idx                  = np.where(wind_nans)

        all_nan_idx = np.unique(np.append(wind_nans_idx, predictions_nanidx))
        predictions = np.delete(predictions, all_nan_idx)
        wind_gusts = np.delete(wind_gusts, all_nan_idx)

        tpr, fpr, prob_threshold, distances, auc = self.ROC(gustprob=predictions, prob_threshold=quantiles, isgustbool=wind_gusts, printing=True)
        return tpr, fpr, prob_threshold, distances, auc



    def distanceNorm(self, x, y):  # normiert auf 1
        d       = (y - x)
        return d 


    def ROC(self, gustprob, prob_threshold, isgustbool, printing=True):
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
            distances[idist]      = self.distanceNorm(fpr[idist], tpr[idist])
        idxmaxdist           = np.ravel(np.where(distances == np.max(distances)))       # find the index of the maximum distance "idxmaxdist" (according to the probability threshold with best predictive power)
        if printing:
            print('\nBest probability threshold: ' + str(prob_threshold[idxmaxdist]) + '\n')
        integy  = np.flip(tpr)
        integx  = np.flip(fpr)
        # auc     = 2*integrate.trapezoid(y=integy-integx, x=integx)#[-1]
        auc     = integrate.trapezoid(y=integy, x=integx)#[-1]
        print(f"AUC: {auc}")
        return tpr, fpr, prob_threshold[idxmaxdist], distances[idxmaxdist], auc


def save_performances(*, ar, tpr, fpr, prob_threshold, distances, auc, mode, train_per, gust_th):
    """
    This is my first doc string! 
    """
    PERFORMANCE_PATH = os.path.join(ARRAY_PATH, f"{dname}", f"AutoRegression2", f"AR({ar.p})", f"GustTh={gust_th}", f"Beta=1")
    os.makedirs(PERFORMANCE_PATH, exist_ok=True)
    ROCsave = np.column_stack([tpr, fpr]) 
    np.savetxt(os.path.join(PERFORMANCE_PATH, f"{dname}-p={ar.p}-training_per={train_per}-gust_th={gust_th}-beta=1-mode={mode}.dat"), ROCsave)
    return

def roc_plot(performances, colorname, p):

    _ = plt.figure(1, figsize=(5,5))
    colors = sns.color_palette(colorname, len(performances))

    i = 0
    for threshold, (true_positive_rate, false_positive_rate) in performances.items():
        plt.plot(false_positive_rate, true_positive_rate, "o", label=f"{threshold:.2f}", color=colors[i])
        i += 1
    plt.plot(np.arange(2), linestyle="--", color='#0f0f0f30')


    plt.grid(linewidth=0.4, alpha=0.8)
    plt.xlim(0,1)
    plt.ylim(0,1)

    plt.minorticks_on()
    plt.tick_params(direction="in", which="major", length=7, bottom=True, top=True, right=True)
    plt.tick_params(direction="in", which="minor", length=2.5, bottom=True, top=True, right=True)
    plt.xlabel("False positive rate", fontsize=14)
    plt.ylabel("True positive rate", fontsize=14)

    plt.title(f"$AR({p})$", fontsize=14)
    plt.suptitle("Receiver-operating-characteristic", fontsize=16)
    plt.legend(title="Threshold (m/s)", fontsize=10, bbox_to_anchor=(1.02,1.1))

    #plt.savefig(os.path.join(PLOTS_PATH, f"{dname}-AR_overview.png"), format="png", dpi=300, bbox_inches="tight")


# test_set_size = min(3000000, int(len(dataset) * 0.5))
# train_data = dataset[:-test_set_size]
#test_data = dataset[-test_set_size:]

train_data = dataset[7750000:7750500]
test_data = dataset[7750500:]



p_list = [1,2,4]
gust_ths = [1,1.5,2]
mode = "b"

for p in p_list:
    ar = AR(p)
    ar.get_phis(train_data)
    predictions = ar.predict(test_data[:-1])

    for gust_th in gust_ths:
        predictions_int = ar.get_integrals(predictions=predictions, gust_th=gust_th)
        tpr, fpr, prob_threshold, distances, auc = ar.get_performances(data=test_data[p:], predictions=predictions_int, gust_th=gust_th)
        save_performances(ar=ar, tpr=tpr, fpr=fpr, prob_threshold=prob_threshold, distances=distances, auc=auc, mode=mode, train_per=len(train_data), gust_th=gust_th)




# color_palette = ["rocket_r", "dark:salmon_r", "dark:b_r", "dark:seagreen_r"]
# roc_plot(performances, color_palette[0], 2)