import os

import numpy as np
from scipy import special

import pandas as pd
from tqdm import tqdm
import time

rng = np.random.default_rng(11)



DIRNAME = os.path.abspath("")
PLOTS_PATH = os.path.join(DIRNAME, "Plots")
DATA_DIR = os.path.join(os.path.dirname(DIRNAME), "FINO1Data")
ARRAY_PATH = os.path.join(DIRNAME, "Arrays")

os.makedirs(ARRAY_PATH, exist_ok=True)
os.makedirs(PLOTS_PATH, exist_ok=True)
datasets = {path.replace(".npy", ""): np.load(os.path.join(DATA_DIR, path), allow_pickle=True) for path in os.listdir(DATA_DIR) if path.endswith("npy")}



datasets = {key: dataset[1:] - dataset[:-1] for key, dataset in datasets.items()}  
dname = "2015-2017_100m"                                                             
dataset = datasets[dname]



class AR():

    def __init__(self, *, p, train_per, gust_th=1.5, beta=1):

        self.p = p
        self.train_per = train_per
        self.gust_th = gust_th ** beta
        self.beta = beta
        

    def get_rolling(self, *, data, mode, show):

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

    
    def get_phis(self, *, data, var):
        
        def get_yule_walker(*, acfs):
            matrices = acfs[matrix_mask, :].reshape(self.p,self.p,-1).T
            det_mask = np.linalg.det(matrices)
            matrices[det_mask==0] += np.eye(self.p) * 1e-4
            return np.linalg.solve(matrices, acfs[1:].T)
        
        # TODO Implement substract mean Steinerscher Verschiebungssatz

        z = [np.array(data[lag:] * data[:-lag])[:len(data)-self.p] for lag in range(1, self.p+1)]
        acfs = [self.get_rolling(data=z[lag], mode="mean", show=False) for lag in range(self.p)]
        acfs = [np.divide(acf, var, out=np.zeros_like(acf), where=var!=0) for acf in acfs]
        acfs.insert(0, np.ones_like(acfs[0]))
        acfs = np.stack(acfs)
        matrix_mask = np.array([abs(-k+j) for k in range(self.p) for j in range(self.p)])
        splits = max(int(len(acfs[0]) / 200000), 1)
        phis = [get_yule_walker(acfs=acfs_split) for acfs_split in np.array_split(acfs, splits, axis=1)]

        return np.concatenate(phis)
    

    def forecast(self, *, data, phis, var):

        indices = np.arange(self.p+self.train_per-1, len(data))[:, None] - np.arange(self.p)
        predictions = data[indices] * phis
        predictions = predictions.sum(axis=1)

        white_noise = rng.normal(scale=var)
        predictions += white_noise
            
        return predictions
    

    def get_integrals(self, *, predictions, var):
        
        z = (self.gust_th - predictions) / (2*var)**0.5
        predictions = 0.5 * (1 - special.erf(z))
        return np.where(var==0, 0, predictions)


def data_chunks(*, ar, data, var):

    chunk_size = 1000000
    n_chunks = max(int(len(data)/chunk_size), 1)
    indices = np.arange(len(data)) # t-1
    indices_splits = np.array_split(indices, n_chunks)   
    var_indices_splits = indices_splits.copy()
    var_indices_splits[-1] = var_indices_splits[-1][:-ar.train_per-ar.p+1]
    indices_splits[:-1] = [np.append(indices_split, indices_split[-1] + np.arange(1, ar.train_per+ar.p)) for indices_split in indices_splits[:-1]]
    predictions = []
    for indices_split, var_indices_split in tqdm(zip(indices_splits, var_indices_splits), total=len(indices_splits)):
        data_chunk = data[indices_split]
        var_chunk = var[var_indices_split]
        phis = ar.get_phis(data=data_chunk, var=var_chunk)

        prediction = ar.forecast(data=data_chunk, phis=phis, var=var_chunk)
        predictions.append(ar.get_integrals(predictions=prediction, var=var_chunk)) 
        
    return np.concatenate(predictions)


def get_performances(*, data, predictions, p, train_per, gust_th):


    quantiles = np.append([0], 10**np.arange(-9, (1.1), 0.1))
    performances = dict.fromkeys(quantiles)

    wind_nans = np.isnan(data[train_per+p:])
    wind_gusts = np.where(data[train_per+p:] >= gust_th, 3, 0) # G(t)
    n_gusts = wind_gusts.sum() / 3

    for quantile in tqdm(quantiles):

        wind_gusts_pred = np.where(predictions >= quantile, 1, 0) 

        true_positive = np.where(wind_gusts_pred + wind_gusts == 4, 1, 0)
        false_positive = np.where(wind_gusts_pred + wind_gusts == 1, 1, 0)
        
        true_positive_rate = true_positive.sum() / n_gusts
        false_positive_rate = false_positive[~wind_nans].sum() / (len(wind_gusts) - n_gusts - wind_nans.sum())

        performances[quantile] = (true_positive_rate, false_positive_rate)

    return performances

def get_xi(*, performances):
    
    base_1 = np.array([.5,-.5])
    base_2 = np.array([.5, .5])

    points = dict()

    for key, (y,x) in performances.items():
        point = x * base_1 + y * base_2
        points[key] = point

    key = sorted(points.items(), key=lambda item:item[1][1])[-1][0]
    xi = np.array([[x,y] for key, (x,y) in points.items()])

    return np.max(xi[:, 1])

def save_performances(*, performances, p, train_per, gust_th, xi, beta):
    """
    This is my first doc string! 
    """
    quantiles = []
    true_positive_rates = []
    false_negative_rates = []

    for quantile, (true_positive_rate, false_negative_rate) in performances.items():
        quantiles.append(quantile)
        true_positive_rates.append(true_positive_rate)
        false_negative_rates.append(false_negative_rate)

    data = list(zip(quantiles, true_positive_rates, false_negative_rates))
    df = pd.DataFrame(data, columns=["quantiles", "true_positive_rate", "false_negative_rate"])
    PERFORMANCE_PATH = os.path.join(ARRAY_PATH, f"{dname}", f"AutoRegression2", f"AR({p})", f"GustTh={gust_th}", f"Beta={beta}")
    os.makedirs(PERFORMANCE_PATH, exist_ok=True)

    return df.to_csv(os.path.join(PERFORMANCE_PATH, f"{dname}-p={p}-training_per={train_per}-gust_th={gust_th}-beta={beta}-xi={xi:.4f}.csv"))


gust_ths = [1.5]
train_pers = np.unique(np.round(10**np.linspace(np.log10(10), np.log10(60*60), 100))).astype(int)
train_pers = train_pers[30:]
p_parameters = [2]
betas = [1] # 0.644

n = len(gust_ths) * len(train_pers) * len(p_parameters) * len(betas)
i = 0




for beta in betas:
    dataset_gauss = np.sign(dataset) * np.abs(dataset)**beta

    for train_per in train_pers:
        ar = AR(p=None, train_per=None, gust_th=1, beta=1)
        ar.beta = beta
        ar.train_per = train_per
        print(f"Getting rolling variances tau=({ar.train_per}): ")
        var = ar.get_rolling(data=dataset_gauss[:-1], mode="var", show=True)

        for p in p_parameters:
            ar.p = p 
            predictions = data_chunks(ar=ar, data=dataset_gauss[:-1], var=var[ar.p:])

            for gust_th in gust_ths:
                ar.gust_th = gust_th
                
                i += 1
                start = time.time()
                print(f"\n({i}/{n}): Getting AR({vars(ar)}) performances:")
                performances = get_performances(data=dataset, predictions=predictions, p=ar.p, train_per=ar.train_per, gust_th=ar.gust_th)
                xi = get_xi(performances=performances)
                save_performances(performances=performances, p=ar.p, train_per=ar.train_per, gust_th=ar.gust_th, xi=xi, beta=ar.beta)
                end = time.time()

                print(f"AR({ar.p}), {vars(ar)}, took {np.round(end-start, 1)} secs.")