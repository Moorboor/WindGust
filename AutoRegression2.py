import os

import numpy as np
from scipy import special
import pandas as pd

from tqdm import tqdm
rng = np.random.default_rng(11)
import time



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

    def __init__(self, *, p, train_per, gust_th=1.6, beta=1):

        self.p = p
        self.train_per = train_per
        self.gust_th = gust_th ** beta
        

    def get_rolling(self, *, data, mode, show):

        indices = np.arange(0, len(data)-self.train_per+1)
        n_splits = int(len(data) / 200000)+1
        indices_split = np.array_split(indices, n_splits)

        rolling = []
        for indices in tqdm(indices_split, disable=not(show)):
            indices = indices[:, None] + np.arange(self.train_per)
            x = data[indices]

            if mode == "std":
                split_rolling = x.std(axis=1)
            elif mode == "mean":
                split_rolling = x.mean(axis=1)

            nan_pos = np.isnan(split_rolling)
            # TODO CASE: n - n_nans = 1
            if np.any(nan_pos): 
                if mode == "std":
                    nan_rolling = np.apply_along_axis((
                        lambda data: data[~np.isnan(data)].std() if np.isnan(data).any() else 0), axis=1, arr=x[nan_pos, :]
                        )
                elif mode == "mean":
                    nan_rolling = np.apply_along_axis((
                        lambda data: data[~np.isnan(data)].mean() if np.isnan(data).any() else 0), axis=1, arr=x[nan_pos, :]
                        )

                split_rolling = np.nan_to_num(split_rolling, copy=False, nan=0.0)
                split_rolling[nan_pos] += nan_rolling
                
            rolling.append(split_rolling)
            
        return np.concatenate(rolling)

    
    def get_phis(self, *, data, stds):
        
        def get_yule_walker(*, acfs):
            matrices = acfs[matrix_mask, :].reshape(self.p,self.p,-1).T
            det_mask = np.linalg.det(matrices)
            matrices[det_mask==0] += np.eye(self.p) * 1e-4
            return np.linalg.solve(matrices, acfs[1:].T)
        
        # TODO Implement substract mean Steinerscher Verschiebungssatz

        z = [np.array(data[lag:] * data[:-lag])[:lag-self.p-1] for lag in range(1, self.p+1)]
        acfs = [self.get_rolling(data=z[lag], mode="mean", show=False) for lag in range(self.p)]
        acfs = [np.divide(acf, stds**2, out=np.zeros_like(acf), where=stds!= 0) for acf in acfs]
        acfs.insert(0, np.ones_like(acfs[0]))
        acfs = np.stack(acfs)

        matrix_mask = np.array([abs(-k+j) for k in range(self.p) for j in range(self.p)])
        splits = int(len(acfs[0]) / 200000)+1
        phis = [get_yule_walker(acfs=acfs_split) for acfs_split in np.array_split(acfs, splits, axis=1)]

        return np.concatenate(phis)
    

    def forecast(self, *, data, phis, stds):

        indices = np.arange(self.train_per-1, len(data))[:, None] - np.arange(self.p)
        x = data[indices]

        predictions = x[self.p:-1] * phis
        predictions = predictions.sum(axis=1)

        white_noise = rng.normal(scale=(stds**2))
        predictions += white_noise
            
        return predictions
    

    def get_integrals(self, *, predictions, stds):
        
        z = (self.gust_th - predictions) / ((2*stds**2)**0.5)
        return 0.5 * (1 - special.erf(z))
    



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
    PERFORMANCE_PATH = os.path.join(ARRAY_PATH, f"{dname}", f"AutoRegression2", f"{p}", f"{gust_th}", f"{beta}")
    os.makedirs(PERFORMANCE_PATH, exist_ok=True)

    return df.to_csv(os.path.join(PERFORMANCE_PATH, f"{dname}-p={p}-training_per={train_per}-gust_th={gust_th}-beta={beta}-xi={xi:.4f}.csv"))


def data_chunks(*, ar, data, stds):

    chunk_size = 5000000
    n_chunks = int(len(data)/chunk_size)+1
    data_indices = np.arange(len(data)-ar.train_per-ar.p)
    stds_indices = np.arange(ar.p, len(stds)-1)
    data_indices_splits = np.array_split(data_indices, n_chunks)
    stds_indices_splits = np.array_split(stds_indices, n_chunks)

    data_indices_splits = [np.append(indices_split, indices_split[-1] + np.arange(1, ar.train_per+ar.p+1)) for indices_split in data_indices_splits]
    predictions = []
    for data_indices_chunk, stds_indices_chunk in tqdm(zip(data_indices_splits, stds_indices_splits), total=len(data_indices_splits)):
        data_chunk = data[data_indices_chunk]
        stds_chunk = stds[stds_indices_chunk]

        phis = ar.get_phis(data=data_chunk, stds=stds_chunk)
        prediction = ar.forecast(data=data_chunk, phis=phis, stds=stds_chunk)
        predictions.append(ar.get_integrals(predictions=prediction, stds=stds_chunk)) 
        
    return np.concatenate(predictions)


dataset = dataset
gust_ths = [1.5]
train_pers = np.unique(np.round(10**np.linspace(np.log10(10), np.log10(60*60), 100))).astype(int)
train_pers = [20]
p_parameters = [20]
betas = [1] # 0.644

n = len(gust_ths) * len(train_pers) * len(p_parameters) * len(betas)
i = 0

start = time.time()

ar = AR(p=None, train_per=None, gust_th=1, beta=1)

for gust_th in gust_ths:
    ar.gust_th = gust_th
    for beta in betas:
        dataset_gauss = np.sign(dataset) * np.abs(dataset)**beta
        ar.beta = beta

        for train_per in train_pers:
            ar.train_per = train_per

            print(f"Getting rolling standard deviations tau=({train_per}): ")
            stds = ar.get_rolling(data=dataset, mode="std", show=True)

            for p in tqdm(p_parameters):

                ar.p = p 
                i += 1
                
                print(f"\n({i}/{n}): Getting AR({p}) phi parameters: ")
                predictions = data_chunks(ar=ar, data=dataset_gauss, stds=stds)
                performances = get_performances(data=dataset, predictions=predictions, p=p, train_per=train_per, gust_th=gust_th)
                xi = get_xi(performances=performances)
                
                save_performances(performances=performances, p=p,train_per=train_per, gust_th=gust_th, xi=xi, beta=beta)

end = time.time()
print(f"{np.round(end-start, 2)}")