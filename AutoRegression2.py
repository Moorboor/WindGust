import os

import numpy as np
from scipy import special
import pandas as pd

from tqdm import tqdm
rng = np.random.default_rng(11)




DIRNAME = os.path.abspath("")
PLOTS_PATH = os.path.join(DIRNAME, "Plots")
DATA_DIR = os.path.join(os.path.dirname(DIRNAME), "FINO1Data")
ARRAY_PATH = os.path.join(DIRNAME, "Arrays")

os.makedirs(ARRAY_PATH, exist_ok=True)
os.makedirs(PLOTS_PATH, exist_ok=True)
datasets = {path.replace(".npy", ""): np.load(os.path.join(DATA_DIR, path), allow_pickle=True) for path in os.listdir(DATA_DIR) if path.endswith("npy")}



datasets = {key: dataset[1:] - dataset[:-1] for key, dataset in datasets.items()}   # Differentiation to make data stationary
dname = "2015-2017_100m"                                                             # Active dataset
dataset = datasets[dname]



class AR():

    def __init__(self, *, p, train_per, gust_th=1.6, beta=1):

        self.p = p
        self.train_per = train_per
        self.gust_th = gust_th ** beta
        

    def get_rolling(self, *, data, mode):

        indices = np.arange(0, len(data)-self.train_per+1)
        splits = int(len(data) / 20000)+1
        indices_split = np.array_split(indices, splits)

        rolling = []

        for indices in tqdm(indices_split):
            indices = indices[:, None] + np.arange(self.train_per)
            x = data[indices]

            if mode == "std":
                split_rolling = x.std(axis=1)
            elif mode == "mean":
                split_rolling = x.mean(axis=1)

            nan_pos = np.isnan(split_rolling)
            if nan_pos.sum() != 0: 
                if mode == "std":
                    nan_rolling = np.apply_along_axis((lambda data: data[~np.isnan(data)].std() if np.isnan(data).sum()>0 else 0), 1, arr=x[nan_pos, :])
                elif mode == "mean":
                    nan_rolling = np.apply_along_axis((lambda data: data[~np.isnan(data)].mean() if np.isnan(data).sum()>0 else 0), 1, arr=x[nan_pos, :])

                split_rolling = np.nan_to_num(split_rolling, copy=False, nan=0.0)
                split_rolling[nan_pos] += nan_rolling
                
            rolling.append(split_rolling)
            
        return np.concatenate(rolling)

    
    def get_phis(self, *, data, stds):
        
        # TODO Implement substract mean
        #means = np.convolve(data, np.ones(train_per), "valid") / train_per
        #data = data[train_per: -train_per] - means

        z = [np.array(data[lag:] * data[:-lag])[:lag-self.p-1] for lag in range(1, self.p+1)]

        acfs = [self.get_rolling(data=z[lag], mode="mean") for lag in range(self.p)]
        acfs = [acf / (stds[self.p:-1] ** 2) for acf in acfs]

        acfs.insert(0, np.ones_like(acfs[0]))
        acfs = np.stack(acfs)
        
        matrix_mask = np.array([abs(-k+j) for k in range(self.p) for j in range(self.p)])
        matrices = acfs[matrix_mask, :].reshape(self.p,self.p,-1).T
        
        det_mask = np.linalg.det(matrices)
        matrices[det_mask==0] += np.eye(self.p) * 1e-4
        phis = np.linalg.solve(matrices, acfs[1:].T)

        return np.array(phis)
    

    def forecast(self, *, data, phis, stds):

        indices = np.arange(self.train_per-1, len(data))[:, None] - np.arange(self.p)
        x = data[indices]

        predictions = x[p:-1] * phis
        predictions = predictions.sum(axis=1)

        white_noise = rng.normal(scale=(stds[self.p:-1]**2))
        predictions += white_noise

        return predictions
    

    def get_integrals(self, *, predictions, stds):
        
        z = (self.gust_th - predictions) / ((2*stds[self.p:-1]**2)**0.5)
        prediction_integrals = 0.5 * (1 - special.erf(z))
        
        return prediction_integrals
    


def get_performances(*, data, prediction_integral, p, train_per, gust_th):


    quantiles = np.append([0], 10**np.arange(-9, (1.1), 0.1))
    performances = dict.fromkeys(quantiles)

    wind_nans = np.isnan(data[train_per+p:])
    wind_gusts = np.where(data[train_per+p:] >= gust_th, 3, 0) # G(t)
    n_gusts = wind_gusts.sum() / 3
    for quantile in tqdm(quantiles):

        wind_gusts_pred = np.where(prediction_integral >= quantile, 1, 0) 

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
    return df.to_csv(os.path.join(ARRAY_PATH, f"{dname}-p={p}-training_per={train_per}-gust_th={gust_th}-beta={beta}-xi={xi:.4f}.csv"))




gust_ths = [1.5]
train_pers = np.unique(np.round(10**np.linspace(np.log10(10), np.log10(60*60), 100))).astype(int)
train_pers = train_pers[1:3]
p_parameters = [3,4]
betas = [0.644, 1]


ar = AR(p=None, train_per=None, gust_th=1, beta=1)

for gust_th in gust_ths:
    ar.gust_th = gust_th
    for beta in betas:
        dataset_gauss = np.sign(dataset) * np.abs(dataset)**beta
        ar.beta = beta

        for train_per in train_pers:
            ar.train_per = train_per

            print(f"Getting rolling standard deviations tau=({train_per}): ")
            stds = ar.get_rolling(data=dataset, mode="std")

            for p in tqdm(p_parameters):

                ar.p = p 
                print(f"\nGetting AR({p}) phi parameters: ")
                phis = ar.get_phis(data=dataset, stds=stds)

                predictions = ar.forecast(data=dataset_gauss, phis=phis, stds=stds)
                prediction_integrals = ar.get_integrals(predictions=predictions, stds=stds)

                performances = get_performances(data=dataset, prediction_integral=prediction_integrals, p=p, train_per=train_per, gust_th=gust_th)
                xi = get_xi(performances=performances)

                
                save_performances(performances=performances, p=p,train_per=train_per, gust_th=gust_th, xi=xi, beta=beta)
                